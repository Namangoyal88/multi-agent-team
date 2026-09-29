import httpx
import pytest

from research_team.config import GROQ, NVIDIA, load_settings
from research_team.experiments.sandbox import ExecutionDisabled, run_experiment, validate_code
from research_team.llm.client import LLMClient, LLMError
from research_team.llm.router import EscalationUnavailable
from research_team.memory import MemoryLayer
from research_team.tools.adapters import ArxivAdapter, WebSearchAdapter
from research_team.tools.toolkit import ResearchToolkit
from research_team.verification import verify_claims
from research_team.workspace import Workspace
from tests.conftest import ATOM, tool_handler


# ---------- routing ----------
def test_agent_model_routing_is_explicit(svc, llm):
    table = {r["agent"]: r for r in svc.router.routing_table()}
    assert table["researcher"]["model"] == "openai/gpt-oss-20b" and table["researcher"]["provider"] == GROQ
    assert table["mathematician"]["model"] == "openai/gpt-oss-120b"
    assert table["ml_engineer"]["model"] == "openai/gpt-oss-120b"
    assert table["experimenter"]["model"] == "openai/gpt-oss-120b"
    assert table["reviewer"]["model"] == "openai/gpt-oss-20b"
    assert table["verifier"]["model"] == "openai/gpt-oss-20b"
    assert table["report_generator"]["model"] == "qwen/qwen3.8-27b"
    assert table["director"]["provider"] == NVIDIA
    assert [r["agent"] for r in table.values() if r["provider"] == NVIDIA] == ["director"]
    svc.router.call("mathematician", [{"role": "user", "content": "hi"}])
    assert llm.calls[-1][:2] == (GROQ, "openai/gpt-oss-120b")


def test_nvidia_unavailable_raises_clean_error(make_svc):
    with pytest.raises(EscalationUnavailable):
        make_svc(nvidia=False).router.escalate([{"role": "user", "content": "x"}])


# ---------- llm client reliability ----------
def _client(handler, retries=2):
    cfg = load_settings({"GROQ_API_KEY": "k"}, dotenv=False).groq
    cfg = type(cfg)(cfg.name, cfg.base_url, "k", 5.0, retries)
    return LLMClient(cfg, transport=httpx.MockTransport(handler), sleep=lambda s: None)


def test_retries_429_then_succeeds():
    n = {"i": 0}

    def h(req):
        n["i"] += 1
        if n["i"] < 3:
            return httpx.Response(429, headers={"retry-after": "1"}, text="slow")
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>x</think>ok"}}]})
    c = _client(h)
    r = c.chat("m", [{"role": "user", "content": "q"}])
    assert r.content == "ok" and r.attempts == 3 and c.diagnostics["retries"] == 2


def test_5xx_exhausts_and_raises_with_diagnostics():
    c = _client(lambda r: httpx.Response(503, text="down"))
    with pytest.raises(LLMError) as e:
        c.chat("m", [])
    assert e.value.retryable and c.diagnostics["failures"] == 1 and "503" in c.diagnostics["last_error"]


def test_client_error_is_not_retried_and_timeout_handled():
    calls = {"n": 0}

    def h(r):
        calls["n"] += 1
        return httpx.Response(401, text="bad key")
    with pytest.raises(LLMError):
        _client(h).chat("m", [])
    assert calls["n"] == 1

    def boom(r):
        raise httpx.ReadTimeout("t", request=r)
    with pytest.raises(LLMError):
        _client(boom).chat("m", [])


# ---------- research tools ----------
def test_arxiv_parses_and_failures_are_never_silent():
    ok = ArxivAdapter(transport=httpx.MockTransport(tool_handler()), sleep=lambda s: None).search("rag")
    assert ok.ok and len(ok.items) == 2 and ok.items[0]["arxiv_id"].startswith("2401.")
    bad = ArxivAdapter(transport=httpx.MockTransport(tool_handler({"arxiv"})), sleep=lambda s: None).search("rag")
    assert not bad.ok and bad.error and bad.status_code == 500
    unconfigured = WebSearchAdapter({}).search("x")
    assert not unconfigured.ok and unconfigured.note == "not_configured"


def test_literature_search_reports_per_tool_errors():
    tk = ResearchToolkit({}, transport=httpx.MockTransport(tool_handler({"semanticscholar"})), sleep=lambda s: None)
    papers, status = tk.literature_search(["rag"])
    assert papers and status["semantic_scholar"]["errors"] and status["arxiv"]["results"] == 2


# ---------- verification ----------
def _papers():
    return [{"pid": "P1", "title": "T", "abstract": "Agentic retrieval reduces hallucination rates in QA."}]


def test_verification_never_auto_approves(svc, llm):
    claims = [{"id": "C1", "text": "Agentic retrieval reduces hallucination rates", "citations": ["P1"]}, {"id": "C2", "text": "y", "citations": ["P9"]},
              {"id": "C3", "text": "z", "citations": []}]
    r = {c["id"]: c for c in verify_claims(claims, _papers(), svc.router)}
    assert r["C1"]["status"] == "SUPPORTED"          # verbatim quote exists in abstract
    assert r["C2"]["status"] == "UNSUPPORTED"        # fabricated citation
    assert r["C3"]["status"] == "UNVERIFIED"         # no evidence


def test_supported_without_real_quote_is_downgraded_and_llm_failure_is_unverified(svc, llm):
    p = [{"pid": "P1", "title": "T", "abstract": "Completely different text."}]
    r = verify_claims([{"id": "C1", "text": "Agentic retrieval reduces hallucination rates", "citations": ["P1"]}], p, svc.router)
    assert r[0]["status"] == "UNVERIFIED"
    llm.fail_groq = True
    r = verify_claims([{"id": "C1", "text": "The retrieval method lowers hallucinations in QA", "citations": ["P1"]}], _papers(), svc.router)
    assert r[0]["status"] == "UNVERIFIED" and "unavailable" in r[0]["rationale"]


# ---------- sandbox ----------
def test_sandbox_disabled_validated_isolated():
    with pytest.raises(ExecutionDisabled):
        run_experiment("print(1)", enabled=False)
    assert validate_code("import os\nos.system('ls')")
    assert validate_code("import subprocess")
    assert run_experiment("open('/etc/passwd')", enabled=True)["status"] == "rejected"
    ok = run_experiment("print('hi')\nprint('METRICS_JSON: {\"acc\": 0.5}')", enabled=True)
    assert ok["status"] == "ok" and ok["metrics"] == {"acc": 0.5} and "hi" in ok["stdout"]
    assert run_experiment("while True: pass", enabled=True, timeout=1)["status"] == "timeout"
    assert run_experiment("raise ValueError('x')", enabled=True)["status"] == "error"


# ---------- workspace / memory ----------
def test_state_persists_across_reopen(tmp_path):
    db = str(tmp_path / "t.db")
    ws = Workspace(db)
    jid = ws.create_job({"question": "q"})
    ws.save_state(jid, {"done": ["plan"]})
    ws.add_papers(jid, [{"title": "A", "source": "arxiv"}, {"title": "a", "source": "arxiv"}])
    ws2 = Workspace(db)
    assert ws2.load_state(jid) == {"done": ["plan"]}
    assert [p["pid"] for p in ws2.papers(jid)] == ["P1"]   # de-duplicated


def test_vector_memory_semantic_recall():
    ws = Workspace(":memory:")
    m = MemoryLayer(ws)
    m.remember("j1", "finding", "Agentic retrieval reduces hallucination in question answering")
    m.remember("j1", "finding", "Protein folding with graph neural networks")
    top = m.recall("does retrieval reduce hallucination", 1)
    assert "hallucination" in top[0]["text"]
    m.add_edge("Paper", "authored_by", "Ada", "j1")
    assert m.neighbors("Ada")


def test_extract_json_handles_fences_inside_and_around_json():
    from research_team.llm.structured import extract_json
    inner = '{"content": "Design.\\n```python\\nprint(1)\\n```", "confidence": 0.9}'
    assert extract_json(inner)["confidence"] == 0.9
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": [1, 2]} thanks') == {"a": [1, 2]}
