import re
import threading
import time

import httpx
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from frontend.api_client import ResearchAPI
from research_team.agents.deep_director import DeepDirector, heuristic_plan
from research_team.api import create_app
from research_team.config import GROQ, NVIDIA, load_settings
from research_team.jobs import JobManager
from research_team.reporting import sanitize_citations
from tests.conftest import StubDirector

Q = "Investigate whether agentic RAG can reduce hallucinations compared with conventional RAG."


def run(svc, stages, question=Q, **opts):
    svc.director = StubDirector(stages)
    jm = JobManager(svc)
    jid = jm.start(question, opts)
    jm.wait(jid, 60)
    return jid, svc.ws.load_state(jid), svc.ws.get_job(jid)


def test_complex_workflow_runs_only_planned_stages_and_builds_full_report(svc, llm):
    jid, st, job = run(svc, ["literature", "analysis", "gaps", "math", "review"])
    assert job["status"] == "completed"
    assert st["done"] == ["literature", "analysis", "gaps", "math", "review", "verify", "report", "evaluate"]
    assert "ml_impl" not in st["done"] and "experiment_design" not in st["done"]     # dynamically skipped
    agents = {r["agent"] for r in svc.ws.agent_runs(jid)}
    assert agents == {"researcher", "mathematician", "reviewer"}                      # not every agent
    report = st["report"]
    assert len(re.findall(r"^## \d+\. ", report, re.M)) == 24
    assert "No experiments were executed" in report
    assert "[P99]" not in report and "invalid-ref" not in report                      # fabricated cite never survives the report
    assert "Not enough evidence gathered." in report                                      # strict grounding gate
    assert "P1" in {p["pid"] for p in svc.ws.papers(jid)}


def test_simple_question_uses_lightweight_workflow(svc):
    jid, st, _ = run(svc, ["literature"], question="What is agentic RAG?")
    assert st["done"] == ["literature", "verify", "report", "evaluate"]
    assert {r["agent"] for r in svc.ws.agent_runs(jid)} == {"researcher"}


def test_verification_uses_real_claims_and_state_persisted(svc):
    jid, st, _ = run(svc, ["literature"])
    assert st["verification"] and all(v["status"] in {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED",
                                                      "CONTRADICTED", "UNVERIFIED"} for v in st["verification"])
    assert svc.ws.get_artifact(jid, "verification", "claims")
    assert svc.ws.get_artifact(jid, "report", "final.md")


def test_low_confidence_specialist_escalates_to_nvidia(svc, llm):
    llm.specialist_overrides["You are the Researcher"] = {"confidence": 0.1, "needs_escalation": True,
                                                          "escalation_reason": "conflicting evidence"}
    jid, st, _ = run(svc, ["literature"])
    assert any(c[0] == NVIDIA for c in llm.calls)
    assert any(r["escalated"] for r in svc.ws.agent_runs(jid))
    # non-escalated specialists never touch NVIDIA
    assert all(c[0] == GROQ for c in llm.calls if "Verifier" in c[2] or "Report" in c[2])


def test_nvidia_down_workflow_continues(make_svc, llm):
    svc = make_svc(nvidia=False)                      # real DeepDirector, NVIDIA not configured
    llm.specialist_overrides["You are the Researcher"] = {"confidence": 0.1, "needs_escalation": True}
    jm = JobManager(svc)
    jid = jm.start(Q + " Review recent research and identify gaps.", {})
    jm.wait(jid, 60)
    st, job = svc.ws.load_state(jid), svc.ws.get_job(jid)
    assert job["status"] == "completed" and not any(c[0] == NVIDIA for c in llm.calls)
    assert st["plan_source"] == "heuristic_fallback"
    assert any("NVIDIA not configured" in e for e in st["errors"])
    assert any(e["kind"] == "escalation_unavailable" for e in svc.ws.events(jid))


def test_groq_outage_is_reported_not_hidden_and_nothing_fabricated(make_svc, llm):
    svc = make_svc(nvidia=False)
    llm.fail_groq = True
    jid, st, job = run(svc, ["literature", "analysis"])
    assert job["status"] == "completed"
    assert any(e.startswith("literature:") for e in st["errors"])
    assert not any(v["status"] == "SUPPORTED" for v in st["verification"])
    assert "Not performed" in st["report"] or "Report Generator unavailable" in st["report"]


def test_tool_failure_is_surfaced(make_svc):
    svc = make_svc(tool_fail={"arxiv", "semanticscholar"})
    jid, st, _ = run(svc, ["literature"])
    assert any("literature[arxiv]" in e for e in st["errors"])
    assert any("NO papers retrieved" in e for e in st["errors"])
    assert svc.ws.papers(jid) == []
    assert "No sources were retrieved" in st["report"]


def test_experiments_disabled_by_default_and_enabled_runs_real_code(make_svc, llm):
    svc = make_svc()
    _, st, _ = run(svc, ["literature", "experiment_design", "experiment_run"], enable_experiments=True)
    assert "experiment_run" not in st["done"] and any("disabled" in e for e in st["errors"])

    code = "print('METRICS_JSON: {\"acc\": 0.75}')"
    svc = make_svc(code_exec=True)
    llm.specialist_overrides["You are the ML Engineer"] = {"content": f"Design.\n```python\n{code}\n```"}
    jid, st, _ = run(svc, ["literature", "experiment_design", "experiment_run"], enable_experiments=True)
    assert st["experiment_results"]["metrics"] == {"acc": 0.75}
    assert '"acc": 0.75' in st["report"] or "'acc': 0.75" in st["report"]


def test_refine_loop_reruns_evidence_stages(make_svc):
    import httpx as hx
    from research_team.config import load_settings
    svc = make_svc()
    svc.settings.max_refine_iterations = 1
    svc.settings.min_papers = 99          # force "not enough papers"
    jid, st, _ = run(svc, ["literature"])
    assert st["iteration"] == 1 and st["evaluation"]["iteration"] == 1
    assert st["done"].count("literature") == 1 and st["done"][-1] == "evaluate"


def test_heuristic_planner_skips_stages():
    assert heuristic_plan("What is RAG?", {}).stages == ["literature"]
    p = heuristic_plan(Q + " Design an experiment and explain how it could be implemented.", {})
    assert {"analysis", "gaps", "experiment_design", "ml_impl", "review"} <= set(p.stages)
    assert "math" not in p.stages and "experiment_run" not in p.stages
    assert "math" in heuristic_plan("Derive a mathematical objective for optimizing retrieval quality.", {}).stages
    assert "ml_impl" in heuristic_plan("Build a model pipeline for classifying research documents.", {}).stages


def test_citation_sanitizer():
    assert sanitize_citations("a [P1, P7] b", {"P1"}) == "a [P1] b"
    assert "P7" not in sanitize_citations("a [P1, P7] b", {"P1"})


# ---------- Deep Agents: real library, real subagent delegation ----------
class ScriptedChat(BaseChatModel):
    script: list
    calls: int = 0

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        msg = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def test_deep_director_delegates_to_real_subagent(svc):
    plan_json = '{"complexity":"complex","stages":["literature","analysis"],"queries":["agentic rag"],"rationale":"r"}'
    delegate = AIMessage(content="", tool_calls=[{"name": "task", "id": "t1", "args": {
        "description": "Scope the literature on agentic RAG", "subagent_type": "researcher"}}])
    models = {}

    def factory(provider, model):
        if provider == NVIDIA:
            models["director"] = ScriptedChat(script=[delegate, AIMessage(content=plan_json)])
            return models["director"]
        models[model] = ScriptedChat(script=[AIMessage(content="Scoping notes from researcher")])
        return models[model]

    d = DeepDirector(svc.settings, svc.toolkit, factory)
    plan, delegations = d.plan(Q, {})
    assert plan.stages == ["literature", "analysis"]
    assert delegations and delegations[0]["subagent"] == "researcher"
    assert models["openai/gpt-oss-20b"].calls >= 1            # the Groq-routed subagent model actually ran


# ---------- API: background execution + frontend client ----------
def test_api_returns_id_immediately_and_runs_in_background(svc):
    svc.director = StubDirector(["literature"])
    app = create_app(svc)
    gate = threading.Event()
    real = app.state.jobs.graph

    class Slow:
        def invoke(self, *a, **k):
            gate.wait(10)
            return real.invoke(*a, **k)
    app.state.jobs.graph = Slow()
    client = TestClient(app)
    t0 = time.time()
    r = client.post("/research", json={"question": Q})
    assert r.status_code == 202 and time.time() - t0 < 2
    rid = r.json()["research_id"]
    assert client.get(f"/research/{rid}").json()["status"] in ("queued", "running")   # still in flight
    assert client.get(f"/research/{rid}/report").status_code == 409
    gate.set()
    app.state.jobs.wait(rid, 30)
    s = client.get(f"/research/{rid}").json()
    assert s["status"] == "completed" and s["progress"] == 1.0 and s["agents"]
    assert "markdown" in client.get(f"/research/{rid}/report").json()
    assert client.get("/research/nope").status_code == 404


def test_health_models_providers_and_frontend_client(svc):
    svc.director = StubDirector(["literature"])
    client = TestClient(create_app(svc))
    api = ResearchAPI(client=client)
    assert api.health()["nvidia_available"] is True
    assert {m["agent"] for m in api.models()} >= {"researcher", "director"}
    prov = api.providers(check=True)
    assert prov[GROQ]["model_check"]["reachable"] and "qwen/qwen3.8-27b" in prov[GROQ]["model_check"]["missing_ids"]
    rid = api.start(Q)["research_id"]
    client.app.state.jobs.wait(rid, 30)
    assert api.status(rid)["status"] == "completed" and api.papers(rid) and "## 1. Executive" in api.report(rid)
    assert api.events(rid) and api.memory("hallucination")["results"]
    assert rid in [j["id"] for j in api.list()]
