"""Shared fixtures. Every external call (Groq, NVIDIA, arXiv, ...) is mocked: no quota is consumed."""
import json
import re
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_team.agents.deep_director import Plan  # noqa: E402
from research_team.config import GROQ, NVIDIA, load_settings  # noqa: E402
from research_team.services import build_services  # noqa: E402

ABSTRACT = "Agentic retrieval reduces hallucination rates in question answering benchmarks."
ATOM = """<feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2401.00001v1</id><title>Agentic RAG Survey</title>
<summary>{a}</summary><published>2024-01-01T00:00:00Z</published><author><name>A. Author</name></author></entry>
<entry><id>http://arxiv.org/abs/2401.00002v1</id><title>Self-RAG Study</title>
<summary>{a}</summary><published>2024-02-01T00:00:00Z</published><author><name>B. Author</name></author></entry>
</feed>""".format(a=ABSTRACT)


class LLMMock:
    """Fake OpenAI-compatible endpoint. Records (provider, model, kind) for routing assertions."""

    def __init__(self):
        self.calls, self.fail_groq, self.fail_status = [], False, 500
        self.specialist_overrides = {}   # agent-keyword -> dict merged into specialist JSON

    def handler_for(self, provider):
        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "openai/gpt-oss-120b"}]})
            body = json.loads(request.content)
            system = body["messages"][0]["content"]
            self.calls.append((provider, body["model"], system[:40]))
            if provider == GROQ and self.fail_groq:
                return httpx.Response(self.fail_status, text="boom")
            user = body["messages"][1]["content"] if len(body["messages"]) > 1 else ""
            return httpx.Response(200, json={"choices": [{"message": {"content": self.reply(provider, system, user)}}],
                                             "usage": {"total_tokens": 10}})
        return handle

    def reply(self, provider, system, user=""):
        if "Classify this input ONLY" in system:
            user = user.upper()
            return "MALICIOUS" if "IGNORE YOUR PREVIOUS" in user or "JAILBREAK" in user else "BENIGN"
        if "semantic evidence reviewer" in system:
            return json.dumps({"verdicts": [{"id": "C1", "status": "SUPPORTED", "rationale": "ok",
                                               "evidence_quote": "reduces hallucination rates"}]})
        if "Report Generator" in system:
            from research_team.reporting import NARRATIVE
            return json.dumps({k: f"Section {k} text [P1] [P99]." for k in NARRATIVE})
        if "deep-reasoning escalation" in system or "escalation model" in system:
            return json.dumps({"content": "Escalated answer.", "confidence": 0.9, "claims": []})
        out = {"content": "Specialist answer [P1].", "confidence": 0.9, "needs_escalation": False,
               "claims": [{"text": "Agentic retrieval reduces hallucination.", "citations": ["P1"]}]}
        for key, extra in self.specialist_overrides.items():
            if key in system:
                out.update(extra)
        return json.dumps(out)


def tool_handler(fail=None):
    fail = fail or set()

    def handle(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if any(f in host for f in fail):
            return httpx.Response(500, text="upstream down")
        if "arxiv" in host:
            return httpx.Response(200, text=ATOM)
        if "semanticscholar" in host:
            return httpx.Response(200, json={"data": [{"title": "Semantic RAG Paper", "abstract": ABSTRACT,
                                                       "year": 2024, "authors": [{"name": "C"}],
                                                       "externalIds": {"DOI": "10.1/x"}, "url": "http://s2/1"}]})
        if "ncbi" in host:
            return httpx.Response(200, json={"esearchresult": {"idlist": []}})
        if "github" in host:
            return httpx.Response(200, json={"items": []})
        return httpx.Response(404)
    return handle


class StubDirector:
    def __init__(self, stages, complexity="complex", available=True):
        self.stages, self.complexity, self._avail = stages, complexity, available

    def available(self):
        return self._avail

    def plan(self, question, options):
        return Plan(complexity=self.complexity, stages=self.stages, queries=["agentic rag hallucination"],
                    research_types=["literature_review"]), [{"subagent": "researcher", "description": "scope"}]


@pytest.fixture
def llm():
    return LLMMock()


@pytest.fixture
def make_svc(llm):
    def _make(*, nvidia=True, tool_fail=None, code_exec=False, db=":memory:"):
        env = {"GROQ_API_KEY": "g", "NVIDIA_API_KEY": "n" if nvidia else "",
               "NVIDIA_MODEL": "nemotron-test" if nvidia else "", "RESEARCH_DB_PATH": db,
               "ENABLE_CODE_EXECUTION": str(code_exec).lower(), "MAX_REFINE_ITERATIONS": "0"}
        s = load_settings(env, dotenv=False)
        return build_services(s, llm_transports={GROQ: httpx.MockTransport(llm.handler_for(GROQ)),
                                                 NVIDIA: httpx.MockTransport(llm.handler_for(NVIDIA))},
                              tool_transport=httpx.MockTransport(tool_handler(tool_fail)),
                              sleep=lambda s: None)
    return _make


@pytest.fixture
def svc(make_svc):
    return make_svc()
