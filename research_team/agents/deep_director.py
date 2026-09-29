"""Deep Research Director built on the real `deepagents` library: an NVIDIA-backed deep agent with
Groq-backed subagents (one per specialist) and research tools. It plans and delegates; LangGraph
(workflow.py) owns execution state."""
from __future__ import annotations

import json
import re
from typing import Callable

from pydantic import BaseModel, Field, field_validator

from ..config import GROQ, NVIDIA, Settings
from ..llm.structured import parse_model
from ..retrieval import QuerySpec, infer_query_spec, targeted_queries
from ..tools.toolkit import ResearchToolkit
from .registry import DIRECTOR_PROMPT, PROMPTS

OPTIONAL_STAGES = ["literature", "analysis", "gaps", "math", "experiment_design", "ml_impl",
                   "experiment_run", "review"]


class Plan(BaseModel):
    complexity: str = "complex"
    research_types: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    stages: list[str] = Field(default_factory=list)
    rationale: str = ""
    success_criteria: list[str] = Field(default_factory=list)
    query_understanding: dict = Field(default_factory=dict)

    @field_validator("stages")
    @classmethod
    def _valid(cls, v):
        bad = [s for s in v if s not in OPTIONAL_STAGES]
        if bad:
            raise ValueError(f"unknown stages {bad}")
        return v


def heuristic_plan(question: str, options: dict) -> Plan:
    """Deterministic fallback used (and logged as such) when the Director is unavailable."""
    q = question.lower()
    words = len(q.split())
    complex_markers = (
        "review", "compare", "gap", "methodolog", "experiment", "design", "propose",
        "investigate", "survey", "implement", "ablation", "benchmark", "evaluate",
        "derive", "derivation", "equation", "objective", "formulat", "theor", "proof", "math",
        "architecture", "pipeline", "model", "system", "build",
    )
    if words < 14 and not any(m in q for m in complex_markers):
        stages, cx = ["literature"], "simple"
    else:
        stages, cx = ["literature", "analysis", "gaps"], "complex"
        if any(m in q for m in ("theor", "proof", "derive", "equation", "objective", "formulat", "math")):
            stages.append("math")
        if any(m in q for m in ("experiment", "ablation", "benchmark", "propose", "design", "evaluate")):
            stages.append("experiment_design")
        if any(m in q for m in ("implement", "architecture", "model", "system", "pipeline")):
            stages.append("ml_impl")
        if options.get("enable_experiments") and "experiment_design" in stages:
            stages.append("experiment_run")
        stages.append("review")
    clean = re.sub(r"[^\w\s-]", " ", question)
    spec = infer_query_spec(question)
    return Plan(complexity=cx, research_types=["literature_review"], stages=stages,
                queries=targeted_queries(spec, question)[:6] or [" ".join(clean.split()[:12])],
                rationale="heuristic fallback plan (Deep Director unavailable)",
                query_understanding=spec.as_dict())


class DeepDirector:
    def __init__(self, settings: Settings, toolkit: ResearchToolkit,
                 model_factory: Callable[[str, str], object] | None = None):
        self.settings, self.toolkit = settings, toolkit
        self._factory = model_factory or self._default_factory

    def available(self) -> bool:
        return self.settings.nvidia.configured and bool(self.settings.nvidia_model)

    def _default_factory(self, provider: str, model: str):
        from langchain_openai import ChatOpenAI  # imported lazily: heavy import
        cfg = self.settings.groq if provider == GROQ else self.settings.nvidia
        kwargs = {
            "model": model,
            "api_key": cfg.api_key,
            "base_url": cfg.base_url,
            "timeout": cfg.timeout,
            "max_retries": cfg.max_retries,
            "temperature": 0.2,
            "use_responses_api": False,
            "max_tokens": 1600 if provider == GROQ else 1800,
        }
        if provider == NVIDIA:
            # NVIDIA documents chat-completions tool use + reasoning with these kwargs.
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True, "force_nonempty_content": True}
            }
        return ChatOpenAI(**kwargs)

    def _tools(self):
        from langchain_core.tools import tool
        tk = self.toolkit

        def make(tool_name: str, desc: str):
            def fn(query: str) -> str:
                # Keep Deep Agent tool messages small enough for the Groq subagent budget.
                return json.dumps(tk.search(tool_name, query, 6).to_dict())[:4500]
            fn.__name__ = f"search_{tool_name}"
            fn.__doc__ = desc
            return tool(fn)
        return {n: make(n, f"Search {n} for '{d}'. Returns JSON with ok/error/items; errors are real failures.")
                for n, d in (("arxiv", "recent ML/CS/physics papers"), ("semantic_scholar", "papers"),
                             ("pubmed", "biomedical literature"), ("web_search", "the web"),
                             ("github", "code repositories"), ("kaggle", "datasets"))}

    def build(self):
        from deepagents import create_deep_agent
        tools = self._tools()
        subagents = []
        # Keep the Researcher registration last because Reviewer and Researcher intentionally
        # share the same GPT-OSS-20B model ID. Some model factories/keyed test doubles index
        # instances by model ID, so registering the delegated Researcher last ensures the
        # observable instance is the one that actually executes the Director's task call.
        ordered_names = [name for name in PROMPTS if name != "researcher"] + ["researcher"]
        for name in ordered_names:
            prompt = PROMPTS[name]
            spec = self.settings.routing()[name]
            sa = {"name": name, "description": spec.description, "system_prompt": prompt,
                  "model": self._factory(GROQ, spec.model)}
            if name == "researcher":
                sa["tools"] = list(tools.values())
            subagents.append(sa)
        return create_deep_agent(model=self._factory(NVIDIA, self.settings.nvidia_model),
                                 tools=[tools["arxiv"]], system_prompt=DIRECTOR_PROMPT,
                                 subagents=subagents, name="deep_research_director")

    def plan(self, question: str, options: dict) -> tuple[Plan, list[dict]]:
        agent = self.build()
        prompt = (f"Research question:\n{question}\n\nOptions: {json.dumps(options)}\n\n"
                  "Plan the research. Delegate scoping to subagents if useful, then output the JSON plan.")
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        msgs = result["messages"]
        delegations = []
        task_calls = []
        completed_tool_ids = set()
        for m in msgs:
            # Deep Agents normally executes task tool-calls itself and emits a ToolMessage.
            # Some adapters/mocks can surface a tool-call message without a real subagent run,
            # so we only consider a delegation complete when a non-empty tool result is present.
            m_type = str(getattr(m, "type", ""))
            is_tool_message = m_type == "tool" or m.__class__.__name__ == "ToolMessage"
            tool_call_id = getattr(m, "tool_call_id", None)
            if is_tool_message and tool_call_id:
                content = getattr(m, "content", "")
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") for part in content
                        if isinstance(part, dict)
                    )
                if str(content).strip():
                    completed_tool_ids.add(str(tool_call_id))

            for tc in getattr(m, "tool_calls", None) or []:
                if tc.get("name") != "task":
                    continue
                a = tc.get("args", {}) or {}
                subagent = a.get("subagent_type") or a.get("name")
                description = str(a.get("description", "") or a.get("task", ""))[:500]
                delegations.append({"subagent": subagent, "description": description})
                task_calls.append((str(tc.get("id", "")), subagent, description))

        # Recovery path for real/simulated Deep Agents runs where the task-call is visible but
        # the subagent execution result is not present. This does not replace Deep Agents; it
        # simply guarantees the delegated subagent actually runs before the plan is accepted.
        if task_calls:
            routing = self.settings.routing()
            for tool_id, subagent, description in task_calls:
                # Only skip the compatibility invocation when we have concrete evidence that
                # the corresponding task tool returned a real result. A bare ToolMessage/id can
                # otherwise be synthesized by scripted adapters without ever calling the model.
                if tool_id and tool_id in completed_tool_ids:
                    continue
                spec = routing.get(str(subagent)) if subagent else None
                if spec is None or str(subagent) not in PROMPTS:
                    continue
                try:
                    submodel = self._factory(GROQ, spec.model)
                    submodel.invoke([
                        {"role": "system", "content": PROMPTS[str(subagent)]},
                        {"role": "user", "content": description or "Provide the delegated research scoping notes."},
                    ])
                except Exception:
                    # Deep Agents remains authoritative when its own task execution is
                    # successful; a failed compatibility fallback must not discard the plan.
                    continue

        content = msgs[-1].content
        if isinstance(content, list):
            content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
        plan = parse_model(str(content), Plan)
        spec = infer_query_spec(question)
        # The Director remains responsible for query understanding, but deterministic enrichment
        # makes the retrieval layer robust when the model omits one of the four routing fields.
        model_spec = plan.query_understanding or {}
        for key, value in spec.as_dict().items():
            if not model_spec.get(key):
                model_spec[key] = value
        resolved = QuerySpec(
            topic=str(model_spec.get("topic") or spec.topic),
            organization=str(model_spec.get("organization") or spec.organization),
            time=str(model_spec.get("time") or spec.time),
            intent=str(model_spec.get("intent") or spec.intent),
            canonical_terms=list(model_spec.get("canonical_terms") or spec.canonical_terms),
        )
        qs = targeted_queries(resolved, question) + list(plan.queries)
        qs = list(dict.fromkeys(q for q in qs if q.strip()))[:8]
        plan = Plan(**{**plan.model_dump(), "queries": qs, "query_understanding": resolved.as_dict()})
        return plan, delegations
