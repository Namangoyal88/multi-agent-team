"""LangGraph research workflow with dynamic stage routing, escalation, verification and a
refine loop. Stages the plan does not request are skipped."""
from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .experiments.sandbox import ExecutionDisabled, run_experiment
from .llm.client import LLMError
from .llm.router import EscalationUnavailable
from .agents.deep_director import Plan, heuristic_plan
from .context_budget import ContextBudgetManager
from .reporting import generate_report
from .verification import verify_claims

STAGE_ORDER = ["literature", "analysis", "gaps", "math", "experiment_design", "ml_impl",
               "experiment_run", "review", "verify", "report", "evaluate"]
MANDATORY = {"verify", "report", "evaluate"}


class ResearchState(TypedDict, total=False):
    job_id: str
    question: str
    options: dict
    plan: dict
    plan_source: str
    done: list
    iteration: int
    extra_queries: list
    prior_research: list
    tool_status: dict
    literature: str
    analysis: str
    gaps: str
    math: str
    ml_design: str
    experiment_design: str
    experiment_code: str
    experiment_results: dict
    review: str
    claims: list
    claim_seq: int
    verification: list
    report: str
    evaluation: dict
    errors: list


def evidence_text(papers: list[dict], n: int = 12, chars: int = 650) -> str:
    budget = ContextBudgetManager()
    return budget.evidence_block(papers, limit=n, abstract_chars=chars)


def _code_block(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text or "", re.S)
    return m.group(1) if m else ""


def make_workflow(svc):
    ws, runner = svc.ws, svc.runner

    def finish(state, name, updates, *, mark_done=True):
        done = list(state.get("done", [])) + ([name] if mark_done else [])
        ws.save_state(state["job_id"], {**state, **updates, "done": done})
        return {**updates, "done": done}

    def add_claims(state, out, stage):
        claims, seq = list(state.get("claims", [])), state.get("claim_seq", 0)
        for c in out.claims:
            seq += 1
            claims.append({"id": f"C{seq}", "text": c.text, "stage": stage,
                           "citations": [x.strip().upper() for x in c.citations]})
        return claims, seq

    def run_stage(state, name, agent, task, extra_ctx="", *, key: str, with_claims=True):
        papers = ws.papers(state["job_id"])
        ctx = f"EVIDENCE:\n{evidence_text(papers)}"
        if extra_ctx:
            ctx += f"\n\n{extra_ctx}"
        out, err = runner.run(state["job_id"], agent, task, ctx)
        errors = list(state.get("errors", []))
        upd: dict = {}
        if out is None:
            errors.append(f"{name}: {err}")
            upd[key] = ""
        else:
            upd[key] = out.content
            if with_claims:
                upd["claims"], upd["claim_seq"] = add_claims(state, out, name)
        upd["errors"] = errors
        return upd

    # ---- nodes ---------------------------------------------------------
    def plan_node(state):
        jid, q, opts = state["job_id"], state["question"], state.get("options", {})
        ws.log(jid, "director", "start", "Planning research")
        plan, source, delegations = None, "", []
        errors = list(state.get("errors", []))
        if svc.director.available():
            try:
                plan, delegations = svc.director.plan(q, opts)
                source = "deep_director (NVIDIA Nemotron + Deep Agents subagents)"
            except Exception as exc:  # any director failure must not kill the run
                errors.append(f"plan: Deep Director failed: {exc!r}")
                ws.log(jid, "director", "error", f"Deep Director failed: {exc!r}")
        else:
            errors.append("plan: NVIDIA not configured; using heuristic planner")
        if plan is None:
            plan, source = heuristic_plan(q, opts), "heuristic_fallback"
        stages = [s for s in plan.stages]
        if not opts.get("enable_experiments", False) or "experiment_run" in stages and \
                not svc.settings.enable_code_execution:
            if "experiment_run" in stages:
                errors.append("plan: experiment_run skipped (code execution disabled)")
            stages = [s for s in stages if s != "experiment_run"]
        plan = Plan(**{**plan.model_dump(), "stages": stages})
        prior = svc.memory.recall(q, 3, exclude_job=jid)
        ws.put_artifact(jid, "plan", "plan", {**plan.model_dump(), "source": source,
                                              "delegations": delegations})
        ws.put_artifact(jid, "retrieval", "query_understanding", plan.query_understanding or {})
        ws.log(jid, "director", "plan", f"stages={stages}", {"source": source, "delegations": delegations})
        return finish(state, "plan", {"plan": plan.model_dump(), "plan_source": source, "errors": errors,
                                      "prior_research": prior, "iteration": 0, "done": []},
                      mark_done=False)

    def literature_node(state):
        jid, plan = state["job_id"], state["plan"]
        queries = (plan.get("queries") or [state["question"][:120]]) + state.get("extra_queries", [])
        # Keep the model's query understanding as the retrieval contract; refinement may add
        # supplemental queries, but the relevance gate remains the final authority.
        papers, status = svc.toolkit.literature_search(
            list(dict.fromkeys(queries))[:8],
            query_spec=plan.get("query_understanding") or None,
            question=state["question"],
            candidate_cap=48, final_cap=12)
        registry = ws.replace_papers(jid, papers)
        ws.put_artifact(jid, "retrieval", "selected_papers", registry)
        errors = list(state.get("errors", []))
        for t, s in status.items():
            errors += [f"literature[{t}]: {e}" for e in s.get("errors", [])[:2]]
        types = set(plan.get("research_types", []))
        for tool, hit in (("github", {"technical_implementation", "reproducibility_study", "ml_dl_architecture"}),
                          ("kaggle", {"dataset_benchmark_analysis"}), ("web_search", {"technical_implementation"})):
            if types & hit:
                r = svc.toolkit.search(tool, queries[0])
                ws.put_artifact(jid, "sources", tool, r.to_dict())
                if not r.ok:
                    errors.append(f"{tool}: {r.error}")
        ws.put_artifact(jid, "tool_status", "literature", status)
        for p in registry:
            svc.memory.remember(jid, "paper", f"{p['title']}. {p.get('abstract', '')[:600]}",
                                {"pid": p["pid"], "url": p.get("url")})
            for a in p.get("authors", [])[:5]:
                svc.memory.add_edge(p["title"], "authored_by", a, jid)
        upd = {"tool_status": status, "errors": errors}
        if not registry:
            errors.append("literature: NO papers retrieved - see tool errors above; research is ungrounded")
            upd["literature"] = ""
        else:
            upd.update(run_stage({**state, "errors": errors}, "literature", "researcher",
                                 "Summarise the retrieved literature relevant to the question: key papers, "
                                 "methods, findings, datasets/benchmarks. Cite [P#].", key="literature"))
        return finish(state, "literature", upd)

    def simple(name, agent, task, key, ctx_keys=(), claims=True):
        def node(state):
            extra = "\n".join(f"{k.upper()}:\n{state.get(k, '')}" for k in ctx_keys if state.get(k))
            return finish(state, name, run_stage(state, name, agent, task, extra, key=key,
                                                 with_claims=claims))
        return node

    def experiment_design_node(state):
        # Preserve the original experiment_design stage, but make the requested architecture explicit:
        # ML Engineer (120B) designs -> Prompt Guard 86M gates -> sandbox executes -> Reviewer reviews.
        upd = run_stage(state, "experiment_design", "ml_engineer",
                        "Design one executable research experiment (hypothesis, baselines, ablations, "
                        "metrics, seeds, configs). When feasible in pure-python/numpy in a few seconds, "
                        "include ONE self-contained python code block that prints a final "
                        "'METRICS_JSON: {...}' line of measured values.",
                        "\n".join(f"{k.upper()}:\n{state.get(k, '')}" for k in ("gaps", "ml_design", "math")
                                  if state.get(k)), key="experiment_design", with_claims=False)
        design = upd.get("experiment_design", "")
        code = _code_block(design)
        guard_input = design
        if code:
            guard_input += "\n\nEXPERIMENT_CODE:\n" + code
        guard = svc.safety.scan(guard_input, agent="experimenter", job_id=state["job_id"])
        ws.put_artifact(state["job_id"], "experiment_safety", "prompt_guard", guard.to_dict())
        if guard.status != "BENIGN":
            reason = f"Prompt Guard blocked experiment design: {guard.status}. {guard.rationale}"
            upd["experiment_code"] = ""
            upd["errors"] = list(state.get("errors", [])) + [reason]
            ws.log(state["job_id"], "experimenter", "safety_block", reason)
        else:
            upd["experiment_code"] = code
            ws.log(state["job_id"], "experimenter", "safety_pass", guard.rationale)
        return finish(state, "experiment_design", upd)

    def experiment_run_node(state):
        jid, code = state["job_id"], state.get("experiment_code", "")
        if not code:
            res = {"executed": False, "reason": "No runnable code was produced by the Experimenter."}
        else:
            try:
                ws.log(jid, "experimenter", "run", "Executing experiment in sandbox")
                res = run_experiment(code, enabled=svc.settings.enable_code_execution,
                                     timeout=svc.settings.sandbox_timeout)
                if not res.get("executed"):
                    res["reason"] = "Code rejected by validation: " + "; ".join(res.get("problems", []))
            except ExecutionDisabled as exc:
                res = {"executed": False, "reason": str(exc)}
        ws.put_artifact(jid, "experiment", "result", res)
        ws.put_artifact(jid, "experiment", "code", code)
        return finish(state, "experiment_run", {"experiment_results": res})

    def verify_node(state):
        jid = state["job_id"]
        ws.log(jid, "verifier", "start", f"Verifying {len(state.get('claims', []))} claims")
        results = verify_claims(state.get("claims", []), ws.papers(jid), svc.router, svc.safety, ws, jid) if state.get("claims") else []
        ws.put_artifact(jid, "verification", "claims", results)
        for r in results:
            if r["status"] in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
                svc.memory.remember(jid, "finding", r["text"], {"citations": r["citations"]})
        return finish(state, "verify", {"verification": results})

    def report_node(state):
        md = generate_report(state, svc)
        ws.put_artifact(state["job_id"], "report", "final.md", md)
        return finish(state, "report", {"report": md})

    def evaluate_node(state):
        jid, ver = state["job_id"], state.get("verification", [])
        n = len(ver)
        good = sum(1 for v in ver if v["status"] in ("SUPPORTED", "PARTIALLY_SUPPORTED"))
        bad = [v for v in ver if v["status"] in ("UNSUPPORTED", "CONTRADICTED")]
        npapers = len(ws.papers(jid))
        issues = []
        if "literature" in state["plan"]["stages"] and npapers < svc.settings.min_papers:
            issues.append(f"only {npapers} papers retrieved")
        if n >= 3 and good / n < 0.3:
            issues.append(f"only {good}/{n} claims verified")
        it = state.get("iteration", 0)
        ev = {"verdict": "accept", "issues": issues, "papers": npapers, "claims": n, "verified": good,
              "unsupported_or_contradicted": len(bad), "iteration": it}
        contradicted = [v for v in ver if v["status"] == "CONTRADICTED"]
        if contradicted and svc.router.nvidia_available():
            try:
                r = svc.router.escalate([{"role": "user", "content":
                    "Briefly reconcile these contradicted claims and say what evidence would resolve them:\n"
                    + "\n".join(f"- {c['text']} ({c['rationale']})" for c in contradicted)}])
                ev["conflict_note"] = r.content[:1500]
                ws.log(jid, "nvidia", "escalation", "Conflicting evidence reviewed by Nemotron")
            except EscalationUnavailable as exc:
                ev["conflict_note"] = f"escalation unavailable: {exc}"
        if issues and it < svc.settings.max_refine_iterations and "literature" in state["plan"]["stages"]:
            ev["verdict"] = "refine"
            redo = {"literature", "analysis", "gaps", "verify", "report", "evaluate"}
            words = " ".join(state["question"].split()[:8])
            ws.log(jid, "director", "refine", f"Re-running evidence stages: {issues}")
            upd = {"evaluation": ev, "iteration": it + 1, "extra_queries": [f"{words} survey", f"{words} benchmark"],
                   "claims": [c for c in state.get("claims", []) if c["stage"] not in redo],
                   "done": [d for d in state.get("done", []) if d not in redo]}
            ws.save_state(jid, {**state, **upd})
            return upd
        return finish(state, "evaluate", {"evaluation": ev})

    nodes = {"plan": plan_node, "literature": literature_node,
             "analysis": simple("analysis", "researcher",
                                "Compare methodologies, results, datasets and benchmarks across the papers. "
                                "Flag contradictions between papers.", "analysis", ("literature",)),
             "gaps": simple("gaps", "researcher", "Identify unresolved research gaps and contradictory "
                            "findings; set needs_escalation if evidence conflicts.", "gaps",
                            ("literature", "analysis")),
             "math": simple("math", "mathematician", "Formulate the problem mathematically: objectives, "
                            "metrics, derivations and assumptions relevant to the question.", "math",
                            ("gaps",), claims=False),
             "experiment_design": experiment_design_node,
             "ml_impl": simple("ml_impl", "ml_engineer", "Propose the ML/system architecture, data "
                               "preprocessing, evaluation methodology and how to implement it.", "ml_design",
                               ("gaps", "math"), claims=False),
             "experiment_run": experiment_run_node,
             "review": simple("review", "reviewer", "Critically review the proposed methodology, novelty "
                              "claims, experimental design and reproducibility.", "review",
                              ("gaps", "ml_design", "experiment_design"), claims=False),
             "verify": verify_node, "report": report_node, "evaluate": evaluate_node}

    def route(state: ResearchState) -> str:
        wanted = set(state["plan"]["stages"]) | MANDATORY
        for s in STAGE_ORDER:
            if s in wanted and s not in state.get("done", []):
                return s
        return END

    g = StateGraph(ResearchState)
    for n, fn in nodes.items():
        g.add_node(n, fn)
    g.add_edge(START, "plan")
    targets = {s: s for s in STAGE_ORDER} | {END: END}
    for n in nodes:
        g.add_conditional_edges(n, route, targets)
    return g.compile()
