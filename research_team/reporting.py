"""Grounded final research package generation.

The Report Generator sees validated claims plus explicitly labelled proposal notes. Raw stage
outputs are never used as a fallback, and factual sections must contain citations to claims that
passed verification.
"""
from __future__ import annotations

import json
import re

from .llm.client import LLMError
from .context_budget import ContextBudgetManager
from .llm.structured import StructuredOutputError, extract_json

NARRATIVE = [
    "executive_summary", "problem_definition", "research_objectives", "literature_review",
    "comparison_of_methods", "key_findings", "research_gaps", "mathematical_analysis",
    "proposed_methodology", "system_architecture", "experimental_design", "baselines", "datasets",
    "evaluation_metrics", "critical_review", "limitations", "future_directions", "conclusions",
]
NA = "_Not performed or not applicable for this run._"
_NOT_ENOUGH = "Not enough evidence gathered."
_CITE = re.compile(r"\[(P\d+(?:\s*,\s*P\d+)*)\]")

FACTUAL_SECTIONS = {
    "executive_summary", "problem_definition", "literature_review", "comparison_of_methods",
    "key_findings", "research_gaps", "limitations", "future_directions", "conclusions",
}


def sanitize_citations(text: str, valid: set[str]) -> str:
    def fix(m):
        ids = [i.strip() for i in m.group(1).split(",")]
        kept = [i for i in ids if i in valid]
        return ("[" + ", ".join(kept) + "]") if kept else ""
    return _CITE.sub(fix, text)


def _citation_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for match in _CITE.findall(text):
        ids.update(i.strip() for i in match.split(","))
    return ids


def _strict_section(text: str, valid: set[str], good_claim_cites: set[str], *, factual: bool) -> str:
    raw = (text or "").strip()
    if not raw:
        return NA
    raw_ids = _citation_ids(raw)
    # A factual section that originally contains an invalid citation is unsafe even if the
    # sanitizer later removes that citation. Do not let a fabricated reference become
    # accidentally "grounded" merely because a valid citation remains in the same marker.
    if factual and any(cid not in valid for cid in raw_ids):
        return _NOT_ENOUGH
    text = sanitize_citations(raw, valid)
    if not text:
        return NA
    if factual:
        # No validated claim means no evidence-backed factual narrative is safe to publish.
        if not good_claim_cites:
            return _NOT_ENOUGH
        # Every non-empty non-heading line in a factual section must point at a validated citation.
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not _citation_ids(stripped) & good_claim_cites:
                return _NOT_ENOUGH
    return text


def _narrative(state: dict, papers: list[dict], good_claims: list[dict], svc) -> tuple[dict, str | None]:
    mgr = ContextBudgetManager()
    plist = "\n".join(f"[{p['pid']}] {p['title']} ({p.get('year', '')})" for p in papers[:12]) or "(none)"
    # Only validated claims are evidence. Cap the list to keep the report request predictable.
    evidence = "\n".join(
        f"- {c['text'][:500]} [{', '.join(c['citations'])}] (status={c['status']})" for c in good_claims[:30]
    ) or "(no validated empirical claims)"
    proposal = {k: str(state.get(k, ""))[:1800] for k in ("math", "ml_design", "experiment_design", "review")
               if state.get(k)}
    user_content = mgr.report_context(
        f"QUESTION: {state['question']}\n\nPAPERS:\n{plist}\n\nVALIDATED CLAIMS:\n{evidence}\n\n"
        f"PROPOSAL NOTES (not evidence):\n{proposal}")
    system = (
        "You are the Report Generator. Build a research report from the provided evidence. "
        "VALIDATED CLAIMS are the only source for empirical/factual statements. Proposal notes "
        "are not evidence and may only be used to describe proposed methodology, architecture, "
        "mathematical formulation, experimental design, or critical review of the proposal. "
        "For every factual section, every statement must carry a citation to a validated claim. "
        "Never invent papers, metrics, results, benchmarks, or facts. If evidence is insufficient, "
        "write 'Not enough evidence gathered.' Reply with ONE JSON object whose keys are exactly: "
        + ", ".join(NARRATIVE) + ". Values are markdown strings.")
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    try:
        resp = svc.router.call("report_generator", msgs, temperature=0.1,
                               max_tokens=2200, response_format={"type": "json_object"})
        data = extract_json(resp.content)
        if not isinstance(data, dict):
            raise StructuredOutputError("report JSON is not an object")
        return {k: str(data.get(k, "")).strip() for k in NARRATIVE}, None
    except (LLMError, StructuredOutputError) as first_exc:
        # Formatting recovery is deliberately tiny: never resend the full report context.
        repair = [
            {"role": "system", "content": system + " Return ONLY a JSON object."},
            {"role": "user", "content": "Repair this partial report response without adding facts:\n"
             + (getattr(locals().get("resp", None), "content", "")[:7000])},
        ]
        try:
            resp2 = svc.router.call("report_generator", repair, temperature=0.0, max_tokens=1800,
                                    response_format={"type": "json_object"})
            data = extract_json(resp2.content)
            if not isinstance(data, dict):
                raise StructuredOutputError("report JSON is not an object")
            return {k: str(data.get(k, "")).strip() for k in NARRATIVE}, None
        except (LLMError, StructuredOutputError) as second_exc:
            return {}, f"Report Generator unavailable after structured-output recovery ({first_exc}; {second_exc}); only verified material is included."


def generate_report(state: dict, svc) -> str:
    papers = svc.ws.papers(state["job_id"])
    valid = {p["pid"] for p in papers}
    ver = state.get("verification", [])
    good = [c for c in ver if c["status"] in ("SUPPORTED", "PARTIALLY_SUPPORTED")]
    good_claim_cites = {cid for c in good for cid in c.get("citations", []) if cid in valid}
    nar, warn = _narrative(state, papers, good, svc)

    def sec(k: str) -> str:
        return _strict_section(nar.get(k), valid, good_claim_cites, factual=k in FACTUAL_SECTIONS)

    exp = state.get("experiment_results") or {}
    if exp.get("executed"):
        metrics = exp.get("metrics")
        if metrics is not None:
            exp_txt = (f"Executed (status: {exp['status']}, {exp.get('duration_s')}s).\n\n"
                       f"Metrics reported by the experiment script:\n```json\n"
                       f"{json.dumps(metrics, indent=2, default=str)}\n```")
        else:
            exp_txt = (f"Executed (status: {exp['status']}) but no METRICS_JSON line was emitted; "
                       "no numeric experiment results are reported.")
    else:
        exp_txt = f"**No experiments were executed.** {exp.get('reason', '')}".strip()

    rows = "\n".join(
        f"| {c['id']} | {c['text'][:140]} | {', '.join(c['citations']) or '-'} | "
        f"**{c['status']}** | {c.get('rationale', '')[:140]} |" for c in ver
    )
    plist = "\n".join(
        f"- **[{p['pid']}]** {p['title']} — {', '.join(p.get('authors', [])[:3])}"
        f" ({p.get('year', 'n.d.')}) · {p.get('source', 'unknown')} · {p.get('url', '')}"
        for p in papers
    )
    ev = state.get("evaluation", {})
    parts = [f"# Research Report\n\n> Question: {state['question']}\n"]
    if warn:
        parts.append(f"> ⚠️ {warn}\n")

    sections = [
        ("Executive Summary", sec("executive_summary")),
        ("Research Question", state["question"]),
        ("Problem Definition", sec("problem_definition")),
        ("Research Objectives", sec("research_objectives")),
        ("Literature Review", sec("literature_review")),
        ("Relevant Papers and Citations", plist or NA),
        ("Comparison of Existing Methods", sec("comparison_of_methods")),
        ("Key Findings", sec("key_findings")),
        ("Research Gaps", sec("research_gaps")),
        ("Mathematical / Theoretical Analysis", sec("mathematical_analysis")),
        ("Proposed Methodology", sec("proposed_methodology")),
        ("System / ML Architecture", sec("system_architecture")),
        ("Experimental Design", sec("experimental_design")),
        ("Baselines", sec("baselines")),
        ("Datasets", sec("datasets")),
        ("Evaluation Metrics", sec("evaluation_metrics")),
        ("Experiment Results", exp_txt),
        ("Ablation / Comparison Results", "Only reported when produced by an executed experiment. "
         + ("See above." if exp.get("metrics") is not None else "None available.")),
        ("Critical Review", sec("critical_review")),
        ("Claim / Evidence Verification",
         ("Verification uses retrieved abstracts only (not full texts).\n\n| ID | Claim | Cites | Status | "
          "Rationale |\n|---|---|---|---|---|\n" + rows) if ver else NA),
        ("Limitations", sec("limitations")),
        ("Future Research Directions", sec("future_directions")),
        ("Final Conclusions", sec("conclusions")),
        ("References", plist or "_No sources were retrieved."),
    ]
    for i, (title, body) in enumerate(sections, 1):
        parts.append(f"## {i}. {title}\n\n{body}\n")
    tools = state.get("tool_status", {})
    parts.append(
        "## Appendix: Reproducibility & Run Information\n\n"
        f"- Plan source: {state.get('plan_source')}\n"
        f"- Stages run: {', '.join(state.get('done', []))}\n"
        f"- Evaluation: {ev.get('verdict')} {ev.get('issues', '')}\n"
        f"- Tool status: {tools}\n"
        f"- Errors: {state.get('errors') or 'none'}\n"
    )
    return "\n".join(parts)
