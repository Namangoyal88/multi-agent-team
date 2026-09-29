# Multi-Agent AI Research Team — QA Report

Date: 2026-09-29

## Scope

This rebuild preserves the original architecture and workflow stages. The fixes are confined to retrieval relevance, context budgeting, structured-output recovery, duplicate-source control, the requested experiment safety path, verification hardening, and Deep Agents delegation recovery.

Original workflow remains:

`User → NVIDIA Nemotron 3 Ultra Deep Research Director → specialist agents/tools → shared workspace/memory → LangGraph stages → verification → report → evaluation/refinement`

## Model routing

| Role | Model |
|---|---|
| Deep Research Director | `nvidia/nemotron-3-ultra-550b-a55b` |
| Researcher | `openai/gpt-oss-20b` |
| Mathematician | `openai/gpt-oss-120b` |
| ML Engineer | `openai/gpt-oss-120b` |
| Experimenter / Prompt Guard | `openai/gpt-oss-120b` |
| Reviewer | `openai/gpt-oss-20b` |
| Verifier / Prompt Guard | `openai/gpt-oss-20b` |
| Report Generator | `qwen/qwen3.8-27b` |

## Test result

The complete non-UI project suite was exercised in this build with a minimal compatibility shim for the unavailable LangGraph / LangChain / Deep Agents runtime packages:

`43 passed`

This covers:

- core persistence and routing tests
- hardening tests
- retrieval relevance and targeted-query tests
- structured-output recovery tests
- context-budget tests
- Prompt Guard safety tests
- experiment design → safety gate → sandbox path
- deterministic verification → Prompt Guard → Reviewer fallback
- workflow orchestration, escalation, API background execution, and frontend API client

The Streamlit UI test was not executed in this container because the runtime image does not include the `streamlit` package. The source was compiled successfully.

## Fix 1 — Retrieval relevance

The retrieval layer now extracts a semantic research subject instead of treating the entire question as the topic.

Examples:

`What is agentic RAG?` → `Agentic RAG`

`Investigate whether agentic RAG can reduce hallucinations compared with conventional RAG.` → `Agentic RAG`

`New Google research related to nested machine learning` →

- topic: `Nested Learning`
- organization: `Google Research`
- time: `recent/current`
- intent: `research discovery`

Entity-specific Nested Learning queries remain targeted and the candidate set is deduplicated, ranked, filtered, and capped before specialist calls.

## Fix 2 — 413 / context growth

A shared `ContextBudgetManager` bounds expensive requests.

Targets:

- specialist context: ≤ 5,000 estimated input tokens
- verifier context: ≤ 4,000
- report context: ≤ 5,600
- Groq request ceiling: 7,600 estimated input + output tokens

The LLM client also performs a final provider-level message trim and turns an actual 413/431 response into a bounded, explicit error instead of an uncontrolled retry loop.

## Fix 3 — structured-output failures

`extract JSON → normalize → schema validation → compact repair retry`

recovers the observed near-miss forms, including:

`["literature_review"]`

and

`[{"text":"Recovered experiment notes"}]`

The repair request does not resend the full research evidence context.

## Fix 4 — duplicate / garbage source accumulation

Raw results are deduplicated across providers using scholarly IDs, URLs, and normalized titles. Re-running literature replaces the active paper registry so stale `P1...P52` records cannot accumulate across refinement iterations.

At most 48 ranked candidates and 12 selected evidence papers are promoted to downstream stages.

## Fix 5 — experiment pipeline

The existing `experiment_design` and `experiment_run` stage names are preserved. Internally:

`ML Engineer 120B → Prompt Guard 86M → sandbox execution → Reviewer 20B`

Prompt Guard is only a safety classifier. It never generates the experiment and never executes code.

## Fix 6 — verification pipeline

Verification is:

`deterministic citation precheck → deterministic evidence matching → Prompt Guard 22M safety scan → Reviewer 20B semantic fallback`

Prompt Guard 22M screens untrusted retrieved evidence for injection/jailbreak content. The Reviewer handles semantic claim/evidence reasoning where deterministic matching is insufficient.

A `SUPPORTED` result still requires a verbatim evidence quote that occurs in the cited abstract.

## Fix 7 — Deep Agents delegation recovery

The NVIDIA Director remains a real Deep Agents orchestrator. When a task tool-call is surfaced without a corresponding tool execution message, a compatibility fallback directly invokes the configured specialist model once so the declared delegation is not silently skipped.

This specifically restores the expected delegated Researcher `openai/gpt-oss-20b` execution in the scripted Deep Agents test without replacing Deep Agents as the primary orchestration layer.

## Container limitation

The execution image used for this QA pass does not have the project's actual `langchain_core`, `langgraph`, `deepagents`, or `streamlit` runtime packages installed, and outbound package installation is unavailable. Therefore the 43-pass result above is a source-level workflow validation using a small compatibility shim rather than a claim of a fully native dependency environment.

No live Groq or NVIDIA API quota was consumed by these automated tests.
