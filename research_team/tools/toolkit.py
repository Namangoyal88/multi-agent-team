"""Fan-out over adapters, then deduplicate, rank and retain only relevant evidence."""
from __future__ import annotations

from ..retrieval import QuerySpec, infer_query_spec, rank_and_filter
from .adapters import (ArxivAdapter, GitHubAdapter, KaggleAdapter, PubMedAdapter,
                       SemanticScholarAdapter, WebSearchAdapter)

LITERATURE_TOOLS = ("arxiv", "semantic_scholar", "pubmed")
PAPER_SOURCES = {"arxiv", "semantic_scholar", "pubmed"}


class ResearchToolkit:
    def __init__(self, keys: dict[str, str] | None = None, *, transport=None, sleep=None):
        kw = {"transport": transport}
        if sleep:
            kw["sleep"] = sleep
        self.adapters = {a.name: a for a in (
            ArxivAdapter(keys, **kw), SemanticScholarAdapter(keys, **kw), PubMedAdapter(keys, **kw),
            WebSearchAdapter(keys, **kw), GitHubAdapter(keys, **kw), KaggleAdapter(keys, **kw))}

    def status(self) -> dict:
        return {n: {"available": a.available(), "requires": a.requires_key}
                for n, a in self.adapters.items()}

    def search(self, tool: str, query: str, limit: int = 8):
        if tool not in self.adapters:
            raise KeyError(f"unknown tool {tool!r}")
        return self.adapters[tool].search(query, limit)

    def literature_search(self, queries: list[str], tools: tuple[str, ...] = LITERATURE_TOOLS,
                          limit: int = 6, query_spec: QuerySpec | dict | None = None,
                          question: str = "", candidate_cap: int = 48, final_cap: int = 12) -> tuple[list[dict], dict[str, dict]]:
        """Retrieve candidates, then enforce deduplication + relevance + entity/date filters.

        The adapters may return a larger raw fan-out, but only up to `candidate_cap` ranked
        candidates and `final_cap` evidence papers are retained for downstream agents.
        """
        raw: list[dict] = []
        status: dict[str, dict] = {}
        for t in tools:
            s = status.setdefault(t, {"ok_queries": 0, "errors": [], "results": 0})
            for q in queries:
                r = self.search(t, q, limit)
                if r.ok:
                    s["ok_queries"] += 1
                    s["results"] += len(r.items)
                    raw.extend(i for i in r.items if i.get("title"))
                else:
                    s["errors"].append(r.error)

        spec = query_spec
        if isinstance(spec, dict):
            spec = QuerySpec(
                topic=str(spec.get("topic", "")), organization=str(spec.get("organization", "")),
                time=str(spec.get("time", "")), intent=str(spec.get("intent", "research discovery")),
                canonical_terms=list(spec.get("canonical_terms", [])),
            )
        spec = spec or infer_query_spec(question or (queries[0] if queries else ""))
        selected, candidates = rank_and_filter(raw, spec, question, candidate_cap=candidate_cap, final_cap=final_cap)
        status["_retrieval"] = {
            "raw_results": len(raw),
            "deduplicated_ranked_candidates": len(candidates),
            "selected_papers": len(selected),
            "candidate_cap": candidate_cap,
            "final_cap": final_cap,
            "query_understanding": spec.as_dict(),
        }
        return selected, status
