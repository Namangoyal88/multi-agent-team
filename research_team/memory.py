"""Memory layers: short-term (per job, in-process), long-term (SQLite), vector (local hashed
embeddings, no heavy deps), and an optional lightweight knowledge graph."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import defaultdict

from .workspace import Workspace

DIM = 512
_TOKEN = re.compile(r"[a-z0-9]+")


def embed(text: str) -> dict[int, float]:
    """Hashed bag-of-words (+bigrams) vector, L2-normalised. Swap for a real embedding model later."""
    toks = _TOKEN.findall(text.lower())
    grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    vec: dict[int, float] = defaultdict(float)
    for g in grams:
        h = int(hashlib.md5(g.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0 if (h >> 64) % 2 == 0 else -1.0
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


class MemoryLayer:
    def __init__(self, ws: Workspace, *, enable_graph: bool = True):
        self.ws = ws
        self.enable_graph = enable_graph
        self._short: dict[str, dict] = defaultdict(dict)

    # short-term ---------------------------------------------------------
    def short_set(self, job_id: str, key: str, value):
        self._short[job_id][key] = value

    def short_get(self, job_id: str, key: str, default=None):
        return self._short[job_id].get(key, default)

    def short_clear(self, job_id: str):
        self._short.pop(job_id, None)

    # long-term + vector -------------------------------------------------
    def remember(self, job_id: str, kind: str, text: str, meta: dict | None = None):
        meta_s = json.dumps(meta or {})
        self.ws._exec("INSERT INTO long_term(job_id, kind, text, meta, ts) VALUES(?,?,?,?,?)",
                      (job_id, kind, text, meta_s, time.time()))
        self.ws._exec("INSERT INTO vectors(job_id, kind, text, vec, meta) VALUES(?,?,?,?,?)",
                      (job_id, kind, text, json.dumps(embed(text)), meta_s))

    def recall(self, query: str, k: int = 5, *, exclude_job: str | None = None,
               kind: str | None = None) -> list[dict]:
        q = embed(query)
        rows = self.ws._all("SELECT job_id, kind, text, vec, meta FROM vectors")
        scored = []
        for r in rows:
            if r["job_id"] == exclude_job or (kind and r["kind"] != kind):
                continue
            vec = {int(i): v for i, v in json.loads(r["vec"]).items()}
            s = cosine(q, vec)
            if s > 0.05:
                scored.append({"score": round(s, 4), "job_id": r["job_id"], "kind": r["kind"],
                               "text": r["text"], "meta": json.loads(r["meta"])})
        return sorted(scored, key=lambda x: -x["score"])[:k]

    def stats(self) -> dict:
        n = lambda t: self.ws._all(f"SELECT COUNT(*) c FROM {t}")[0]["c"]  # noqa: E731
        return {"long_term_items": n("long_term"), "vector_items": n("vectors"),
                "graph_edges": n("graph_edges") if self.enable_graph else None}

    # knowledge graph (optional) ----------------------------------------
    def add_edge(self, src: str, rel: str, dst: str, job_id: str = ""):
        if self.enable_graph:
            self.ws._exec("INSERT OR IGNORE INTO graph_edges VALUES(?,?,?,?)", (src, rel, dst, job_id))

    def neighbors(self, node: str) -> list[dict]:
        rows = self.ws._all("SELECT src, rel, dst FROM graph_edges WHERE src=? OR dst=?", (node, node))
        return [dict(r) for r in rows]
