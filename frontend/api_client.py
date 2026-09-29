"""Thin client used by the Streamlit dashboard (also exercised by the tests)."""
from __future__ import annotations

import httpx


class ResearchAPI:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", client: httpx.Client | None = None):
        self.c = client or httpx.Client(base_url=base_url, timeout=30)

    def _j(self, method: str, url: str, **kw):
        r = self.c.request(method, url, **kw)
        r.raise_for_status()
        return r.json()

    def start(self, question: str, enable_experiments: bool = False):
        return self._j("POST", "/research", json={"question": question,
                                                   "enable_experiments": enable_experiments})

    def list(self): return self._j("GET", "/research")
    def status(self, rid): return self._j("GET", f"/research/{rid}")
    def events(self, rid, after=0): return self._j("GET", f"/research/{rid}/events", params={"after": after})
    def report(self, rid): return self._j("GET", f"/research/{rid}/report")["markdown"]
    def papers(self, rid): return self._j("GET", f"/research/{rid}/papers")
    def artifacts(self, rid, kind=None): return self._j("GET", f"/research/{rid}/artifacts", params={"kind": kind} if kind else None)
    def health(self): return self._j("GET", "/health")
    def models(self): return self._j("GET", "/models")
    def providers(self, check=False): return self._j("GET", "/providers", params={"check": check})
    def memory(self, q): return self._j("GET", "/memory/search", params={"q": q})
