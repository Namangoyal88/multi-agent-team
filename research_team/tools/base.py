"""Research-tool adapters. Every call returns a ToolResult: failures are explicit, never "no results"."""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field

import httpx


@dataclass
class ToolResult:
    tool: str
    ok: bool
    items: list[dict] = field(default_factory=list)
    error: str | None = None      # set whenever ok is False
    status_code: int | None = None
    note: str | None = None       # e.g. "not configured"
    latency_s: float = 0.0

    def to_dict(self):
        return asdict(self)


class Adapter:
    name = "adapter"
    requires_key: str | None = None

    def __init__(self, keys: dict[str, str] | None = None, *, transport: httpx.BaseTransport | None = None,
                 retries: int = 2, sleep=time.sleep):
        self.keys = keys or {}
        self.http = httpx.Client(timeout=30.0, transport=transport, follow_redirects=True)
        self.retries, self._sleep = retries, sleep

    def available(self) -> bool:
        return self.requires_key is None or bool(self.keys.get(self.requires_key))

    def _get(self, url: str, **kw) -> httpx.Response:
        """GET with backoff on 429/5xx/network errors. Raises httpx.HTTPError when exhausted."""
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self.http.get(url, **kw)
                if r.status_code == 429 or r.status_code >= 500:
                    last = httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                else:
                    r.raise_for_status()
                    return r
            except httpx.HTTPStatusError:
                raise
            except httpx.HTTPError as exc:
                last = exc
            if attempt < self.retries:
                self._sleep(min(2 ** attempt, 8))
        assert last is not None
        raise last

    def _post(self, url: str, **kw) -> httpx.Response:
        r = self.http.post(url, **kw)
        r.raise_for_status()
        return r

    def search(self, query: str, limit: int = 8) -> ToolResult:
        if not self.available():
            return ToolResult(self.name, False, error=f"{self.name} not configured: set {self.requires_key}",
                              note="not_configured")
        start = time.time()
        try:
            items = self._search(query, limit)
            return ToolResult(self.name, True, items, latency_s=round(time.time() - start, 3),
                              note=None if items else "query succeeded but returned zero results")
        except httpx.HTTPStatusError as exc:
            return ToolResult(self.name, False, error=f"{self.name} HTTP error: {exc}",
                              status_code=exc.response.status_code)
        except (httpx.HTTPError, ET.ParseError, ValueError, KeyError, TypeError) as exc:
            return ToolResult(self.name, False, error=f"{self.name} failed: {exc!r}")

    def _search(self, query: str, limit: int) -> list[dict]:
        raise NotImplementedError
