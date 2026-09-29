"""OpenAI-compatible chat client (Groq and NVIDIA) with retries, backoff and diagnostics."""
from __future__ import annotations

import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

from ..config import GROQ, ProviderConfig
from ..context_budget import ContextBudgetManager, estimate_tokens


class LLMError(Exception):
    def __init__(self, provider: str, message: str, *, status: int | None = None,
                 retryable: bool = False, attempts: int = 1):
        super().__init__(f"[{provider}] {message}")
        self.provider, self.status, self.retryable, self.attempts = provider, status, retryable, attempts


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    latency_s: float
    attempts: int
    usage: dict = field(default_factory=dict)


_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLMClient:
    def __init__(self, cfg: ProviderConfig, *, transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep, request_budget: int | None = None):
        self.cfg = cfg
        self._http = httpx.Client(base_url=cfg.base_url.rstrip("/") + "/", timeout=cfg.timeout,
                                  transport=transport)
        self._sleep = sleep
        self._lock = threading.Lock()
        self._budget = ContextBudgetManager()
        self._request_budget = request_budget
        self.diagnostics = {"calls": 0, "successes": 0, "failures": 0, "retries": 0,
                            "last_error": None, "last_status": None, "last_latency_s": None}

    def _incr(self, key: str, n: int = 1):
        with self._lock:
            self.diagnostics[key] += n

    def _set(self, **kw):
        with self._lock:
            self.diagnostics.update(kw)

    def _headers(self):
        return {"Authorization": f"Bearer {self.cfg.api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        return min(2 ** attempt + random.uniform(0, 0.5), 30.0)

    def chat(self, model: str, messages: list[dict], *, temperature: float = 0.2,
             max_tokens: int = 1600, response_format: dict | None = None) -> LLMResponse:
        if not self.cfg.configured:
            raise LLMError(self.cfg.name, "API key not configured")
        if not model:
            raise LLMError(self.cfg.name, "model ID not configured")

        # The current free Groq plan documents 8K TPM for GPT-OSS/Qwen. Keep a local
        # margin below that ceiling so the API never receives the previously observed
        # ~8.3K/12K-token requests. The value is configurable in project settings and
        # specialist/report callers also apply tighter stage budgets upstream.
        provider_budget = (self._request_budget if self._request_budget is not None else
                           (7600 if self.cfg.name == GROQ else 16000))
        bounded, _ = self._budget.fit_messages(
            messages, input_budget=max(provider_budget - max_tokens, 200),
            output_tokens=max_tokens, provider_budget=provider_budget)
        body = {"model": model, "messages": bounded, "temperature": temperature,
                "max_tokens": max_tokens}
        if response_format is not None:
            body["response_format"] = response_format
        self._incr("calls")
        start, last_err, status = time.time(), "unknown", None
        for attempt in range(self.cfg.max_retries + 1):
            retry_after = None
            try:
                r = self._http.post("chat/completions", json=body, headers=self._headers())
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_err, status = f"network/timeout: {exc!r}", None
            else:
                status, retry_after = r.status_code, r.headers.get("retry-after")
                if status == 200:
                    try:
                        data = r.json()
                        text = data["choices"][0]["message"].get("content") or ""
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        self._incr("failures")
                        self._set(last_error=f"malformed response: {exc!r}")
                        raise LLMError(self.cfg.name, f"malformed response: {exc!r}",
                                       attempts=attempt + 1)
                    lat = time.time() - start
                    self._incr("successes")
                    self._set(last_status=200, last_latency_s=round(lat, 3), last_error=None)
                    return LLMResponse(_THINK.sub("", text).strip(), model, self.cfg.name, lat,
                                       attempt + 1, data.get("usage", {}))
                last_err = f"HTTP {status}: {r.text[:300]}"
                if status in (413, 431):
                    self._incr("failures")
                    self._set(last_status=status, last_error=last_err)
                    raise LLMError(self.cfg.name,
                                   f"request too large after context budgeting: {last_err}",
                                   status=status, attempts=attempt + 1)
                if status != 429 and status < 500:  # 400/401/403/404... will not fix themselves
                    self._incr("failures")
                    self._set(last_status=status, last_error=last_err)
                    raise LLMError(self.cfg.name, last_err, status=status, attempts=attempt + 1)
            if attempt < self.cfg.max_retries:
                self._incr("retries")
                self._set(last_status=status, last_error=last_err)
                self._sleep(self._backoff(attempt, retry_after))
        self._incr("failures")
        self._set(last_status=status, last_error=last_err)
        raise LLMError(self.cfg.name,
                       f"gave up after {self.cfg.max_retries + 1} attempts: {last_err}",
                       status=status, retryable=True, attempts=self.cfg.max_retries + 1)

    def list_models(self) -> list[str]:
        if not self.cfg.configured:
            raise LLMError(self.cfg.name, "API key not configured")
        try:
            r = self._http.get("models", headers=self._headers())
            r.raise_for_status()
            return sorted(m["id"] for m in r.json().get("data", []))
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise LLMError(self.cfg.name, f"could not list models: {exc!r}")
