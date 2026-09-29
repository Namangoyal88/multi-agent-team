"""Explicit agent->model routing and NVIDIA escalation. NVIDIA is never the default specialist."""
from __future__ import annotations

from ..config import GROQ, NVIDIA, AgentSpec, Settings
from .client import LLMClient, LLMError, LLMResponse


class EscalationUnavailable(Exception):
    pass


class ModelRouter:
    def __init__(self, settings: Settings, clients: dict[str, LLMClient]):
        self.settings = settings
        self.clients = clients

    def spec(self, agent: str) -> AgentSpec:
        table = self.settings.routing()
        if agent not in table:
            raise KeyError(f"unknown agent {agent!r}")
        return table[agent]

    def call(self, agent: str, messages: list[dict], **kw) -> LLMResponse:
        """Call the model assigned to `agent` on its own provider (Groq for specialists)."""
        s = self.spec(agent)
        return self.clients[s.provider].chat(s.model, messages, **kw)

    def nvidia_available(self) -> bool:
        return self.settings.nvidia.configured and bool(self.settings.nvidia_model)

    def escalate(self, messages: list[dict], **kw) -> LLMResponse:
        """Send a hard task to NVIDIA Nemotron. Raises EscalationUnavailable if it cannot run."""
        if not self.nvidia_available():
            raise EscalationUnavailable("NVIDIA not configured (API key and/or NVIDIA_MODEL missing)")
        try:
            return self.clients[NVIDIA].chat(self.settings.nvidia_model, messages, **kw)
        except LLMError as exc:
            raise EscalationUnavailable(str(exc)) from exc

    def routing_table(self) -> list[dict]:
        rows = []
        for name, s in self.settings.routing().items():
            rows.append({"agent": name, "provider": s.provider, "model": s.model or "(not configured)",
                         "role": s.description,
                         "layer": "Deep Reasoning / Escalation" if s.provider == NVIDIA
                         else "Groq specialist"})
        return rows

    def provider_status(self, *, check_models: bool = False) -> dict:
        out = {}
        for name, client in self.clients.items():
            info = {"configured": client.cfg.configured, "base_url": client.cfg.base_url,
                    "diagnostics": dict(client.diagnostics)}
            if name == NVIDIA:
                info["model_configured"] = bool(self.settings.nvidia_model)
            if check_models and client.cfg.configured:
                try:
                    live = client.list_models()
                    wanted = ({m for m in self.settings.groq_models.values()} if name == GROQ
                              else {self.settings.nvidia_model} - {""})
                    info["model_check"] = {"reachable": True,
                                           "missing_ids": sorted(wanted - set(live))}
                except LLMError as exc:
                    info["model_check"] = {"reachable": False, "error": str(exc)}
            out[name] = info
        return out
