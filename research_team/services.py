"""Dependency container wiring settings, workspace, memory, router, tools, runner and director."""
from __future__ import annotations

from dataclasses import dataclass

from .agents.deep_director import DeepDirector
from .agents.runner import SpecialistRunner
from .config import GROQ, NVIDIA, Settings
from .llm.client import LLMClient
from .llm.router import ModelRouter
from .memory import MemoryLayer
from .safety import PromptGuardGate
from .tools.toolkit import ResearchToolkit
from .workspace import Workspace


@dataclass
class Services:
    settings: Settings
    ws: Workspace
    memory: MemoryLayer
    router: ModelRouter
    toolkit: ResearchToolkit
    runner: SpecialistRunner
    director: DeepDirector
    safety: PromptGuardGate | None = None


def build_services(settings: Settings, *, llm_transports: dict | None = None, tool_transport=None,
                   model_factory=None, sleep=None) -> Services:
    llm_transports = llm_transports or {}
    kw = {"sleep": sleep} if sleep else {}
    clients = {GROQ: LLMClient(settings.groq, transport=llm_transports.get(GROQ),
                               request_budget=settings.groq_request_budget, **kw),
               NVIDIA: LLMClient(settings.nvidia, transport=llm_transports.get(NVIDIA),
                                  request_budget=16000, **kw)}
    ws = Workspace(settings.db_path)
    router = ModelRouter(settings, clients)
    toolkit = ResearchToolkit(settings.tool_keys, transport=tool_transport, sleep=sleep)
    safety = PromptGuardGate(router, ws)
    return Services(settings, ws, MemoryLayer(ws, enable_graph=settings.enable_knowledge_graph), router,
                    toolkit, SpecialistRunner(router, ws), DeepDirector(settings, toolkit, model_factory), safety)
