"""Settings and the explicit agent -> provider/model routing table."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

GROQ = "groq"
NVIDIA = "nvidia"

SPECIALISTS = ("researcher", "mathematician", "ml_engineer", "experimenter",
               "reviewer", "verifier", "report_generator")


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str = ""
    timeout: float = 90.0
    max_retries: int = 4

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    provider: str
    model: str
    description: str


def _parse_dotenv_value(value: str) -> str:
    """Parse the small dotenv subset used by this project.

    Matching single/double quotes are removed. An unquoted inline comment is
    removed only when separated by whitespace so secrets containing ``#`` stay intact.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return re.sub(r"\s+#.*$", "", value).strip()


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), _parse_dotenv_value(v))


@dataclass
class Settings:
    groq: ProviderConfig
    nvidia: ProviderConfig
    nvidia_model: str
    groq_models: dict[str, str]
    db_path: str = "data/research_team.db"
    enable_code_execution: bool = False
    enable_knowledge_graph: bool = True
    escalation_threshold: float = 0.5
    max_refine_iterations: int = 1
    min_papers: int = 3
    sandbox_timeout: int = 30
    groq_request_budget: int = 7600
    specialist_input_budget: int = 5000
    verifier_input_budget: int = 4000
    report_input_budget: int = 5600
    tool_keys: dict[str, str] = field(default_factory=dict)

    def routing(self) -> dict[str, AgentSpec]:
        desc = {
            "researcher": "Literature discovery, paper summaries, related work, datasets, gaps",
            "mathematician": "Formulation, derivations, proofs, theoretical analysis",
            "ml_engineer": "ML architecture, algorithms, preprocessing, code, evaluation method",
            "experimenter": "Experiment design, baselines, ablations, execution, metrics",
            "reviewer": "Critical review of methodology, novelty, reproducibility",
            "verifier": "Fact/citation checking and claim-evidence matching",
            "report_generator": "Integrates validated findings into the final report",
        }
        table = {a: AgentSpec(a, GROQ, self.groq_models[a], desc[a]) for a in SPECIALISTS}
        table["director"] = AgentSpec(
            "director", NVIDIA, self.nvidia_model,
            "Deep Research Director / Deep Reasoning & Escalation (NVIDIA Nemotron 3 Ultra)")
        return table


def load_settings(env: dict[str, str] | None = None, *, dotenv: bool = True) -> Settings:
    if dotenv and env is None:
        _load_dotenv()
    e = env if env is not None else os.environ
    g = lambda k, d="": e.get(k, d)  # noqa: E731
    defaults = {
        "researcher": "openai/gpt-oss-20b",
        "mathematician": "openai/gpt-oss-120b",
        "ml_engineer": "openai/gpt-oss-120b",
        "experimenter": "openai/gpt-oss-120b",
        "reviewer": "openai/gpt-oss-20b",
        "verifier": "openai/gpt-oss-20b",
        "report_generator": "qwen/qwen3.8-27b",
    }
    env_key = {"report_generator": "GROQ_REPORT_MODEL"}
    models = {a: g(env_key.get(a, f"GROQ_{a.upper()}_MODEL"), d) for a, d in defaults.items()}
    return Settings(
        groq=ProviderConfig(GROQ, "https://api.groq.com/openai/v1", g("GROQ_API_KEY")),
        nvidia=ProviderConfig(NVIDIA, g("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                              g("NVIDIA_API_KEY"), timeout=180.0),
        nvidia_model=g("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
        groq_models=models,
        db_path=g("RESEARCH_DB_PATH", "data/research_team.db"),
        enable_code_execution=g("ENABLE_CODE_EXECUTION", "false").lower() == "true",
        enable_knowledge_graph=g("ENABLE_KNOWLEDGE_GRAPH", "true").lower() == "true",
        escalation_threshold=float(g("ESCALATION_CONFIDENCE_THRESHOLD", "0.5")),
        max_refine_iterations=int(g("MAX_REFINE_ITERATIONS", "1")),
        min_papers=int(g("MIN_PAPERS", "3")),
        sandbox_timeout=int(g("SANDBOX_TIMEOUT", "30")),
        groq_request_budget=int(g("GROQ_REQUEST_BUDGET", "7600")),
        specialist_input_budget=int(g("SPECIALIST_INPUT_BUDGET", "5000")),
        verifier_input_budget=int(g("VERIFIER_INPUT_BUDGET", "4000")),
        report_input_budget=int(g("REPORT_INPUT_BUDGET", "5600")),
        tool_keys={k: g(k) for k in ("SEMANTIC_SCHOLAR_API_KEY", "TAVILY_API_KEY", "GITHUB_TOKEN",
                                     "KAGGLE_USERNAME", "KAGGLE_KEY", "NCBI_API_KEY")},
    )
