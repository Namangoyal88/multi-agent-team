# MULTI AGENT AI RESEARCH TEAM

A local-first multi-agent research system. An NVIDIA Nemotron 3 Ultra Deep Research Director scopes and delegates complex research; Groq models perform specialist stages; LangGraph owns stateful execution; the verifier gates evidence; and the Report Generator produces a grounded research package. SQLite stores jobs, papers, artifacts, events, and research memory.

## Architecture

```text
User question
   ↓
NVIDIA Nemotron 3 Ultra — Deep Research Director
   ↓ task delegation
Groq specialists
   ├─ Researcher
   ├─ Mathematician
   ├─ ML Engineer
   ├─ Experimenter
   ├─ Reviewer
   ├─ Verifier
   └─ Report Generator
   ↓
Research tools
   ├─ arXiv
   ├─ Semantic Scholar
   ├─ PubMed
   ├─ Tavily web search
   ├─ GitHub
   └─ Kaggle
   ↓
SQLite workspace + memory
   ↓
LangGraph workflow
   ↓
Claim/evidence verification
   ↓
Grounded final report
   ↓
Evaluation → optional evidence refinement
```

## Model routing

| Role | Provider | Model |
|---|---|---|
| Deep Director + escalation | NVIDIA | `nvidia/nemotron-3-ultra-550b-a55b` |
| Researcher | Groq | `openai/gpt-oss-20b` |
| Mathematician | Groq | `openai/gpt-oss-120b` |
| ML Engineer | Groq | `openai/gpt-oss-120b` |
| Experimenter / safety gate | Groq | `openai/gpt-oss-120b` |
| Reviewer | Groq | `openai/gpt-oss-20b` |
| Verifier / evidence safety gate | Groq | `openai/gpt-oss-20b` |
| Report Generator | Groq | `qwen/qwen3.8-27b` |

## Setup — Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env

uvicorn research_team.api:get_app --factory --port 8000
# second terminal
streamlit run frontend\streamlit_app.py
```

For reproducible installs with uv:

```powershell
uv sync --locked
```

## Configuration

The built-in dotenv loader accepts unquoted values and matching single/double quotes. Quotes are removed before values enter the provider configuration, so both of these are valid:

```env
NVIDIA_MODEL=nvidia/nemotron-3-ultra-550b-a55b
NVIDIA_MODEL='nvidia/nemotron-3-ultra-550b-a55b'
```

Never commit real API keys.

`ENABLE_CODE_EXECUTION=false` is the safe default. When enabled, the experiment runner uses AST validation, a restricted builtins set, a restricted standard-library import allow-list, isolated working directory, and a subprocess timeout. This is defense-in-depth, not a complete security boundary; hostile code should run in a container/VM.

## Workflow rules

- The NVIDIA Director decides which optional stages are needed; it does not run every specialist for every question.
- `verify`, `report`, and `evaluate` are mandatory workflow stages.
- Retrieval now adds query understanding, targeted query generation, candidate deduplication, relevance/entity/date scoring, and a final 8–12 paper evidence gate before expensive specialist calls.
- Context is budgeted before provider calls (Director ≈4.5K, specialists ≤5K, verifier ≤4K, report ≈5.6K input tokens) with a second provider-level request-size guard.
- GPT-OSS/Qwen specialist calls use JSON Object Mode plus a common normalize/repair retry; list-form Director/Specialist responses are recoverable when structurally unambiguous.
- The `experiment_design` stage remains in the same LangGraph position but now implements the requested design → Prompt Guard 86M → sandbox execution → Reviewer flow.
- Verification is deterministic-first, then Prompt Guard 22M screens untrusted evidence, then Reviewer 20B handles semantic claim/evidence judgment where needed.
- Final factual report sections are produced from validated claims. Raw stage output is never used as a silent report fallback.
- Executed experiment results are reported only from an executed sandbox run and its `METRICS_JSON` payload.

## Tests

```powershell
pytest -q
```

The test suite uses mock HTTP transports so Groq/NVIDIA/tool tests do not consume API quota. Live provider smoke tests must be run from an environment with outbound HTTPS access.

## Provider compatibility

NVIDIA documents `nvidia/nemotron-3-ultra-550b-a55b` as an OpenAI-compatible chat-completions model with reasoning and tool-use support. The project creates it as a `ChatOpenAI` instance with Chat Completions forced (`use_responses_api=False`) and NVIDIA's documented `chat_template_kwargs` for thinking/tool-call content.

Groq exposes the configured Qwen/GPT-OSS models through its OpenAI-compatible Chat API. The project keeps provider routing explicit: Groq handles specialist work and NVIDIA is reserved for the Deep Director/escalation path.

## Retrieval and quota hardening

The active paper registry is replaced on each evidence pass so refinement cannot accumulate stale PIDs.
The retrieval layer can collect multiple adapter results internally, but promotes at most 48 ranked candidates and 12 papers to the downstream workspace.

Groq's current free-plan rate-limit table lists 8K TPM for GPT-OSS 20B, GPT-OSS 120B, and Qwen 3.8 27B; the project therefore defaults to a 7,600-token request budget plus smaller stage-specific input/output budgets. Groq also documents JSON Object/Schema support for the GPT-OSS and Qwen models used here.
