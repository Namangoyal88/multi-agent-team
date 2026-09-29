"""Context budgeting and compaction for provider-safe research calls.

The workflow keeps the original stage architecture, but every expensive model call is
built from a bounded context. Token counts here are deliberately conservative estimates
because the project supports multiple tokenizer families without adding tokenizer-heavy
dependencies.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Conservative approximation: English prose averages roughly 3-4 characters/token;
# using 3.5 gives useful headroom for JSON punctuation, IDs and code.
CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text or "") / CHARS_PER_TOKEN)) if text else 0


def truncate_text(text: str, max_tokens: int, *, marker: str = "\n...[truncated]...") -> str:
    text = text or ""
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    max_chars = max(32, int(max_tokens * CHARS_PER_TOKEN))
    keep = max(0, max_chars - len(marker))
    return text[:keep].rstrip() + marker


def compact_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text or "").strip()


@dataclass(frozen=True)
class BudgetConfig:
    director_input_tokens: int = 4500
    specialist_input_tokens: int = 5000
    verifier_input_tokens: int = 4000
    report_input_tokens: int = 5600
    safety_chunk_tokens: int = 400
    groq_request_tokens: int = 7600
    specialist_output_tokens: int = 1600
    verifier_output_tokens: int = 1400
    report_output_tokens: int = 2200
    director_output_tokens: int = 1600


class ContextBudgetManager:
    """Pack context in a stable order and enforce hard token targets."""

    def __init__(self, config: BudgetConfig | None = None):
        self.config = config or BudgetConfig()

    def evidence_block(self, papers: list[dict], *, limit: int = 12, abstract_chars: int = 650) -> str:
        rows = []
        for p in papers[:limit]:
            abstract = compact_whitespace(p.get("abstract") or "")
            if len(abstract) > abstract_chars:
                abstract = abstract[:abstract_chars].rstrip() + "..."
            rows.append(f"[{p.get('pid','?')}] {p.get('title','Untitled')} ({p.get('year','')}): {abstract}")
        return "\n".join(rows) or "(no papers retrieved)"

    def specialist_context(self, question: str, evidence: str, extra: str = "") -> str:
        base = f"QUESTION: {question}\n\nEVIDENCE:\n{evidence}"
        if extra:
            base += f"\n\nSTAGE NOTES:\n{extra}"
        # Preserve the question/evidence prefix first; later-stage notes are expendable.
        return truncate_text(base, self.config.specialist_input_tokens)

    def verifier_context(self, text: str) -> str:
        return truncate_text(text, self.config.verifier_input_tokens)

    def report_context(self, text: str) -> str:
        return truncate_text(text, self.config.report_input_tokens)

    def fit_messages(self, messages: list[dict], *, input_budget: int, output_tokens: int,
                     provider_budget: int | None = None) -> tuple[list[dict], bool]:
        """Return bounded messages; truncate only the final user message first.

        System instructions and task wording are preserved. This is a last-resort guard
        for code paths outside the main workflow too.
        """
        budget = provider_budget or input_budget + output_tokens
        current = sum(estimate_tokens(str(m.get("content", ""))) for m in messages) + output_tokens
        if current <= budget:
            return messages, False

        over = current - budget
        out = [dict(m) for m in messages]
        # Prefer trimming the last user message because that's where evidence/context lives.
        idxs = [i for i, m in reversed(list(enumerate(out))) if m.get("role") == "user"]
        for i in idxs:
            content = str(out[i].get("content", ""))
            allowed = max(200, estimate_tokens(content) - over - 5)
            new_content = truncate_text(content, allowed)
            saved = estimate_tokens(content) - estimate_tokens(new_content)
            out[i]["content"] = new_content
            over -= max(saved, 0)
            if over <= 0:
                return out, True

        # Last-resort trimming of non-system messages if a caller supplied unusually large text.
        for i, m in enumerate(out):
            if m.get("role") == "system":
                continue
            content = str(m.get("content", ""))
            if estimate_tokens(content) <= 200:
                continue
            allowed = max(100, estimate_tokens(content) - over - 5)
            new_content = truncate_text(content, allowed)
            saved = estimate_tokens(content) - estimate_tokens(new_content)
            out[i]["content"] = new_content
            over -= max(saved, 0)
            if over <= 0:
                break
        return out, True
