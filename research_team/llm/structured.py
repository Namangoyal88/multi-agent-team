"""Structured-output helpers: tolerant JSON extraction + pydantic validation."""
from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    pass


def extract_json(text: str):
    """Return the first JSON object/array found in text. Tries the raw text first, so code fences
    that live *inside* a JSON string value are never mistaken for a wrapper fence."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    wrapped = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if wrapped:
        try:
            return json.loads(wrapped.group(1))
        except ValueError:
            text = wrapped.group(1)
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(text[i:])
                return obj
            except ValueError:
                continue
    raise StructuredOutputError("no JSON found in model output")


def _normalize_for_model(data, model: type[T]):
    """Recover common near-miss shapes before schema validation.

    Models occasionally return a list of labels/text blocks instead of the requested object.
    We normalize only safe, structurally obvious cases; ambiguous data still fails closed.
    """
    name = getattr(model, "__name__", "")
    if name == "Plan":
        if isinstance(data, list):
            stages, research_types = [], []
            rationale_parts = []
            for item in data:
                if isinstance(item, str):
                    if item in {"literature_review", "literature"}:
                        research_types.append("literature_review")
                        stages.append("literature")
                    elif item in {"analysis", "gaps", "math", "experiment_design", "ml_impl", "experiment_run", "review"}:
                        stages.append(item)
                    else:
                        rationale_parts.append(item)
                elif isinstance(item, dict):
                    value = item.get("stage") or item.get("type") or item.get("name")
                    if isinstance(value, str) and value in {"literature", "analysis", "gaps", "math", "experiment_design", "ml_impl", "experiment_run", "review"}:
                        stages.append(value)
            if not research_types and "literature" in stages:
                research_types = ["literature_review"]
            return {"complexity": "simple" if stages == ["literature"] else "complex",
                    "research_types": research_types, "queries": [], "stages": list(dict.fromkeys(stages)),
                    "rationale": "Recovered Director output from a list-form response. " + " ".join(rationale_parts)[:500],
                    "success_criteria": []}
        if isinstance(data, dict) and "plan" in data and isinstance(data["plan"], dict):
            return data["plan"]
    if name == "SpecialistOutput":
        if isinstance(data, list):
            bits = []
            for item in data:
                if isinstance(item, str):
                    bits.append(item)
                elif isinstance(item, dict):
                    bits.append(str(item.get("content") or item.get("text") or "").strip())
            bits = [b for b in bits if b]
            return {"content": "\n\n".join(bits), "confidence": 0.5, "needs_escalation": False,
                    "escalation_reason": "", "claims": []}
        if isinstance(data, dict) and "content" not in data and "text" in data:
            return {**data, "content": data.get("text", "")}
    return data


def parse_model(text: str, model: type[T]) -> T:
    try:
        data = _normalize_for_model(extract_json(text), model)
        return model.model_validate(data)
    except ValidationError as exc:
        raise StructuredOutputError(f"schema validation failed: {exc.errors()[:3]}") from exc
