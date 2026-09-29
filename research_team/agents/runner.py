"""Runs one specialist call: Groq model -> validated structured output -> optional NVIDIA escalation.
Every run is recorded in the workspace so the dashboard can show model/provider/status/timing."""
from __future__ import annotations

import time
import uuid

from pydantic import BaseModel, Field

from ..config import NVIDIA
from ..context_budget import BudgetConfig, ContextBudgetManager
from ..llm.client import LLMError
from ..llm.router import EscalationUnavailable, ModelRouter
from ..llm.structured import StructuredOutputError, parse_model
from ..workspace import Workspace
from .registry import OUTPUT_CONTRACT, PROMPTS


class Claim(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)


class SpecialistOutput(BaseModel):
    content: str
    confidence: float = 0.5
    needs_escalation: bool = False
    escalation_reason: str = ""
    claims: list[Claim] = Field(default_factory=list)


class SpecialistRunner:
    def __init__(self, router: ModelRouter, ws: Workspace):
        self.router, self.ws = router, ws
        self.threshold = router.settings.escalation_threshold
        self.budget = ContextBudgetManager(BudgetConfig(
            specialist_input_tokens=router.settings.specialist_input_budget,
            verifier_input_tokens=router.settings.verifier_input_budget,
            report_input_tokens=router.settings.report_input_budget,
            groq_request_tokens=router.settings.groq_request_budget))

    def _record(self, job_id, run_id, data):
        self.ws.upsert_agent_run(job_id, run_id, data)

    def run(self, job_id: str, agent: str, task: str, context: str = "",
            tools_used: list[str] | None = None) -> tuple[SpecialistOutput | None, str | None]:
        """Returns (output, error). Never raises for provider problems; failures are recorded."""
        spec = self.router.spec(agent)
        run_id = uuid.uuid4().hex[:8]
        rec = {"run_id": run_id, "agent": agent, "provider": spec.provider, "model": spec.model,
               "task": task[:200], "status": "running", "progress": 0.1,
               "tools_used": tools_used or [], "output": "", "escalated": False,
               "started": time.time(), "duration_s": None, "error": None}
        self._record(job_id, run_id, rec)
        self.ws.log(job_id, agent, "start", task[:200])
        bounded_context = self.budget.specialist_context(
            self._clip_task(task), context, "")
        messages = [{"role": "system", "content": PROMPTS[agent] + "\n\n" + OUTPUT_CONTRACT},
                    {"role": "user", "content": f"TASK:\n{task}\n\nCONTEXT:\n{bounded_context}"}]
        out, err = None, None
        try:
            out = self._call_structured(agent, messages)
            rec["progress"] = 0.7
        except (LLMError, StructuredOutputError) as exc:
            err = str(exc)
            self.ws.log(job_id, agent, "error", err)
            out = (None if agent in {"experimenter", "verifier"}
                   else self._escalate_failure(job_id, agent, task, context, err, rec))
            if out:
                err = None
        if out and agent not in {"experimenter", "verifier"} and \
                (out.needs_escalation or out.confidence < self.threshold) and not rec["escalated"]:
            out = self._escalate(job_id, agent, task, context, out, rec) or out
        rec.update(status="failed" if out is None else "done", progress=1.0,
                   duration_s=round(time.time() - rec["started"], 2), error=err,
                   output=(out.content[:400] if out else ""))
        self._record(job_id, run_id, rec)
        self.ws.log(job_id, agent, "finish" if out else "failed", rec["status"],
                    {"escalated": rec["escalated"], "error": err})
        return out, err

    @staticmethod
    def _clip_task(task: str) -> str:
        return (task or "").strip()[:1800]

    def _call_structured(self, agent, messages) -> SpecialistOutput:
        bounded, truncated = self.budget.fit_messages(
            messages, input_budget=self.budget.config.specialist_input_tokens,
            output_tokens=self.budget.config.specialist_output_tokens,
            provider_budget=self.budget.config.groq_request_tokens)
        # GPT-OSS/Qwen on Groq support JSON Object/Schema modes. JSON Object Mode is
        # deliberately used here because the same runner also handles heterogeneous
        # specialist schemas and a recovery path remains in place for malformed output.
        resp = self.router.call(agent, bounded, max_tokens=self.budget.config.specialist_output_tokens,
                                response_format={"type": "json_object"})
        try:
            result = parse_model(resp.content, SpecialistOutput)
            return result
        except StructuredOutputError as first_exc:
            # Repair retry contains only the failed output + compact contract, not the full evidence
            # context, so a formatting failure cannot recreate an oversized request.
            compact = [{"role": "system", "content":
                        OUTPUT_CONTRACT + "\nReturn exactly one JSON object; no markdown, no array."},
                       {"role": "user", "content":
                        "Repair this model output into the contract without adding facts:\n" + resp.content[:6000]}]
            try:
                repaired = self.router.call(agent, compact, temperature=0.0,
                                            max_tokens=1200,
                                            response_format={"type": "json_object"})
                return parse_model(repaired.content, SpecialistOutput)
            except Exception as exc:
                raise StructuredOutputError(f"structured output failed after recovery: {first_exc}; {exc}") from exc

    def _escalate(self, job_id, agent, task, context, draft, rec):
        msgs = [{"role": "system", "content": "You are the deep-reasoning escalation model for a research "
                 "team. Improve on the specialist draft; resolve conflicts; do not invent evidence. "
                 "Cite only provided paper IDs.\n\n" + OUTPUT_CONTRACT},
                {"role": "user", "content": f"AGENT: {agent}\nTASK:\n{task}\nREASON: "
                 f"{draft.escalation_reason or 'low confidence'}\nSPECIALIST DRAFT:\n{draft.content}\n\n"
                 f"CONTEXT:\n{context}"}]
        try:
            resp = self.router.escalate(msgs)
            out = parse_model(resp.content, SpecialistOutput)
        except (EscalationUnavailable, StructuredOutputError) as exc:
            self.ws.log(job_id, agent, "escalation_unavailable", str(exc))
            rec["escalation_error"] = str(exc)
            return None
        rec["escalated"] = True
        self.ws.log(job_id, NVIDIA, "escalation", f"{agent} task handled by Nemotron")
        return out

    def _escalate_failure(self, job_id, agent, task, context, err, rec):
        draft = SpecialistOutput(content="(specialist failed: " + err[:200] + ")", confidence=0.0,
                                 needs_escalation=True, escalation_reason="specialist failure")
        return self._escalate(job_id, agent, task, context, draft, rec)
