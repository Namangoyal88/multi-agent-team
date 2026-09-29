"""FastAPI backend. POST /research returns immediately with a research ID; poll for progress."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .config import load_settings
from .jobs import JobManager
from .services import Services, build_services


class ResearchRequest(BaseModel):
    question: str = Field(min_length=8)
    enable_experiments: bool = False
    options: dict = Field(default_factory=dict)


def create_app(svc: Services | None = None) -> FastAPI:
    svc = svc or build_services(load_settings())
    jobs = JobManager(svc)
    app = FastAPI(title="MULTI AGENT AI RESEARCH TEAM")
    app.state.svc, app.state.jobs = svc, jobs

    def job_or_404(rid: str) -> dict:
        j = svc.ws.get_job(rid)
        if not j:
            raise HTTPException(404, "unknown research id")
        return j

    @app.post("/research", status_code=202)
    def start(req: ResearchRequest):
        opts = {**req.options, "enable_experiments": req.enable_experiments}
        return {"research_id": jobs.start(req.question, opts), "status": "queued"}

    @app.get("/research")
    def list_research():
        return svc.ws.list_jobs()

    @app.get("/research/{rid}")
    def status(rid: str):
        j = job_or_404(rid)
        state = svc.ws.load_state(rid) or {}
        done = state.get("done", [])
        stages = (state.get("plan") or {}).get("stages", [])
        total = len(set(stages) | {"verify", "report", "evaluate"}) + 1
        return {"research_id": rid, "status": j["status"], "error": j["error"],
                "question": j["request"]["question"], "plan": state.get("plan"),
                "plan_source": state.get("plan_source"), "stages_done": done,
                "progress": 1.0 if j["status"] == "completed" else round(min(len(done) + (1 if state.get("plan") else 0), total) / total, 2),
                "errors": state.get("errors", []), "evaluation": state.get("evaluation"),
                "agents": svc.ws.agent_runs(rid)}

    @app.get("/research/{rid}/events")
    def events(rid: str, after: int = Query(0)):
        job_or_404(rid)
        return svc.ws.events(rid, after)

    @app.get("/research/{rid}/report")
    def report(rid: str):
        job_or_404(rid)
        md = svc.ws.get_artifact(rid, "report", "final.md")
        if md is None:
            raise HTTPException(409, "report not ready")
        return {"research_id": rid, "markdown": md}

    @app.get("/research/{rid}/papers")
    def papers(rid: str):
        job_or_404(rid)
        return svc.ws.papers(rid)

    @app.get("/research/{rid}/artifacts")
    def artifacts(rid: str, kind: str | None = None):
        job_or_404(rid)
        return svc.ws.get_artifacts(rid, kind)

    @app.get("/memory/search")
    def memory_search(q: str, k: int = 5):
        return {"results": svc.memory.recall(q, k), "stats": svc.memory.stats()}

    @app.get("/health")
    def health():
        return {"status": "ok", "nvidia_available": svc.router.nvidia_available(),
                "groq_configured": svc.settings.groq.configured,
                "code_execution_enabled": svc.settings.enable_code_execution,
                "tools": svc.toolkit.status(), "memory": svc.memory.stats()}

    @app.get("/models")
    def models():
        return svc.router.routing_table()

    @app.get("/providers")
    def providers(check: bool = False):
        return svc.router.provider_status(check_models=check)

    return app


def get_app() -> FastAPI:  # uvicorn research_team.api:get_app --factory
    return create_app()
