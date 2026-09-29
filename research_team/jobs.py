"""Background research execution. start() returns a research ID immediately; work runs in a thread."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .services import Services
from .workflow import make_workflow


class JobManager:
    def __init__(self, svc: Services, max_workers: int = 2):
        self.svc = svc
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="research")
        self.graph = make_workflow(svc)
        self.futures: dict[str, object] = {}

    def start(self, question: str, options: dict | None = None) -> str:
        options = options or {}
        jid = self.svc.ws.create_job({"question": question, "options": options})
        self.svc.ws.log(jid, "system", "queued", "Research job queued")
        self.futures[jid] = self.pool.submit(self._run, jid, question, options)
        return jid

    def _run(self, jid: str, question: str, options: dict):
        ws = self.svc.ws
        ws.update_job(jid, "running")
        try:
            final = self.graph.invoke({"job_id": jid, "question": question, "options": options,
                                       "done": [], "errors": [], "claims": [], "claim_seq": 0},
                                      {"recursion_limit": 60})
            ws.save_state(jid, final)
            ws.update_job(jid, "completed")
            ws.log(jid, "system", "completed", "Research finished")
        except Exception as exc:  # recorded, never silent
            ws.update_job(jid, "failed", f"{type(exc).__name__}: {exc}")
            ws.log(jid, "system", "failed", f"{type(exc).__name__}: {exc}")
        finally:
            self.svc.memory.short_clear(jid)

    def wait(self, jid: str, timeout: float | None = None):
        f = self.futures.get(jid)
        if f:
            f.result(timeout)
