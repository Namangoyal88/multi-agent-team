"""Shared research workspace: SQLite-backed jobs, events, papers, artifacts, agent runs, state."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, status TEXT, request TEXT, error TEXT,
  created REAL, updated REAL);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, ts REAL,
  agent TEXT, kind TEXT, message TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS papers(job_id TEXT, pid TEXT, data TEXT, PRIMARY KEY(job_id, pid));
CREATE TABLE IF NOT EXISTS artifacts(job_id TEXT, kind TEXT, name TEXT, data TEXT, updated REAL,
  PRIMARY KEY(job_id, kind, name));
CREATE TABLE IF NOT EXISTS agent_runs(job_id TEXT, run_id TEXT, data TEXT, PRIMARY KEY(job_id, run_id));
CREATE TABLE IF NOT EXISTS long_term(id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, kind TEXT,
  text TEXT, meta TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS vectors(id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, kind TEXT,
  text TEXT, vec TEXT, meta TEXT);
CREATE TABLE IF NOT EXISTS graph_edges(src TEXT, rel TEXT, dst TEXT, job_id TEXT,
  UNIQUE(src, rel, dst));
"""


class Workspace:
    def __init__(self, db_path: str):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def _exec(self, sql: str, args: tuple = ()):
        with self._lock:
            cur = self._conn.execute(sql, args)
            self._conn.commit()
            return cur

    def _all(self, sql: str, args: tuple = ()):
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    # ---- jobs ------------------------------------------------------------
    def create_job(self, request: dict) -> str:
        jid = uuid.uuid4().hex[:12]
        now = time.time()
        self._exec("INSERT INTO jobs VALUES(?,?,?,?,?,?)",
                   (jid, "queued", json.dumps(request), None, now, now))
        return jid

    def update_job(self, jid: str, status: str, error: str | None = None):
        self._exec("UPDATE jobs SET status=?, error=?, updated=? WHERE id=?",
                   (status, error, time.time(), jid))

    def get_job(self, jid: str) -> dict | None:
        rows = self._all("SELECT * FROM jobs WHERE id=?", (jid,))
        if not rows:
            return None
        r = dict(rows[0])
        r["request"] = json.loads(r["request"])
        return r

    def list_jobs(self, limit: int = 50) -> list[dict]:
        rows = self._all("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,))
        return [{**dict(r), "request": json.loads(r["request"])} for r in rows]

    # ---- events ----------------------------------------------------------
    def log(self, jid: str, agent: str, kind: str, message: str, payload: dict | None = None):
        self._exec("INSERT INTO events(job_id, ts, agent, kind, message, payload) VALUES(?,?,?,?,?,?)",
                   (jid, time.time(), agent, kind, message, json.dumps(payload or {})))

    def events(self, jid: str, after: int = 0) -> list[dict]:
        rows = self._all("SELECT * FROM events WHERE job_id=? AND id>? ORDER BY id", (jid, after))
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    # ---- papers (the only valid citation sources) ------------------------
    def add_papers(self, jid: str, papers: list[dict]) -> list[dict]:
        with self._lock:
            existing = self.papers(jid)
            seen = {self._key(p): p for p in existing}
            n = len(existing)
            for p in papers:
                k = self._key(p)
                if k in seen:
                    continue
                n += 1
                p = {**p, "pid": f"P{n}"}
                seen[k] = p
                self._conn.execute("INSERT INTO papers VALUES(?,?,?)", (jid, p["pid"], json.dumps(p)))
            self._conn.commit()
        return self.papers(jid)

    def replace_papers(self, jid: str, papers: list[dict]) -> list[dict]:
        """Replace the active evidence registry with the latest ranked set.

        Refinement reruns the literature stage, so keeping stale/irrelevant papers would make
        P1..P52 accumulate and would also leave old citations in the report. Replacing the active
        registry keeps citation IDs compact and ensures downstream agents see one curated evidence set.
        """
        with self._lock:
            self._conn.execute("DELETE FROM papers WHERE job_id=?", (jid,))
            seen = set()
            n = 0
            for raw in papers:
                p = dict(raw)
                key = self._key(p)
                if not key or key in seen:
                    continue
                seen.add(key)
                n += 1
                p["pid"] = f"P{n}"
                self._conn.execute("INSERT INTO papers VALUES(?,?,?)", (jid, p["pid"], json.dumps(p)))
            self._conn.commit()
        return self.papers(jid)

    @staticmethod
    def _key(p: dict) -> str:
        return (p.get("doi") or p.get("arxiv_id") or p.get("url") or p.get("title", "")).lower().strip()

    def papers(self, jid: str) -> list[dict]:
        rows = self._all("SELECT data FROM papers WHERE job_id=?", (jid,))
        out = [json.loads(r["data"]) for r in rows]
        return sorted(out, key=lambda p: int(p["pid"][1:]))

    # ---- artifacts / state / agent runs ---------------------------------
    def put_artifact(self, jid: str, kind: str, name: str, data):
        self._exec("INSERT OR REPLACE INTO artifacts VALUES(?,?,?,?,?)",
                   (jid, kind, name, json.dumps(data, default=str), time.time()))

    def get_artifacts(self, jid: str, kind: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM artifacts WHERE job_id=?", [jid]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        return [{"kind": r["kind"], "name": r["name"], "data": json.loads(r["data"]),
                 "updated": r["updated"]} for r in self._all(sql + " ORDER BY updated", tuple(args))]

    def get_artifact(self, jid: str, kind: str, name: str):
        rows = self._all("SELECT data FROM artifacts WHERE job_id=? AND kind=? AND name=?",
                         (jid, kind, name))
        return json.loads(rows[0]["data"]) if rows else None

    def save_state(self, jid: str, state: dict):
        self.put_artifact(jid, "state", "latest", state)

    def load_state(self, jid: str) -> dict | None:
        return self.get_artifact(jid, "state", "latest")

    def upsert_agent_run(self, jid: str, run_id: str, data: dict):
        self._exec("INSERT OR REPLACE INTO agent_runs VALUES(?,?,?)", (jid, run_id, json.dumps(data)))

    def agent_runs(self, jid: str) -> list[dict]:
        return [json.loads(r["data"]) for r in
                self._all("SELECT data FROM agent_runs WHERE job_id=? ORDER BY rowid", (jid,))]
