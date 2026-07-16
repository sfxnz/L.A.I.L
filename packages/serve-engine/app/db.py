"""SQLite run index."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import DB_PATH


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                intent TEXT,
                model_id TEXT,
                summary_json TEXT,
                path TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                progress REAL DEFAULT 0,
                message TEXT,
                result_json TEXT,
                log_path TEXT
            )
            """
        )
        c.commit()


def insert_run(
    run_id: str,
    created_at: str,
    kind: str,
    intent: str | None,
    model_id: str | None,
    summary: dict[str, Any],
    path: str,
) -> None:
    with _conn() as c:
        c.execute(
            """
            INSERT OR REPLACE INTO runs
            (run_id, created_at, kind, intent, model_id, summary_json, path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                created_at,
                kind,
                intent,
                model_id,
                json.dumps(summary),
                path,
            ),
        )
        c.commit()


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["summary"] = json.loads(d.pop("summary_json") or "{}")
        out.append(d)
    return out


def get_run(run_id: str) -> dict[str, Any] | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["summary"] = json.loads(d.pop("summary_json") or "{}")
    return d


def upsert_job(
    job_id: str,
    kind: str,
    status: str,
    created_at: str,
    updated_at: str,
    progress: float = 0,
    message: str = "",
    result: dict[str, Any] | None = None,
    log_path: str | None = None,
) -> None:
    with _conn() as c:
        c.execute(
            """
            INSERT INTO jobs
            (job_id, kind, status, created_at, updated_at, progress, message, result_json, log_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
              status=excluded.status,
              updated_at=excluded.updated_at,
              progress=excluded.progress,
              message=excluded.message,
              result_json=excluded.result_json,
              log_path=COALESCE(excluded.log_path, jobs.log_path)
            """,
            (
                job_id,
                kind,
                status,
                created_at,
                updated_at,
                progress,
                message,
                json.dumps(result) if result is not None else None,
                log_path,
            ),
        )
        c.commit()


def get_job(job_id: str) -> dict[str, Any] | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    raw = d.pop("result_json")
    d["result"] = json.loads(raw) if raw else None
    return d


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        raw = d.pop("result_json")
        d["result"] = json.loads(raw) if raw else None
        out.append(d)
    return out


def load_envelope(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())
