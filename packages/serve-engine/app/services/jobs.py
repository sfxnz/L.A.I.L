"""Background job runner with log files for SSE.

Job states: queued → running → completed | failed | cancelled.
"""
from __future__ import annotations

import asyncio
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from ..config import DATA_DIR
from .. import db
from .metadata import utc_now

_executor = ThreadPoolExecutor(max_workers=2)
# One cancel flag per job in this process; popped once the runner writes a terminal state.
_cancel_events: dict[str, threading.Event] = {}

TERMINAL_STATES = ("completed", "failed", "cancelled")
PROGRESS_WRITE_INTERVAL_S = 0.25


class JobCancelled(Exception):
    """Raised inside a job once its cancel flag is set."""


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


class JobLog:
    def __init__(self, job_id: str):
        self.path = DATA_DIR / "logs" / f"{job_id}.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def write(self, line: str) -> None:
        with self.path.open("a") as f:
            f.write(line.rstrip() + "\n")


def request_cancel(job_id: str) -> dict[str, Any] | None:
    """Flag a job for cancellation; returns the job row or None if unknown.

    The runner flips the row to `cancelled` at its next check (between bench waves,
    per SSE chunk). A queued/running row with no runner in this process (orphaned by
    a restart) is marked `cancelled` directly.
    """
    ev = _cancel_events.get(job_id)
    if ev is not None:
        ev.set()
        return db.get_job(job_id)
    job = db.get_job(job_id)
    if not job or job.get("status") in TERMINAL_STATES:
        return job
    db.upsert_job(
        job_id=job_id,
        kind=job["kind"],
        status="cancelled",
        created_at=job["created_at"],
        updated_at=utc_now(),
        progress=job.get("progress") or 0,
        message="cancelled (no runner in this process)",
        result={"cancelled": True},
        log_path=job.get("log_path"),
    )
    return db.get_job(job_id)


async def start_job(kind: str, fn: Callable[..., Any], **kwargs: Any) -> str:
    """Run blocking fn(log, progress, cancel, **kwargs) in the thread pool.

    The row is `queued` until the executor picks the job up; `progress()` writes to
    sqlite at most every PROGRESS_WRITE_INTERVAL_S (messages always reach the log).
    """
    job_id = new_job_id()
    log = JobLog(job_id)
    now = utc_now()
    cancel = threading.Event()
    _cancel_events[job_id] = cancel

    def write(status: str, progress: float, message: str, result: dict[str, Any] | None = None) -> None:
        db.upsert_job(
            job_id=job_id,
            kind=kind,
            status=status,
            created_at=now,
            updated_at=utc_now(),
            progress=progress,
            message=message,
            result=result,
            log_path=str(log.path),
        )

    write("queued", 0, "queued")
    last_write = 0.0

    def progress(p: float, msg: str = "") -> None:
        nonlocal last_write
        if msg:
            log.write(msg)
        mono = time.monotonic()
        if mono - last_write < PROGRESS_WRITE_INTERVAL_S:
            return
        last_write = mono
        write("running", p, msg)

    def runner() -> None:
        try:
            if cancel.is_set():
                raise JobCancelled("cancelled while queued")
            write("running", 0, "started")
            log.write(f"=== job {job_id} kind={kind} ===")
            result = fn(log=log, progress=progress, cancel=cancel, **kwargs)
            write(
                "completed",
                1.0,
                "done",
                result if isinstance(result, dict) else {"result": result},
            )
            log.write("=== completed ===")
        except JobCancelled as e:
            log.write(f"=== cancelled: {e} ===")
            write("cancelled", 0, str(e) or "cancelled", {"cancelled": True})
        except Exception as e:
            log.write(f"ERROR: {e}")
            log.write(traceback.format_exc())
            write("failed", 0, str(e), {"error": str(e)})
        finally:
            _cancel_events.pop(job_id, None)

    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, runner)
    return job_id


def read_log_tail(path: str | Path, offset: int = 0) -> tuple[str, int]:
    """Bytes appended since `offset` (seek, not a full re-read). Restarts on truncation."""
    p = Path(path)
    if not p.exists():
        return "", 0
    size = p.stat().st_size
    if offset > size:
        offset = 0
    if size == offset:
        return "", offset
    with p.open("rb") as f:
        f.seek(offset)
        data = f.read()
    return data.decode("utf-8", errors="replace"), offset + len(data)
