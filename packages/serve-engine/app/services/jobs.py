"""Background job runner with log files for SSE.

Job states: queued → running → completed | failed | cancelled.

Jobs run on one of three single-worker executors, chosen by kind:
  · lifecycle — serve (a multi-node load can poll for 30 min);
  · stop      — never queues behind a load or a bench, and cancels queued loads;
  · bench     — every other kind (agentic evals): one at a time, so two benches
                never measure a GPU they share.
The controller's streams benches run outside this process; they take the bench
lease (`acquire_lease`) so they and a bench job never overlap either.
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

LIFECYCLE_KINDS = frozenset({"serve"})
_executors: dict[str, Any] = {
    "lifecycle": ThreadPoolExecutor(max_workers=1, thread_name_prefix="job-lifecycle"),
    "stop": ThreadPoolExecutor(max_workers=1, thread_name_prefix="job-stop"),
    "bench": ThreadPoolExecutor(max_workers=1, thread_name_prefix="job-bench"),
}
# One cancel flag per job in this process; popped once the runner writes a terminal state.
_cancel_events: dict[str, threading.Event] = {}
# Kind of every job that has a runner in this process (queued or running).
_live_kinds: dict[str, str] = {}
_state_lock = threading.Lock()
# External bench lease (the controller's streams bench): {"id", "owner", "expires"} (monotonic).
_lease: dict[str, Any] | None = None

TERMINAL_STATES = ("completed", "failed", "cancelled")
PROGRESS_WRITE_INTERVAL_S = 0.25


def job_class(kind: str) -> str:
    if kind == "stop":
        return "stop"
    return "lifecycle" if kind in LIFECYCLE_KINDS else "bench"


class BenchBusy(Exception):
    """A bench job or the external bench lease already holds the GPU."""

    def __init__(self, holder: dict[str, Any]):
        super().__init__(f"bench busy: {holder}")
        self.holder = holder


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

    The runner flips the row to `cancelled` at its next check: before each golden case;
    for tool-eval, a watcher terminates the subprocess group within ~0.5 s. Serve jobs do
    not honour cancel yet. A queued/running row with no runner in this process (orphaned
    by a restart) is marked `cancelled` directly.
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


def is_live(job_id: str) -> bool:
    """True while this process has a runner (queued or running) for the job."""
    return job_id in _cancel_events


def _active_lease() -> dict[str, Any] | None:
    if _lease is not None and _lease["expires"] > time.monotonic():
        return _lease
    return None


def acquire_lease(lease_id: str, owner: str, ttl_s: float) -> dict[str, Any] | None:
    """Take or renew the external bench lease. Returns the conflicting holder, or None."""
    global _lease
    with _state_lock:
        for jid, kind in _live_kinds.items():
            if job_class(kind) == "bench":
                return {"job_id": jid, "kind": kind}
        cur = _active_lease()
        if cur is not None and cur["id"] != lease_id:
            return {"lease_id": cur["id"], "owner": cur["owner"]}
        _lease = {"id": lease_id, "owner": owner, "expires": time.monotonic() + ttl_s}
        return None


def release_lease(lease_id: str) -> None:
    global _lease
    with _state_lock:
        if _lease is not None and _lease["id"] == lease_id:
            _lease = None


async def start_job(kind: str, fn: Callable[..., Any], **kwargs: Any) -> str:
    """Run blocking fn(log, progress, cancel, **kwargs) on the executor for `kind`.

    The row is `queued` until the executor picks the job up. `progress()` writes to
    sqlite at most every PROGRESS_WRITE_INTERVAL_S; a throttled update is flushed when
    the interval ends, so the last message always lands (messages always reach the log).
    A bench-class job raises BenchBusy while the external bench lease is held; a stop
    job cancels serve jobs that are queued or running.
    """
    cls = job_class(kind)
    job_id = new_job_id()
    with _state_lock:
        if cls == "bench":
            lease = _active_lease()
            if lease is not None:
                raise BenchBusy({"lease_id": lease["id"], "owner": lease["owner"]})
        cancel = threading.Event()
        _cancel_events[job_id] = cancel
        _live_kinds[job_id] = kind
        superseded = (
            [jid for jid, k in _live_kinds.items() if job_class(k) == "lifecycle"] if cls == "stop" else []
        )
    for jid in superseded:
        ev = _cancel_events.get(jid)
        if ev is not None:
            ev.set()
    log = JobLog(job_id)
    now = utc_now()

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
    plock = threading.Lock()
    st: dict[str, Any] = {"last": 0.0, "msg": "", "pending": None, "timer": None, "closed": False}

    def _write_progress(p: float, msg: str) -> None:
        # caller holds plock
        st["pending"] = None
        st["last"] = time.monotonic()
        st["msg"] = msg
        write("running", p, msg)

    def _flush() -> None:
        with plock:
            st["timer"] = None
            if not st["closed"] and st["pending"] is not None:
                _write_progress(*st["pending"])

    def progress(p: float, msg: str = "") -> None:
        if msg:
            log.write(msg)
        with plock:
            if st["closed"]:
                return
            msg = msg or st["msg"]
            wait = PROGRESS_WRITE_INTERVAL_S - (time.monotonic() - st["last"])
            if wait <= 0:
                _write_progress(p, msg)
                return
            st["pending"] = (p, msg)
            if st["timer"] is None:
                t = threading.Timer(wait, _flush)
                t.daemon = True
                st["timer"] = t
                t.start()

    def close_progress() -> None:
        with plock:
            st["closed"] = True
            if st["timer"] is not None:
                st["timer"].cancel()
                st["timer"] = None

    def runner() -> None:
        try:
            if cancel.is_set():
                raise JobCancelled("cancelled while queued")
            with plock:
                st["last"] = time.monotonic()
                write("running", 0, "started")
            log.write(f"=== job {job_id} kind={kind} ===")
            result = fn(log=log, progress=progress, cancel=cancel, **kwargs)
            close_progress()
            write(
                "completed",
                1.0,
                "done",
                result if isinstance(result, dict) else {"result": result},
            )
            log.write("=== completed ===")
        except JobCancelled as e:
            close_progress()
            log.write(f"=== cancelled: {e} ===")
            write("cancelled", 0, str(e) or "cancelled", {"cancelled": True})
        except Exception as e:
            close_progress()
            log.write(f"ERROR: {e}")
            log.write(traceback.format_exc())
            write("failed", 0, str(e), {"error": str(e)})
        finally:
            with _state_lock:
                _cancel_events.pop(job_id, None)
                _live_kinds.pop(job_id, None)

    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executors[cls], runner)
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
