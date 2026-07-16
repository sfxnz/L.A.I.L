"""Background job runner with log files for SSE."""
from __future__ import annotations

import asyncio
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from ..config import DATA_DIR
from .. import db
from .metadata import utc_now

_executor = ThreadPoolExecutor(max_workers=2)


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


async def start_job(kind: str, fn: Callable[..., Any], **kwargs: Any) -> str:
    """Run blocking fn(log, progress, **kwargs) in thread pool."""
    job_id = new_job_id()
    log = JobLog(job_id)
    now = utc_now()
    db.upsert_job(
        job_id=job_id,
        kind=kind,
        status="running",
        created_at=now,
        updated_at=now,
        progress=0,
        message="started",
        log_path=str(log.path),
    )

    def progress(p: float, msg: str = "") -> None:
        db.upsert_job(
            job_id=job_id,
            kind=kind,
            status="running",
            created_at=now,
            updated_at=utc_now(),
            progress=p,
            message=msg,
            log_path=str(log.path),
        )
        if msg:
            log.write(msg)

    def runner() -> None:
        try:
            log.write(f"=== job {job_id} kind={kind} ===")
            result = fn(log=log, progress=progress, **kwargs)
            db.upsert_job(
                job_id=job_id,
                kind=kind,
                status="completed",
                created_at=now,
                updated_at=utc_now(),
                progress=1.0,
                message="done",
                result=result if isinstance(result, dict) else {"result": result},
                log_path=str(log.path),
            )
            log.write("=== completed ===")
        except Exception as e:
            log.write(f"ERROR: {e}")
            log.write(traceback.format_exc())
            db.upsert_job(
                job_id=job_id,
                kind=kind,
                status="failed",
                created_at=now,
                updated_at=utc_now(),
                progress=0,
                message=str(e),
                result={"error": str(e)},
                log_path=str(log.path),
            )

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, runner)
    return job_id


def read_log_tail(path: str | Path, offset: int = 0) -> tuple[str, int]:
    p = Path(path)
    if not p.exists():
        return "", 0
    data = p.read_text(errors="replace")
    if offset > len(data):
        offset = 0
    return data[offset:], len(data)
