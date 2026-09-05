"""Job runner: queued→running, cancel flag, throttled progress, seek-based tail, WAL."""
from __future__ import annotations

import asyncio
import concurrent.futures
import sqlite3
import threading

from app import db
from app.services import jobs, perf


async def _wait_terminal(job_id: str, ticks: int = 500) -> dict:
    for _ in range(ticks):
        row = db.get_job(job_id)
        if row and row["status"] in jobs.TERMINAL_STATES:
            return row
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish: {db.get_job(job_id)}")


class HeldExecutor:
    """Accepts submissions but runs nothing until the test calls them."""

    def __init__(self) -> None:
        self.fns: list = []

    def submit(self, fn, *args, **kwargs):
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self.fns.append(fn)
        return fut


def test_init_db_enables_wal(isolated_data):
    with sqlite3.connect(str(db.DB_PATH)) as c:
        assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_job_is_queued_until_the_executor_picks_it_up(isolated_data, monkeypatch):
    held = HeldExecutor()
    monkeypatch.setattr(jobs, "_executor", held)
    seen: dict = {}

    def work(log, progress, cancel, **kw):
        seen["status_while_running"] = db.get_job(job_id)["status"]
        seen["cancel_is_event"] = isinstance(cancel, threading.Event)
        return {"ok": True}

    async def go():
        jid = await jobs.start_job("test", work)
        return jid, db.get_job(jid)

    job_id, queued = asyncio.run(go())
    assert queued["status"] == "queued" and queued["message"] == "queued"
    held.fns[0]()
    assert seen == {"status_while_running": "running", "cancel_is_event": True}
    row = db.get_job(job_id)
    assert row["status"] == "completed" and row["progress"] == 1.0 and row["result"] == {"ok": True}
    assert jobs._cancel_events.get(job_id) is None


def test_cancel_while_queued_never_runs_the_job(isolated_data, monkeypatch):
    held = HeldExecutor()
    monkeypatch.setattr(jobs, "_executor", held)
    ran = {"n": 0}

    def work(log, progress, cancel, **kw):
        ran["n"] += 1

    job_id = asyncio.run(jobs.start_job("test", work))
    row = jobs.request_cancel(job_id)
    assert row["status"] == "queued"
    held.fns[0]()
    row = db.get_job(job_id)
    assert row["status"] == "cancelled" and "queued" in row["message"]
    assert row["result"] == {"cancelled": True}
    assert ran["n"] == 0


def test_cancel_stops_a_running_bench_wave_and_persists_nothing(isolated_data, monkeypatch):
    started = threading.Event()

    def fake_stream(base, model, user_content, max_tokens, label, cancel=None):
        started.set()
        cancel.wait(5)
        return perf.ReqResult(False, 0.1, None, None, None, label, error="cancelled")

    monkeypatch.setattr(perf, "stream_one", fake_stream)

    def work(log, progress, cancel, **kw):
        return perf.run_workflow_bench(
            base_url="http://vllm.invalid:8000",
            model="m",
            concurrencies=[2, 4],
            workload="prose",
            log=log,
            progress=progress,
            cancel=cancel,
        )

    async def go():
        jid = await jobs.start_job("decode_prose", work)
        assert started.wait(5)
        row = jobs.request_cancel(jid)
        assert row["status"] == "running"  # flag set; the runner flips the row
        return jid, await _wait_terminal(jid)

    job_id, row = asyncio.run(go())
    assert row["status"] == "cancelled"
    assert row["result"] == {"cancelled": True}
    assert row["message"] == "cancelled before prose · 4"  # flag seen at the wave boundary
    assert list((isolated_data / "runs").iterdir()) == []
    assert "=== cancelled" in (isolated_data / "logs" / f"{job_id}.log").read_text()
    assert db.list_runs() == []


def test_failed_job_carries_the_error_message(isolated_data):
    def work(log, progress, cancel, **kw):
        raise RuntimeError(perf.EXTERNAL_RUNNERS_MISSING)

    async def go():
        jid = await jobs.start_job("perf_prefill", work)
        return await _wait_terminal(jid)

    row = asyncio.run(go())
    assert row["status"] == "failed"
    assert row["message"] == perf.EXTERNAL_RUNNERS_MISSING


def test_progress_writes_are_throttled_but_messages_reach_the_log(isolated_data, monkeypatch):
    statuses: list[str] = []
    real = db.upsert_job

    def spy(**kw):
        statuses.append(kw["status"])
        real(**kw)

    monkeypatch.setattr(jobs.db, "upsert_job", spy)

    def work(log, progress, cancel, **kw):
        for i in range(50):
            progress(i / 50, f"step {i}")
        return {}

    async def go():
        jid = await jobs.start_job("test", work)
        return jid, await _wait_terminal(jid)

    job_id, _ = asyncio.run(go())
    assert statuses[0] == "queued" and statuses[-1] == "completed"
    # "started" + the first progress; the other 49 calls land inside the 250 ms window
    assert statuses.count("running") <= 3, statuses
    log = (isolated_data / "logs" / f"{job_id}.log").read_text()
    assert all(f"step {i}" in log for i in range(50))


def test_request_cancel_marks_an_orphaned_job_and_leaves_terminal_ones(isolated_data):
    db.upsert_job(
        job_id="orphan", kind="serve", status="running", created_at="t0", updated_at="t0",
        progress=0.3, message="loading", log_path=None,
    )
    row = jobs.request_cancel("orphan")
    assert row["status"] == "cancelled" and "no runner" in row["message"]
    db.upsert_job(
        job_id="done", kind="serve", status="completed", created_at="t0", updated_at="t1",
        progress=1.0, message="done", log_path=None,
    )
    assert jobs.request_cancel("done")["status"] == "completed"
    assert jobs.request_cancel("nope") is None


def test_read_log_tail_is_seek_based(tmp_path):
    p = tmp_path / "j.log"
    p.write_text("a\nb\n")
    chunk, off = jobs.read_log_tail(p, 0)
    assert (chunk, off) == ("a\nb\n", 4)
    assert jobs.read_log_tail(p, off) == ("", 4)
    with p.open("a", encoding="utf-8") as f:
        f.write("cé\n")
    chunk, off2 = jobs.read_log_tail(p, off)
    assert chunk == "cé\n" and off2 == 4 + len("cé\n".encode())
    p.write_text("x\n")  # rewritten shorter than the offset → start over
    assert jobs.read_log_tail(p, off2) == ("x\n", 2)
    assert jobs.read_log_tail(tmp_path / "missing.log", 7) == ("", 0)


# ─── routes ───────────────────────────────────────────────────────────────────


def test_cancel_route_404_and_orphan(isolated_data, monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main_mod

    monkeypatch.setattr(main_mod, "_LAIL_TOKEN", "")
    client = TestClient(main_mod.app)
    assert client.post("/api/jobs/nope/cancel").status_code == 404
    db.upsert_job(
        job_id="orphan", kind="serve", status="running", created_at="t0", updated_at="t0",
        progress=0.3, message="loading", log_path=None,
    )
    r = client.post("/api/jobs/orphan/cancel")
    assert r.status_code == 200
    assert r.json() == {"job_id": "orphan", "status": "cancelled"}


def test_bench_job_cancelled_through_the_routes(isolated_data, monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main_mod
    from app.services import agentic, cluster, metadata, status_sampler

    monkeypatch.setattr(main_mod, "_LAIL_TOKEN", "")
    monkeypatch.delenv("LAIL_HOST", raising=False)
    monkeypatch.delenv("LAB_HOST", raising=False)

    async def fake_probe(base_url, timeout=5.0):
        return {"base_url": base_url, "healthy": True, "models": [], "version": None, "metrics": {}, "error": None}

    monkeypatch.setattr(metadata, "probe_endpoint", fake_probe)
    monkeypatch.setattr(metadata, "collect_hardware", lambda: {})
    monkeypatch.setattr(metadata, "list_vllm_containers", lambda: [])
    monkeypatch.setattr(cluster, "collect_cluster", lambda: {"nodes": [], "summary": {}})
    monkeypatch.setattr(agentic, "tool_eval_available", lambda: {"available": False})
    monkeypatch.setattr(status_sampler, "SAMPLER", status_sampler.StatusSampler())

    started = threading.Event()

    def fake_stream(base, model, user_content, max_tokens, label, cancel=None):
        started.set()
        cancel.wait(5)
        return perf.ReqResult(False, 0.1, None, None, None, label, error="cancelled")

    monkeypatch.setattr(perf, "stream_one", fake_stream)

    with TestClient(main_mod.app) as client:
        r = client.post(
            "/api/bench/perf",
            json={"runner": "decode", "workload": "prose", "concurrencies": [1, 8], "model": "m"},
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
        assert started.wait(5)
        assert client.get(f"/api/jobs/{job_id}").json()["status"] == "running"
        r = client.post(f"/api/jobs/{job_id}/cancel")
        assert r.status_code == 200 and r.json()["job_id"] == job_id
        for _ in range(500):
            row = client.get(f"/api/jobs/{job_id}").json()
            if row["status"] in jobs.TERMINAL_STATES:
                break
            threading.Event().wait(0.01)
    assert row["status"] == "cancelled"
    assert row["kind"] == "decode_prose"
    assert db.list_runs() == []
