"""Job runner: executors by class, bench lease, cancel flag, throttled progress, seek-based tail, WAL."""
from __future__ import annotations

import asyncio
import concurrent.futures
import sqlite3
import threading
import time

import pytest

from app import db
from app.services import agentic, jobs


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
    monkeypatch.setitem(jobs._executors, "bench", held)
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
    monkeypatch.setitem(jobs._executors, "bench", held)
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


def _fake_tool_eval(monkeypatch, tmp_path, seconds: int = 30):
    """A tool-eval-bench stand-in that stays silent for `seconds`."""
    script = tmp_path / "tool-eval-bench"
    script.write_text(f"#!/bin/sh\nsleep {seconds}\n")
    script.chmod(0o755)
    monkeypatch.setattr(agentic, "tool_eval_available", lambda: {"available": True, "path": str(script)})


def test_cancel_terminates_a_silent_tool_eval_subprocess_and_persists_nothing(isolated_data, monkeypatch):
    _fake_tool_eval(monkeypatch, isolated_data)

    def work(log, progress, cancel, **kw):
        return agentic.run_tool_eval_bench(base_url="http://vllm.invalid:8000", log=log, progress=progress, cancel=cancel)

    async def go():
        jid = await jobs.start_job("agentic_tool_eval", work)
        for _ in range(200):
            if db.get_job(jid)["status"] == "running":
                break
            await asyncio.sleep(0.01)
        t0 = time.monotonic()
        assert jobs.request_cancel(jid)["status"] == "running"  # flag set; the runner flips the row
        row = await _wait_terminal(jid)
        return jid, row, time.monotonic() - t0

    job_id, row, took = asyncio.run(go())
    assert row["status"] == "cancelled" and "terminated" in row["message"]
    assert took < 3  # the subprocess printed nothing: the watcher, not stdout, noticed
    assert db.list_runs() == []
    assert "=== cancelled" in (isolated_data / "logs" / f"{job_id}.log").read_text()


def test_cancel_stops_golden_tools_between_cases(isolated_data, monkeypatch):
    class _Models:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"data": [{"id": "m"}]}'

    monkeypatch.setattr(agentic.urllib.request, "urlopen", lambda *a, **k: _Models())
    cancel = threading.Event()
    calls = {"n": 0}

    def fake_chat(base, model, prompt):
        calls["n"] += 1
        if calls["n"] == 3:  # probe + 2 cases, then the operator cancels
            cancel.set()
        return {"choices": [{"message": {"tool_calls": []}}]}

    monkeypatch.setattr(agentic, "_chat_tools", fake_chat)
    with pytest.raises(jobs.JobCancelled, match="before case 3/12"):
        agentic.run_golden_tools(base_url="http://vllm.invalid:8000", cancel=cancel)
    assert db.list_runs() == []


def test_failed_job_carries_the_error_message(isolated_data):
    def work(log, progress, cancel, **kw):
        raise RuntimeError("tool-eval-bench not installed")

    async def go():
        jid = await jobs.start_job("agentic_tool_eval", work)
        return await _wait_terminal(jid)

    row = asyncio.run(go())
    assert row["status"] == "failed"
    assert row["message"] == "tool-eval-bench not installed"


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
    # "started" + at most one write per 250 ms window; the 50 calls take far less than that
    assert statuses.count("running") <= 3, statuses
    log = (isolated_data / "logs" / f"{job_id}.log").read_text()
    assert all(f"step {i}" in log for i in range(50))


def test_throttled_last_message_is_flushed_while_the_job_blocks(isolated_data):
    """serve_model posts several steps within ms, then blocks for minutes in docker run."""
    seen: dict = {}

    def work(log, progress, cancel, **kw):
        for msg in ("image=x", "flags=y", "docker run…", "waiting for readiness"):
            progress(0.25, msg)
        time.sleep(jobs.PROGRESS_WRITE_INTERVAL_S * 3)
        seen["row"] = db.get_job(job_id)
        return {}

    async def go():
        nonlocal job_id
        job_id = await jobs.start_job("agentic_golden", work)
        return await _wait_terminal(job_id)

    job_id = ""
    asyncio.run(go())
    assert seen["row"]["status"] == "running"
    assert seen["row"]["message"] == "waiting for readiness"


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
    from app.services import cluster, metadata, status_sampler

    monkeypatch.setattr(main_mod, "_LAIL_TOKEN", "")
    monkeypatch.delenv("LAIL_HOST", raising=False)
    monkeypatch.delenv("LAB_HOST", raising=False)

    async def fake_probe(base_url, timeout=5.0):
        return {"base_url": base_url, "healthy": True, "models": [], "version": None, "metrics": {}, "error": None}

    monkeypatch.setattr(metadata, "probe_endpoint", fake_probe)
    monkeypatch.setattr(metadata, "collect_hardware", lambda: {})
    monkeypatch.setattr(metadata, "list_vllm_containers", lambda: [])
    monkeypatch.setattr(cluster, "collect_cluster", lambda: {"nodes": [], "summary": {}})
    monkeypatch.setattr(status_sampler, "SAMPLER", status_sampler.StatusSampler())
    _fake_tool_eval(monkeypatch, isolated_data)

    with TestClient(main_mod.app) as client:
        r = client.post("/api/bench/agentic", json={"suite": "tool_eval", "model": "m"})
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
        for _ in range(500):
            if client.get(f"/api/jobs/{job_id}").json()["status"] == "running":
                break
            threading.Event().wait(0.01)
        r = client.post(f"/api/jobs/{job_id}/cancel")
        assert r.status_code == 200 and r.json()["job_id"] == job_id
        for _ in range(500):
            row = client.get(f"/api/jobs/{job_id}").json()
            if row["status"] in jobs.TERMINAL_STATES:
                break
            threading.Event().wait(0.01)
    assert row["status"] == "cancelled"
    assert row["kind"] == "agentic_tool_eval"
    assert db.list_runs() == []


# ─── executors, lease, orphans ────────────────────────────────────────────────


def test_stop_never_queues_behind_a_load_or_a_bench_and_cancels_a_queued_load(isolated_data, monkeypatch):
    lifecycle, bench, stop = HeldExecutor(), HeldExecutor(), HeldExecutor()
    monkeypatch.setitem(jobs._executors, "lifecycle", lifecycle)
    monkeypatch.setitem(jobs._executors, "bench", bench)
    monkeypatch.setitem(jobs._executors, "stop", stop)
    noop = lambda log, progress, cancel, **kw: {}  # noqa: E731

    async def go():
        serve = await jobs.start_job("serve", noop)
        evals = await jobs.start_job("agentic_tool_eval", noop)
        halt = await jobs.start_job("stop", noop)
        return serve, evals, halt

    serve, evals, halt = asyncio.run(go())
    assert (len(lifecycle.fns), len(bench.fns), len(stop.fns)) == (1, 1, 1)
    assert jobs._cancel_events[serve].is_set()  # the queued load will not start after a stop
    assert not jobs._cancel_events[evals].is_set()
    stop.fns[0]()
    assert db.get_job(halt)["status"] == "completed"
    lifecycle.fns[0]()
    assert db.get_job(serve)["status"] == "cancelled"
    bench.fns[0]()


def test_bench_lease_excludes_bench_jobs_both_ways(isolated_data, monkeypatch):
    held = HeldExecutor()
    monkeypatch.setitem(jobs._executors, "bench", held)
    noop = lambda log, progress, cancel, **kw: {}  # noqa: E731
    monkeypatch.setattr(jobs, "_lease", None)

    assert jobs.acquire_lease("run-a", "controller", 60) is None
    assert jobs.acquire_lease("run-a", "controller", 60) is None  # renewal
    assert jobs.acquire_lease("run-b", "controller", 60) == {"lease_id": "run-a", "owner": "controller"}
    with pytest.raises(jobs.BenchBusy):
        asyncio.run(jobs.start_job("agentic_golden", noop))
    # lifecycle jobs are not benches
    monkeypatch.setitem(jobs._executors, "lifecycle", HeldExecutor())
    asyncio.run(jobs.start_job("serve", noop))
    jobs.release_lease("run-a")

    jid = asyncio.run(jobs.start_job("agentic_golden", noop))
    assert jobs.acquire_lease("run-c", "controller", 60) == {"job_id": jid, "kind": "agentic_golden"}
    held.fns[0]()
    assert jobs.acquire_lease("run-c", "controller", 60) is None
    jobs.release_lease("run-c")


def test_expired_lease_does_not_block(isolated_data, monkeypatch):
    monkeypatch.setattr(jobs, "_lease", {"id": "dead", "owner": "controller", "expires": time.monotonic() - 1})
    assert jobs.acquire_lease("new", "controller", 60) is None
    jobs.release_lease("new")


def test_lease_routes(isolated_data, monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main_mod

    monkeypatch.setattr(main_mod, "_LAIL_TOKEN", "")
    monkeypatch.setattr(jobs, "_lease", None)
    client = TestClient(main_mod.app)
    assert client.post("/api/bench/lease", json={"lease_id": "r1"}).status_code == 200
    r = client.post("/api/bench/lease", json={"lease_id": "r2"})
    assert r.status_code == 409 and r.json()["detail"]["lease_id"] == "r1"
    r = client.post("/api/bench/agentic", json={"suite": "golden"})
    assert r.status_code == 409 and r.json()["detail"]["error"] == "bench_busy"
    assert client.delete("/api/bench/lease/r1").status_code == 200
    assert client.post("/api/bench/lease", json={"lease_id": "r2"}).status_code == 200
    client.delete("/api/bench/lease/r2")


def test_startup_fails_orphaned_jobs(isolated_data):
    for jid, status in (("a", "running"), ("b", "queued"), ("c", "completed")):
        db.upsert_job(job_id=jid, kind="serve", status=status, created_at="t0", updated_at="t0", message="m")
    assert db.fail_orphaned_jobs() == 2
    assert db.get_job("a")["status"] == "failed" and "restart" in db.get_job("a")["message"]
    assert db.get_job("b")["status"] == "failed"
    assert db.get_job("c")["status"] == "completed"


def test_job_sse_ends_for_a_row_with_no_runner(isolated_data, monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main_mod

    monkeypatch.setattr(main_mod, "_LAIL_TOKEN", "")
    db.upsert_job(job_id="ghost", kind="serve", status="running", created_at="t0", updated_at="t0", message="loading")
    client = TestClient(main_mod.app)
    with client.stream("GET", "/api/jobs/ghost/logs") as r:
        body = "".join(r.iter_text())
    assert '"status": "failed"' in body and "orphaned" in body


def test_job_list_leaves_the_result_envelope_to_get_job(isolated_data, monkeypatch):
    held = HeldExecutor()
    monkeypatch.setitem(jobs._executors, "bench", held)

    def work(log, progress, cancel, **kw):
        return {"big": "x" * 1000}

    job_id = asyncio.run(jobs.start_job("test", work))
    held.fns[0]()
    listed = next(j for j in db.list_jobs() if j["job_id"] == job_id)
    assert "result" not in listed and "result_json" not in listed
    assert db.get_job(job_id)["result"] == {"big": "x" * 1000}
