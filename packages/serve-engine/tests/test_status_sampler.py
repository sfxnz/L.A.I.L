"""Background status sampler: cached /api/status, concurrent probe, one tok/s writer."""
from __future__ import annotations

import asyncio
from collections import Counter

import httpx
import pytest

from app.services import agentic, cluster, metadata, perf, status_sampler
from app.services.status_sampler import StatusSampler

STATUS_KEYS = {
    "healthy", "base_url", "model_id", "models", "version", "metrics", "engine", "hardware",
    "containers", "headroom", "error", "presets", "serve_examples", "tool_eval", "cluster",
}


def _probe(base_url: str = "http://127.0.0.1:8000") -> dict:
    return {
        "base_url": base_url,
        "healthy": True,
        "models": [{"id": "org/m", "max_model_len": 8192}],
        "version": {"version": "0.28.1"},
        "metrics": {
            "gen_tok_per_s": 42.0, "prompt_tok_per_s": 300.0, "gpu_kv_cache_usage": 0.125,
            "requests_running": 2.0, "block_size": 16.0, "num_gpu_blocks": 100.0,
        },
        "error": None,
    }


def _fake_collectors(monkeypatch, calls: Counter) -> None:
    async def fake_probe(base_url, timeout=5.0):
        calls["probe"] += 1
        return _probe(base_url)

    def fake_hw():
        calls["hw"] += 1
        return {"gpu_sku": "NVIDIA GB10", "available_gib": 30.0, "ram_gib": 121.7}

    def fake_containers():
        calls["containers"] += 1
        return [{"name": "spark-vllm", "status": "Up 2 hours", "image": "vllm/vllm-openai:v0.27.1", "id": "abc"}]

    def fake_cluster():
        calls["cluster"] += 1
        return {
            "nodes": [{"id": "a", "state": "serving"}, {"id": "b", "state": "idle"}],
            "summary": {"healthy": True},
        }

    def fake_tool_eval():
        calls["tool_eval"] += 1
        return {"available": False}

    def fake_inspect(name):
        calls["inspect"] += 1
        return {"cmd": ["--port", "8000"], "started_at": "2026-09-05T15:53:25.963708557Z"}

    monkeypatch.setattr(metadata, "probe_endpoint", fake_probe)
    monkeypatch.setattr(metadata, "collect_hardware", fake_hw)
    monkeypatch.setattr(metadata, "list_vllm_containers", fake_containers)
    monkeypatch.setattr(metadata, "docker_inspect_flags", fake_inspect)
    monkeypatch.setattr(cluster, "collect_cluster", fake_cluster)
    monkeypatch.setattr(agentic, "tool_eval_available", fake_tool_eval)


def _no_collectors(monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("collector called on the request path")

    async def aboom(*a, **k):
        raise AssertionError("probe_endpoint called on the request path")

    monkeypatch.setattr(metadata, "probe_endpoint", aboom)
    monkeypatch.setattr(metadata, "collect_hardware", boom)
    monkeypatch.setattr(metadata, "list_vllm_containers", boom)
    monkeypatch.setattr(metadata, "docker_inspect_flags", boom)
    monkeypatch.setattr(cluster, "collect_cluster", boom)
    monkeypatch.setattr(agentic, "tool_eval_available", boom)


def _sampled(monkeypatch) -> StatusSampler:
    calls: Counter = Counter()
    _fake_collectors(monkeypatch, calls)
    s = StatusSampler("http://127.0.0.1:8000")

    async def go():
        await s.sample_cluster()
        await s.sample()

    asyncio.run(go())
    return s


# ─── probe_endpoint ───────────────────────────────────────────────────────────


def _mock_httpx(monkeypatch, handler) -> None:
    real = httpx.AsyncClient
    monkeypatch.setattr(
        metadata.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )


def test_probe_endpoint_issues_the_four_gets_concurrently(monkeypatch):
    inflight = {"n": 0, "peak": 0}
    paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        inflight["n"] += 1
        inflight["peak"] = max(inflight["peak"], inflight["n"])
        await asyncio.sleep(0.05)
        inflight["n"] -= 1
        p = request.url.path
        if p == "/health":
            return httpx.Response(200)
        if p == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "org/m", "max_model_len": 4096}]})
        if p == "/version":
            return httpx.Response(200, json={"version": "0.28.1"})
        if p == "/metrics":
            return httpx.Response(200, text="vllm:num_requests_running 1\nvllm:generation_tokens_total 100\n")
        return httpx.Response(404)

    _mock_httpx(monkeypatch, handler)
    metadata.reset_live_rate_state()
    out = asyncio.run(metadata.probe_endpoint("http://vllm.invalid:8000/"))
    assert sorted(paths) == ["/health", "/metrics", "/v1/models", "/version"]
    assert inflight["peak"] == 4, "GETs were sequential"
    assert out["base_url"] == "http://vllm.invalid:8000"
    assert out["healthy"] is True
    assert out["models"][0]["id"] == "org/m"
    assert out["version"] == {"version": "0.28.1"}
    assert out["metrics"]["requests_running"] == 1.0
    assert out["error"] is None


def test_probe_endpoint_reports_health_and_models_errors_like_before(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    _mock_httpx(monkeypatch, handler)
    out = asyncio.run(metadata.probe_endpoint("http://vllm.invalid:8000"))
    assert out["healthy"] is False
    assert out["models"] == [] and out["version"] is None and out["metrics"] == {}
    assert out["error"] == "health: refused models: refused"


def test_probe_endpoint_models_alone_marks_healthy(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        raise httpx.ConnectError("nope")

    _mock_httpx(monkeypatch, handler)
    out = asyncio.run(metadata.probe_endpoint("http://vllm.invalid:8000"))
    assert out["healthy"] is True
    assert out["error"] == "health: nope"


# ─── sampler ──────────────────────────────────────────────────────────────────


def test_sampler_publishes_snapshot_and_status_reads_the_cache(monkeypatch):
    calls: Counter = Counter()
    _fake_collectors(monkeypatch, calls)
    s = StatusSampler("http://127.0.0.1:8000")

    async def go():
        await s.sample_cluster()
        snap = await s.sample()
        return snap, await s.status(), await s.status()

    snap, first, second = asyncio.run(go())
    assert set(snap) == STATUS_KEYS | {"sampled_at"}
    assert set(first) == STATUS_KEYS | {"sampled_at", "stale_s"}
    assert snap["healthy"] is True
    assert snap["model_id"] == "org/m"
    assert snap["headroom"] == "tight"
    assert snap["containers"][0]["name"] == "spark-vllm"
    assert snap["tool_eval"] == {"available": False}
    assert snap["sampled_at"].endswith("+00:00")
    # live tok/s from the probe lands on serving nodes only
    assert snap["cluster"]["nodes"][0]["gen_tok_per_s"] == 42.0
    assert snap["cluster"]["nodes"][1]["gen_tok_per_s"] is None
    assert first["stale_s"] >= 0 and second["stale_s"] >= first["stale_s"]
    # engine block: metrics + /v1/models + /version + cached docker inspect
    assert snap["engine"]["kv_usage_pct"] == 12.5
    assert snap["engine"]["requests_running"] == 2
    assert snap["engine"]["kv_capacity_tokens"] == 1600
    assert snap["engine"]["max_model_len"] == 8192
    assert snap["engine"]["version"] == "0.28.1"
    assert snap["engine"]["flags"] == ["--port", "8000"]
    assert snap["engine"]["flags_fingerprint"] and snap["engine"]["uptime_s"] > 0
    assert snap["engine"]["requests_waiting"] is None  # not in this probe → null, not 0
    # reads never re-run collectors
    assert dict(calls) == {"probe": 1, "hw": 1, "containers": 1, "cluster": 1, "tool_eval": 1, "inspect": 1}
    assert s.probe() == _probe()


def test_docker_inspect_runs_once_per_slow_interval_and_on_container_change(monkeypatch):
    calls: Counter = Counter()
    _fake_collectors(monkeypatch, calls)
    s = StatusSampler("http://127.0.0.1:8000", slow_s=1000.0)

    async def go():
        await s.sample()
        await s.sample()
        await s.sample()
        assert calls["inspect"] == 1
        monkeypatch.setattr(
            metadata, "list_vllm_containers",
            lambda: [{"name": "other", "status": "Up 1 second", "image": "vllm/vllm-openai:latest", "id": "z"}],
        )
        await s.sample()
        assert calls["inspect"] == 2
        monkeypatch.setattr(metadata, "list_vllm_containers", lambda: [])
        return await s.sample()

    snap = asyncio.run(go())
    assert snap["engine"]["flags"] == [] and snap["engine"]["uptime_s"] is None
    assert calls["inspect"] == 2


def test_status_before_the_first_sample_is_a_sane_placeholder():
    s = StatusSampler("http://127.0.0.1:8000")
    body = asyncio.run(s.status(timeout=0.01))
    assert set(body) == STATUS_KEYS | {"sampled_at", "stale_s"}
    assert body["healthy"] is None
    assert body["sampled_at"] is None and body["stale_s"] is None
    assert body["engine"]["kv_usage_pct"] is None and body["engine"]["flags"] == []
    assert "warming up" in body["error"]
    assert body["cluster"]["nodes"] == [] and body["cluster"]["pending"] is True
    assert body["models"] == [] and body["containers"] == []


def test_cluster_tick_republishes_without_waiting_for_the_fast_tick(monkeypatch):
    calls: Counter = Counter()
    _fake_collectors(monkeypatch, calls)
    s = StatusSampler("http://127.0.0.1:8000")

    async def go():
        await s.sample()
        before = await s.status()
        await s.sample_cluster()
        return before, await s.status()

    before, after = asyncio.run(go())
    assert before["cluster"]["pending"] is True and before["tool_eval"] is None
    assert after["cluster"]["nodes"][0]["gen_tok_per_s"] == 42.0
    assert after["tool_eval"] == {"available": False}
    assert after["sampled_at"] == before["sampled_at"]


def test_sampler_loop_survives_exceptions_and_first_status_awaits_a_sample(monkeypatch):
    calls: Counter = Counter()
    _fake_collectors(monkeypatch, calls)
    flaky = {"n": 0}

    async def flaky_probe(base_url, timeout=5.0):
        flaky["n"] += 1
        if flaky["n"] == 1:
            raise RuntimeError("boom")
        return _probe(base_url)

    monkeypatch.setattr(metadata, "probe_endpoint", flaky_probe)
    s = StatusSampler("http://127.0.0.1:8000", fast_s=0.01, slow_s=0.01)

    async def go():
        await s.start()
        try:
            return await s.status(timeout=2.0)
        finally:
            await s.stop()

    body = asyncio.run(go())
    assert flaky["n"] >= 2
    assert body["healthy"] is True and body["sampled_at"] is not None
    assert s._tasks == []


# ─── /api/status route ────────────────────────────────────────────────────────


def test_status_route_returns_the_cached_snapshot_without_collectors(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main_mod

    monkeypatch.setattr(main_mod, "_LAIL_TOKEN", "")
    s = _sampled(monkeypatch)
    monkeypatch.setattr(status_sampler, "SAMPLER", s)
    _no_collectors(monkeypatch)

    client = TestClient(main_mod.app)  # no lifespan: only the injected sampler serves reads
    r = client.get("/api/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == STATUS_KEYS | {"sampled_at", "stale_s"}
    assert body["healthy"] is True and body["model_id"] == "org/m"
    assert body["sampled_at"] and isinstance(body["stale_s"], (int, float))
    assert body["cluster"]["nodes"][0]["gen_tok_per_s"] == 42.0


def test_lifespan_starts_the_sampler_and_replaces_on_event(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main_mod

    monkeypatch.setattr(main_mod, "_LAIL_TOKEN", "")
    monkeypatch.delenv("LAIL_HOST", raising=False)
    monkeypatch.delenv("LAB_HOST", raising=False)
    calls: Counter = Counter()
    _fake_collectors(monkeypatch, calls)
    monkeypatch.setattr(status_sampler, "SAMPLER", StatusSampler("http://127.0.0.1:8000"))
    assert main_mod.app.router.on_startup == []

    with TestClient(main_mod.app) as client:
        body = client.get("/api/status").json()
    assert body["sampled_at"] is not None
    assert body["healthy"] is True
    assert calls["hw"] >= 1 and calls["cluster"] >= 1
    assert status_sampler.SAMPLER._tasks == []


# ─── bench threads use the cached probe ───────────────────────────────────────


def _fake_stream(base, model, user_content, max_tokens, label, cancel=None):
    return perf.ReqResult(True, 1.0, 0.2, 60, 100, label, last_s=0.9)


def test_workflow_bench_uses_the_cached_probe_and_never_probes_itself(isolated_data, monkeypatch):
    _no_collectors(monkeypatch)
    monkeypatch.setattr(metadata, "collect_hardware", lambda: {"gpu_sku": "NVIDIA GB10"})
    monkeypatch.setattr(metadata, "list_vllm_containers", lambda: [])
    monkeypatch.setattr(perf, "stream_one", _fake_stream)

    env = perf.run_workflow_bench(
        base_url="http://127.0.0.1:8000",
        model="org/m",
        concurrencies=[1],
        workload="prose",
        probe=_probe("http://127.0.0.1:8000"),
    )
    assert env["engine"]["version"] == "0.28.1"
    assert env["engine"]["metrics_snapshot"]["gen_tok_per_s"] == 42.0
    assert env["endpoint"]["models"] == [{"id": "org/m", "max_model_len": 8192}]
    assert env["model"]["max_model_len"] == 8192


def test_workflow_bench_ignores_a_cached_probe_for_another_endpoint(isolated_data, monkeypatch):
    _no_collectors(monkeypatch)
    monkeypatch.setattr(metadata, "collect_hardware", lambda: {"gpu_sku": "NVIDIA GB10"})
    monkeypatch.setattr(metadata, "list_vllm_containers", lambda: [])
    monkeypatch.setattr(perf, "stream_one", _fake_stream)

    env = perf.run_workflow_bench(
        base_url="http://127.0.0.1:8888",
        model="org/other",
        concurrencies=[1],
        workload="prose",
        probe=_probe("http://127.0.0.1:8000"),
    )
    assert env["engine"]["version"] is None
    assert env["endpoint"]["models"] == []
    assert env["model"]["id"] == "org/other"


@pytest.mark.parametrize("gib,expected", [(None, "ok"), (10, "critical"), (40, "tight"), (80, "ok")])
def test_headroom_thresholds_unchanged(gib, expected):
    assert status_sampler.headroom_for(gib) == expected
