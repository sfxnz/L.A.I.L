"""Background status sampler: cached /api/status, concurrent probe, one tok/s writer."""
from __future__ import annotations

import asyncio
import time
from collections import Counter

import httpx
import pytest

from app.services import agentic, cluster, metadata, status_sampler
from app.services.status_sampler import StatusSampler

STATUS_KEYS = {
    "healthy", "base_url", "model_id", "models", "version", "metrics", "engine", "hardware",
    "containers", "headroom", "error", "cluster",
    "sampled_at", "sampled_at_ms",
}


def _probe(base_url: str = "http://127.0.0.1:8000") -> dict:
    return {
        "base_url": base_url,
        "healthy": True,
        "models": [{"id": "org/m", "max_model_len": 8192}],
        "version": {"version": "0.28.1"},
        "metrics": {
            "decode_tok_per_s": 42.0, "throughput_tok_per_s": 80.0, "gpu_kv_cache_usage": 0.125,
            "requests_running": 2.0, "block_size": 16.0, "num_gpu_blocks": 100.0,
            "sampled_at": 1_700_000_000_000,
        },
        "error": None,
    }


def _fake_collectors(monkeypatch, calls: Counter) -> None:
    async def fake_probe(base_url, timeout=5.0, **kw):
        calls["probe"] += 1
        calls["version"] += bool(kw.get("version", True))
        return _probe(base_url)

    def fake_hw(tel=None):
        calls["hw"] += 1
        return {
            "gpu_sku": "NVIDIA GB10", "available_gib": 30.0, "ram_gib": 121.7, "power_w": 11.5,
            "swap_total_gib": 16.0, "swap_used_gib": 8.0, "sampled_at": 2_000 + calls["hw"],
        }

    def fake_containers():
        calls["containers"] += 1
        return [{"name": "spark-vllm", "status": "Up 2 hours", "image": "vllm/vllm-openai:v0.27.1", "id": "abc"}]

    def fake_cluster(local_telemetry=None, local_containers=None):
        calls["cluster"] += 1
        calls["cluster_reused_ps"] += local_containers is not None
        return {
            "nodes": [
                {"id": "a", "local": True, "state": "serving", "power_w": 9.0, "sampled_at": 1_000},
                {"id": "b", "local": False, "ssh_host": "b.invalid", "state": "serving_worker", "power_w": 7.0, "sampled_at": 1_000},
            ],
            "summary": {"healthy": True},
        }

    def fake_inspect(name):
        calls["inspect"] += 1
        return {"cmd": ["--port", "8000"], "started_at": "2026-09-05T15:53:25.963708557Z"}

    monkeypatch.setattr(metadata, "probe_endpoint", fake_probe)
    monkeypatch.setattr(metadata, "collect_hardware", fake_hw)
    monkeypatch.setattr(metadata, "list_vllm_containers", fake_containers)
    monkeypatch.setattr(metadata, "docker_inspect_flags", fake_inspect)
    monkeypatch.setattr(cluster, "collect_cluster", fake_cluster)
    monkeypatch.setattr(agentic, "tool_eval_available", lambda: pytest.fail("tool-eval probed by the sampler"))
    monkeypatch.setattr(cluster, "PeerStream", FakeStream)


def _now_ms() -> int:
    return int(time.time() * 1000)


class FakeStream:
    """Stands in for the per-peer ssh telemetry stream."""

    started: list["FakeStream"] = []

    def __init__(self, host, interval_s=1.0):
        self.host = host
        self.interval_s = interval_s
        self.error = None
        self.stopped = False
        self.line = {"power_w": 13.3, "available_gib": 18.5, "sampled_at": _now_ms()}

    def start(self):
        FakeStream.started.append(self)

    def stop(self):
        self.stopped = True

    def reading(self):
        return dict(self.line)


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
    assert set(snap) == STATUS_KEYS
    assert set(first) == STATUS_KEYS | {"stale_s"}
    assert snap["healthy"] is True
    assert snap["model_id"] == "org/m"
    # 30 GiB available on a serving GB10 is healthy, not "tight" (old fixed 60 GiB line)
    assert snap["headroom"] == "ok"
    assert snap["containers"][0]["name"] == "spark-vllm"
    assert snap["sampled_at"].endswith("+00:00")
    assert isinstance(snap["sampled_at_ms"], int)
    # the endpoint rate lives on serve.metrics only — never copied onto TP nodes
    assert snap["metrics"]["decode_tok_per_s"] == 42.0
    for n in snap["cluster"]["nodes"]:
        assert "gen_tok_per_s" not in n and "decode_tok_per_s" not in n
    # the local node shows the fast tick's telemetry, not the slow tick's
    local, remote = snap["cluster"]["nodes"]
    assert local["power_w"] == 11.5 and local["sampled_at"] == 2_001
    assert local["mem_pressure"] == "ok"
    assert remote["power_w"] == 7.0  # one-shot sampling starts no ssh streams
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
    assert dict(calls) == {
        "probe": 1, "version": 1, "hw": 1, "containers": 1, "cluster": 1, "cluster_reused_ps": 0,
        "inspect": 1,
    }
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
    assert set(body) == STATUS_KEYS | {"stale_s"}
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
    # after a fast tick, the slow tick reuses its `docker ps` instead of running its own
    assert calls["cluster_reused_ps"] == 1 and calls["containers"] == 1
    assert before["cluster"]["pending"] is True
    assert after["cluster"]["nodes"][0]["power_w"] == 11.5
    assert after["sampled_at"] == before["sampled_at"]


def test_sampler_loop_survives_exceptions_and_first_status_awaits_a_sample(monkeypatch):
    calls: Counter = Counter()
    _fake_collectors(monkeypatch, calls)
    flaky = {"n": 0}

    async def flaky_probe(base_url, timeout=5.0, **kw):
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
    assert set(body) == STATUS_KEYS | {"stale_s"}
    assert body["healthy"] is True and body["model_id"] == "org/m"
    assert body["sampled_at"] and isinstance(body["stale_s"], (int, float))
    assert body["metrics"]["decode_tok_per_s"] == 42.0


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


# ─── memory pressure (headroom) ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "tel,expected",
    [
        (None, "ok"),
        ({}, "ok"),
        # a healthy GB10 TP serve: most RAM pre-allocated by vLLM, swap half used
        ({"available_gib": 13.3, "swap_total_gib": 16.0, "swap_used_gib": 8.1}, "ok"),
        ({"available_gib": 6.0, "swap_total_gib": 16.0, "swap_used_gib": 8.0}, "tight"),
        # RAM nearly gone but swap still has room: tight, not yet the kill line
        ({"available_gib": 3.0, "swap_total_gib": 16.0, "swap_used_gib": 4.0}, "tight"),
        # both RAM and swap near the OOM-guard floor (2 GiB each)
        ({"available_gib": 3.0, "swap_total_gib": 16.0, "swap_used_gib": 13.0}, "critical"),
        ({"available_gib": 3.0, "swap_total_gib": None}, "critical"),
        ({"available_gib": 40.0, "mem_psi_full_avg10": 2.5}, "tight"),
        ({"available_gib": 40.0, "mem_psi_full_avg10": 12.0}, "critical"),
    ],
)
def test_headroom_judges_the_failure_mode_not_a_fixed_gib_line(tel, expected):
    assert status_sampler.headroom_for(tel) == expected


def test_headroom_floor_is_configurable():
    tel = {"available_gib": 10.0, "swap_total_gib": 0.0, "swap_used_gib": 0.0}
    assert status_sampler.headroom_for(tel, floor_gib=2.0) == "ok"
    assert status_sampler.headroom_for(tel, floor_gib=6.0) == "critical"


def test_status_headroom_is_the_worst_node(monkeypatch):
    calls: Counter = Counter()
    _fake_collectors(monkeypatch, calls)

    def starving_worker(local_telemetry=None, local_containers=None):
        return {
            "nodes": [
                {"id": "a", "local": True, "state": "serving"},
                {"id": "b", "local": False, "online": True, "state": "serving_worker", "available_gib": 1.5,
                 "swap_total_gib": 16.0, "swap_used_gib": 15.0, "sampled_at": 1},
            ],
            "summary": {"healthy": True},
        }

    monkeypatch.setattr(cluster, "collect_cluster", starving_worker)
    s = StatusSampler("http://127.0.0.1:8000")

    async def go():
        await s.sample_cluster()
        return await s.sample()

    snap = asyncio.run(go())
    assert snap["cluster"]["nodes"][0]["mem_pressure"] == "ok"
    assert snap["cluster"]["nodes"][1]["mem_pressure"] == "critical"
    assert snap["headroom"] == "critical"


# ─── telemetry overlay ────────────────────────────────────────────────────────


def test_publish_overlays_fresh_telemetry_without_mutating_the_slow_tick_cluster():
    now = _now_ms()
    info = {
        "nodes": [
            {"id": "a", "local": True, "power_w": 9.0, "sampled_at": now - 9_000},
            {"id": "b", "local": False, "online": True, "power_w": 7.0, "available_gib": 18.0, "sampled_at": now - 9_000},
            {"id": "c", "local": False, "online": True, "power_w": 6.0, "sampled_at": now - 1_000},
        ]
    }
    fresh_b, older_c = FakeStream("b"), FakeStream("c")
    older_c.line = {"power_w": 99.0, "sampled_at": now - 2_000}
    older_c.error = "telemetry stream exited (255)"
    out = status_sampler._publish_cluster(
        info, {"power_w": 11.0, "available_gib": 13.3, "sampled_at": now}, {"b": fresh_b, "c": older_c}
    )
    a, b, c = out["nodes"]
    assert a["power_w"] == 11.0 and a["sampled_at"] == now
    assert b["power_w"] == 13.3 and b["available_gib"] == 18.5 and b["sampled_at"] == fresh_b.line["sampled_at"]
    # an older stream line never overwrites a newer slow-tick reading; its error is surfaced
    assert c["power_w"] == 6.0 and c["sampled_at"] == now - 1_000
    assert c["telemetry_error"] == "telemetry stream exited (255)"
    assert b["mem_pressure"] == "ok" and c["mem_pressure"] is None
    # the cached slow-tick dicts (and earlier snapshots built from them) are untouched
    assert info["nodes"][0]["power_w"] == 9.0 and "mem_pressure" not in info["nodes"][1]


def test_a_dead_peer_never_shows_its_last_stream_line_as_current():
    """Offline node + a cached line from before it died: no numbers, no pressure, no headroom vote."""
    now = _now_ms()
    # what _probe_remote_ssh returns when ssh fails: every live field None
    offline = cluster.apply_telemetry({"id": "b", "local": False, "online": False, "state": "offline"}, None)
    last = FakeStream("b")
    last.line = {"available_gib": 3.0, "swap_total_gib": 16.0, "swap_used_gib": 15.0,
                 "temperature_c": 71.0, "sampled_at": now}
    b = status_sampler._publish_cluster({"nodes": [offline]}, None, {"b": last})["nodes"][0]
    assert b["state"] == "offline"
    assert b["available_gib"] is None and b["temperature_c"] is None and b["mem_pressure"] is None


def test_a_wedged_stream_falls_back_to_the_slow_tick_reading():
    """Online node whose stream stopped printing: its line ages out after 3 intervals."""
    now = _now_ms()
    node = {"id": "b", "local": False, "online": True, "power_w": 7.0, "available_gib": 20.0,
            "sampled_at": now - 8_000}
    wedged = FakeStream("b", interval_s=1.0)
    wedged.line = {"power_w": 99.0, "available_gib": 3.0, "sampled_at": now - 4_000}
    b = status_sampler._publish_cluster({"nodes": [node]}, None, {"b": wedged})["nodes"][0]
    assert b["power_w"] == 7.0 and b["available_gib"] == 20.0 and b["sampled_at"] == now - 8_000
    wedged.line["sampled_at"] = now - 500
    b = status_sampler._publish_cluster({"nodes": [node]}, None, {"b": wedged})["nodes"][0]
    assert b["power_w"] == 99.0


def test_streams_follow_the_remote_topology(monkeypatch):
    monkeypatch.setattr(cluster, "PeerStream", FakeStream)
    FakeStream.started.clear()
    s = StatusSampler("http://127.0.0.1:8000")
    s._tasks = [object()]  # running: streams are allowed

    async def go():
        await s._sync_streams([{"id": "a", "local": True}, {"id": "b", "ssh_host": "b.lan"}])
        first = dict(s._streams)
        await s._sync_streams([{"id": "a", "local": True}])
        return first

    first = asyncio.run(go())
    assert list(first) == ["b"] and first["b"].host == "b.lan"
    assert [x.host for x in FakeStream.started] == ["b.lan"]
    assert first["b"].stopped and s._streams == {}


# ─── fixed-rate loop, endpoint following, rate resets ─────────────────────────


def test_loop_is_fixed_rate_not_interval_plus_work():
    s = StatusSampler("http://127.0.0.1:8000")
    starts: list[float] = []

    async def tick():
        starts.append(asyncio.get_running_loop().time())
        await asyncio.sleep(0.04)  # work that used to be added to every period

    async def go():
        task = asyncio.create_task(s._loop(tick, 0.1))
        await asyncio.sleep(0.75)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(go())
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert len(gaps) >= 5
    assert abs(sum(gaps) / len(gaps) - 0.1) < 0.01, gaps


def test_sampler_follows_the_port_the_local_serve_answers_on(monkeypatch):
    calls: Counter = Counter()
    _fake_collectors(monkeypatch, calls)
    monkeypatch.setattr(
        cluster, "collect_cluster",
        lambda local_telemetry=None, local_containers=None: {"nodes": [
            {"id": "a", "local": True, "endpoint_healthy": True, "vllm_url": "http://127.0.0.1:8888"}
        ]},
    )
    s = StatusSampler("http://127.0.0.1:8000")
    probed: list[str] = []
    real = metadata.probe_endpoint

    async def record(base_url, timeout=5.0, **kw):
        probed.append(base_url)
        return await real(base_url, timeout, **kw)

    monkeypatch.setattr(metadata, "probe_endpoint", record)

    async def go():
        await s.sample()
        s._need_version = False
        metadata._LIVE_RATE["prev"] = {"generation_tokens_total": 5.0}
        await s.sample_cluster()
        # the slow tick only records the move: the fast tick owns the rate state
        assert s.base_url == "http://127.0.0.1:8000"
        assert metadata._LIVE_RATE["prev"] == {"generation_tokens_total": 5.0}
        reset_before_probe = []
        monkeypatch.setattr(
            metadata, "probe_endpoint",
            lambda base_url, timeout=5.0, **kw: (reset_before_probe.append(metadata._LIVE_RATE["prev"]), record(base_url, timeout, **kw))[1],
        )
        snap = await s.sample()
        return reset_before_probe, snap

    reset_before_probe, snap = asyncio.run(go())
    assert probed == ["http://127.0.0.1:8000", "http://127.0.0.1:8888"]
    assert reset_before_probe == [None]
    assert s.base_url == snap["base_url"] == "http://127.0.0.1:8888"
    assert s._pending_url is None


def test_version_is_fetched_once_and_again_after_the_container_changes(monkeypatch):
    calls: Counter = Counter()
    _fake_collectors(monkeypatch, calls)
    s = StatusSampler("http://127.0.0.1:8000", slow_s=1000.0)

    async def go():
        for _ in range(3):
            snap = await s.sample()
        assert calls["version"] == 1 and snap["version"] == {"version": "0.28.1"}
        metadata._LIVE_RATE["prev"] = {"generation_tokens_total": 5.0}
        monkeypatch.setattr(
            metadata, "list_vllm_containers",
            lambda: [{"name": "other", "status": "Up 1 second", "image": "vllm/vllm-openai:latest", "id": "z"}],
        )
        await s.sample()  # container changed → counters reset, /version refetched next tick
        assert metadata._LIVE_RATE["prev"] is None
        await s.sample()

    asyncio.run(go())
    assert calls["version"] == 2


def test_a_missed_metrics_scrape_keeps_the_last_gauges_but_no_live_rate(monkeypatch):
    calls: Counter = Counter()
    _fake_collectors(monkeypatch, calls)
    s = StatusSampler("http://127.0.0.1:8000")

    async def no_metrics(base_url, timeout=5.0, **kw):
        return {**_probe(base_url), "metrics": {}}

    async def go():
        await s.sample()
        monkeypatch.setattr(metadata, "probe_endpoint", no_metrics)
        return await s.sample()

    snap = asyncio.run(go())
    # gauges/counters of the last scrape stay (with their timestamp) …
    assert snap["metrics"]["requests_running"] == 2.0
    assert snap["metrics"]["sampled_at"] == 1_700_000_000_000
    # … but no window was measured, so no rate is presented as live
    for key in metadata.LIVE_RATE_KEYS:
        assert snap["metrics"][key] is None, key


# ─── routes read the cache ────────────────────────────────────────────────────


def test_cluster_and_hardware_routes_serve_the_sampler_cache(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main_mod

    monkeypatch.setattr(main_mod, "_LAIL_TOKEN", "")
    s = _sampled(monkeypatch)
    monkeypatch.setattr(status_sampler, "SAMPLER", s)
    _no_collectors(monkeypatch)

    client = TestClient(main_mod.app)
    c = client.get("/api/cluster")
    h = client.get("/api/hardware")
    assert c.status_code == 200 and h.status_code == 200, (c.text, h.text)
    assert c.json()["nodes"][0]["power_w"] == 11.5
    assert h.json()["power_w"] == 11.5


def test_cluster_cache_age_gate():
    s = StatusSampler("http://127.0.0.1:8000")
    assert s.cluster() is None  # pending
    s._cluster = {"nodes": [{"id": "a"}]}
    s._published_cluster = s._cluster
    s._cluster_mono = 0.0  # ancient
    assert s.cluster() is s._published_cluster
    assert s.cluster(max_age_s=30) is None


# ─── long poll (the controller's live stream) ─────────────────────────────────


def test_long_poll_returns_as_soon_as_a_newer_snapshot_is_published(monkeypatch):
    calls: Counter = Counter()
    _fake_collectors(monkeypatch, calls)
    s = StatusSampler("http://127.0.0.1:8000")

    async def go():
        await s.sample_cluster()
        first = await s.sample()
        at = first["sampled_at_ms"]
        # a caller that is behind gets the current snapshot at once
        behind = await asyncio.wait_for(s.status(after_ms=at - 1, wait_s=5), 0.5)
        waiter = asyncio.create_task(s.status(after_ms=at, wait_s=5))
        await asyncio.sleep(0.05)
        assert not waiter.done()  # nothing newer yet: it waits instead of answering stale
        await asyncio.sleep(0.01)  # a later wall-clock ms for the next publish
        t0 = time.monotonic()
        await s.sample()
        newer = await asyncio.wait_for(waiter, 1.0)
        # nothing published: the poll gives up after `wait_s` with the current snapshot
        idle = await s.status(after_ms=newer["sampled_at_ms"], wait_s=0.05)
        return at, behind, newer, time.monotonic() - t0, idle

    at, behind, newer, latency, idle = asyncio.run(go())
    assert behind["sampled_at_ms"] == at
    assert newer["sampled_at_ms"] > at and latency < 0.5
    assert idle["sampled_at_ms"] == newer["sampled_at_ms"]


def test_status_route_long_poll_params(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main_mod

    monkeypatch.setattr(main_mod, "_LAIL_TOKEN", "")
    s = _sampled(monkeypatch)
    monkeypatch.setattr(status_sampler, "SAMPLER", s)
    _no_collectors(monkeypatch)
    client = TestClient(main_mod.app)
    at = client.get("/api/status").json()["sampled_at_ms"]
    r = client.get(f"/api/status?after={at}&wait=0.05")
    assert r.status_code == 200 and r.json()["sampled_at_ms"] == at
    assert client.get(f"/api/status?after={at}&wait=60").status_code == 422


def test_serve_examples_route_carries_the_static_presets(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main_mod
    from app.config import MODEL_PRESETS, SERVE_EXAMPLES

    monkeypatch.setattr(main_mod, "_LAIL_TOKEN", "")
    body = TestClient(main_mod.app).get("/api/serve/examples").json()
    assert body == {"examples": SERVE_EXAMPLES, "presets": list(MODEL_PRESETS.keys())}


def test_published_models_drop_per_call_noise(monkeypatch):
    calls: Counter = Counter()
    _fake_collectors(monkeypatch, calls)

    async def noisy_probe(base_url, timeout=5.0, **kw):
        p = _probe(base_url)
        p["models"] = [{"id": "org/m", "max_model_len": 8192, "created": time.time(), "permission": [{"id": "x"}]}]
        return p

    monkeypatch.setattr(metadata, "probe_endpoint", noisy_probe)
    s = StatusSampler("http://127.0.0.1:8000")
    snap = asyncio.run(s.sample())
    assert snap["models"] == [{"id": "org/m", "max_model_len": 8192}]
    assert s.probe()["models"][0]["permission"]  # envelopes still get the full probe
