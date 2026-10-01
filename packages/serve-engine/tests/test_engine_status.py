"""`serve.engine` in /api/status: vLLM metrics parsing, KV geometry, container flags."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.services import metadata

FIXTURE = Path(__file__).parent / "fixtures" / "vllm_metrics_qwen38_flash_next.prom"
ENGINE_KEYS = {
    "kv_usage_pct", "requests_running", "requests_waiting", "block_size", "num_gpu_blocks",
    "kv_capacity_tokens", "max_model_len", "version", "prefix_cache_hit_rate",
    "preemptions_total", "sleep_state", "uptime_s", "flags_fingerprint", "flags",
}


def _probe(metrics: dict) -> dict:
    return {
        "healthy": True,
        "models": [{"id": "nvidia/Qwen3.8-Flash-Next-NVFP4", "max_model_len": 262144}],
        "version": {"version": "0.28.1rc1.dev437+ge962733e0"},
        "metrics": metrics,
    }


def test_parse_prometheus_reads_real_vllm_scrape_incl_cache_config_labels():
    metadata.reset_live_rate_state()
    m = metadata.parse_prometheus(FIXTURE.read_text())
    assert m["requests_running"] == 0.0
    assert m["requests_waiting"] == 0.0
    assert m["gpu_kv_cache_usage"] == 0.0
    assert m["block_size"] == 1600.0
    assert m["num_gpu_blocks"] == 1393.0
    assert m["kv_cache_size_tokens"] == 1973873.0
    assert m["preemptions_total"] == 0.0
    # `_total`-suffixed counters map onto the historical keys
    assert m["prefix_cache_hits"] == 9.8048e06
    assert m["prefix_cache_queries"] == 1.098554e07
    assert m["prefix_cache_hit_rate_live"] is None  # first sample: no delta yet


def test_prefix_cache_hit_rate_is_a_window_delta_not_cumulative():
    metadata.reset_live_rate_state()
    metadata.live_token_rates({"prefix_cache_hits": 900.0, "prefix_cache_queries": 1000.0}, now=1.0)
    idle = metadata.live_token_rates({"prefix_cache_hits": 900.0, "prefix_cache_queries": 1000.0}, now=3.0)
    assert idle["prefix_cache_hit_rate_live"] is None
    busy = metadata.live_token_rates({"prefix_cache_hits": 925.0, "prefix_cache_queries": 1100.0}, now=5.0)
    assert busy["prefix_cache_hit_rate_live"] == 0.25


def test_build_engine_from_real_scrape_and_container():
    metadata.reset_live_rate_state()
    metrics = metadata.parse_prometheus(FIXTURE.read_text())
    metrics["gpu_kv_cache_usage"] = 0.4237
    inspect = {
        "cmd": ["--port", "8000", "--max-model-len", "262144", "--hf-token", "hf_abcdefghijklmnop"],
        "started_at": "2026-09-05T15:53:25.963708557Z",
    }
    now = datetime(2026, 9, 5, 16, 53, 25, tzinfo=timezone.utc)
    eng = metadata.build_engine(_probe(metrics), inspect, now=now)
    assert set(eng) == ENGINE_KEYS
    assert eng["kv_usage_pct"] == 42.37
    assert eng["requests_running"] == 0 and eng["requests_waiting"] == 0
    assert eng["block_size"] == 1600 and eng["num_gpu_blocks"] == 1393
    # hybrid cache: vLLM's own token capacity, not 1600 × 1393
    assert eng["kv_capacity_tokens"] == 1973873
    assert eng["max_model_len"] == 262144
    assert eng["version"] == "0.28.1rc1.dev437+ge962733e0"
    assert eng["prefix_cache_hit_rate"] is None
    assert eng["preemptions_total"] == 0
    assert eng["uptime_s"] == 3599.0  # started 15:53:25.96, now 16:53:25
    assert eng["flags"] == ["--port", "8000", "--max-model-len", "262144", "--hf-token", "<redacted>"]
    assert eng["flags_fingerprint"] == metadata.flags_fingerprint(eng["flags"])
    assert len(eng["flags_fingerprint"]) == 8
    assert "hf_abcdefghijklmnop" not in repr(eng)


def test_build_engine_kv_capacity_falls_back_to_block_product():
    eng = metadata.build_engine(_probe({"block_size": 16.0, "num_gpu_blocks": 1000.0}), {})
    assert eng["kv_capacity_tokens"] == 16000


def test_build_engine_is_all_null_without_vllm_metrics_or_container():
    # llama.cpp: /metrics has no vllm:* series, /version may be plain text, no docker container.
    probe = {"healthy": True, "models": [{"id": "gguf"}], "version": "b4567", "metrics": {}}
    eng = metadata.build_engine(probe, None)
    assert set(eng) == ENGINE_KEYS
    assert eng["version"] == "b4567"
    assert eng["flags"] == [] and eng["flags_fingerprint"] is None
    for k in ENGINE_KEYS - {"version", "flags"}:
        assert eng[k] is None, k
    assert metadata.build_engine(None, None)["version"] is None


def test_redact_flags_and_fingerprint_are_order_insensitive():
    flags = metadata.redact_flags(
        ["--api-key=sk-abcdefghijklmnop", "--tp", "2", "hf_0123456789abcdef", "--token", "x", "--max-num-batched-tokens", "8192"]
    )
    assert flags == ["--api-key=<redacted>", "--tp", "2", "<redacted>", "--token", "<redacted>", "--max-num-batched-tokens", "8192"]
    assert metadata.flags_fingerprint(["--a", "1", "--b"]) == metadata.flags_fingerprint(["--b", "--a", "1"])
    assert metadata.flags_fingerprint([]) is None


def test_uptime_parses_docker_nanosecond_timestamps():
    now = datetime(2026, 9, 5, 15, 54, 25, tzinfo=timezone.utc)
    assert metadata._uptime_s("2026-09-05T15:53:25.963708557Z", now) == 59.0
    assert metadata._uptime_s("0001-01-01T00:00:00Z", now) is not None  # never-started sentinel still parses
    assert metadata._uptime_s(None, now) is None
    assert metadata._uptime_s("garbage", now) is None
