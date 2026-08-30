"""GPU probe parse and Spark node payload — shipped helpers only."""
from __future__ import annotations

from app.services import cluster, metadata


def test_parse_gpu_telemetry_reads_temp_usage_power():
    tel = metadata.parse_gpu_telemetry(
        "NVIDIA GB10, 47, 83, 32.1, [N/A], [N/A]\n"
    )
    assert tel["gpu_sku"] == "NVIDIA GB10"
    assert tel["temperature_c"] == 47
    assert tel["gpu_util_pct"] == 83
    assert tel["power_w"] == 32.1
    assert tel["memory_used_mib"] is None
    assert tel["memory_total_mib"] is None


def test_parse_gpu_telemetry_blank_is_nil_not_zero():
    tel = metadata.parse_gpu_telemetry("")
    assert tel["gpu_sku"] is None
    assert tel["temperature_c"] is None
    assert tel["gpu_util_pct"] is None
    assert tel["power_w"] is None
    assert tel["memory_used_mib"] is None


def test_parse_gpu_telemetry_na_fields_stay_none():
    tel = metadata.parse_gpu_telemetry("NVIDIA GB10, [N/A], [N/A], [N/A], [N/A], [N/A]")
    assert tel["gpu_sku"] == "NVIDIA GB10"
    assert tel["temperature_c"] is None
    assert tel["gpu_util_pct"] is None
    assert tel["power_w"] is None


def test_node_payload_includes_temperature_and_usage_from_smi():
    tel = metadata.parse_gpu_telemetry("NVIDIA GB10, 41.0, 7, 18.5, [N/A], [N/A]")
    node = cluster.apply_gpu_telemetry({}, tel)
    assert node["temperature_c"] == 41.0
    assert node["gpu_util_pct"] == 7
    assert node["power_w"] == 18.5
    assert node["memory_used_mib"] is None


def test_node_payload_nil_when_smi_returns_nothing():
    node = cluster.apply_gpu_telemetry({"id": "spark1"}, metadata.parse_gpu_telemetry(""))
    assert node["temperature_c"] is None
    assert node["gpu_util_pct"] is None
    assert node["power_w"] is None
    assert 0 not in (node["temperature_c"], node["gpu_util_pct"], node["power_w"])


def test_attach_live_rates_only_on_in_use_sparks():
    info = {
        "nodes": [
            {"id": "a", "state": "serving"},
            {"id": "b", "state": "serving_worker"},
            {"id": "c", "state": "idle"},
        ]
    }
    cluster.attach_live_rates(
        info, {"gen_tok_per_s": 41.2, "prompt_tok_per_s": 210.0}
    )
    assert info["nodes"][0]["gen_tok_per_s"] == 41.2
    assert info["nodes"][0]["prompt_tok_per_s"] == 210.0
    assert info["nodes"][1]["gen_tok_per_s"] == 41.2
    assert info["nodes"][2]["gen_tok_per_s"] is None
    assert info["nodes"][2]["prompt_tok_per_s"] is None


def test_live_token_rates_from_counter_delta():
    metadata.reset_live_rate_state()
    first = metadata.live_token_rates(
        {"generation_tokens_total": 100.0, "prompt_tokens_total": 50.0},
        now=10.0,
    )
    assert first.get("gen_tok_per_s") is None
    assert first.get("prompt_tok_per_s") is None
    second = metadata.live_token_rates(
        {"generation_tokens_total": 250.0, "prompt_tokens_total": 110.0},
        now=12.0,
    )
    assert second["gen_tok_per_s"] == 75.0
    assert second["prompt_tok_per_s"] == 30.0


def test_live_token_rates_prompt_delta_does_not_need_gen_to_move():
    metadata.reset_live_rate_state()
    metadata.live_token_rates(
        {"generation_tokens_total": 100.0, "prompt_tokens_total": 50.0},
        now=10.0,
    )
    only_prompt = metadata.live_token_rates(
        {"generation_tokens_total": 100.0, "prompt_tokens_total": 250.0},
        now=12.0,
    )
    assert only_prompt["prompt_tok_per_s"] == 100.0
    assert only_prompt.get("gen_tok_per_s") is None


def test_live_token_rates_holds_prefill_while_decode_counters_move():
    metadata.reset_live_rate_state()
    metadata.live_token_rates(
        {"generation_tokens_total": 10.0, "prompt_tokens_total": 20.0},
        now=1.0,
    )
    prefill = metadata.live_token_rates(
        {"generation_tokens_total": 10.0, "prompt_tokens_total": 220.0},
        now=3.0,
    )
    assert prefill["prompt_tok_per_s"] == 100.0
    decode = metadata.live_token_rates(
        {"generation_tokens_total": 210.0, "prompt_tokens_total": 220.0},
        now=5.0,
    )
    assert decode["gen_tok_per_s"] == 100.0
    assert decode["prompt_tok_per_s"] == 100.0
    idle = metadata.live_token_rates(
        {"generation_tokens_total": 210.0, "prompt_tokens_total": 220.0},
        now=7.0,
    )
    assert idle.get("gen_tok_per_s") is None
    assert idle.get("prompt_tok_per_s") is None


def test_parse_prometheus_zero_throughput_is_nil():
    metadata.reset_live_rate_state()
    parsed = metadata.parse_prometheus(
        "vllm:avg_generation_throughput_toks_per_s 0\n"
        "vllm:avg_prompt_throughput_toks_per_s 0\n"
    )
    assert parsed.get("gen_tok_per_s") is None
    assert parsed.get("prompt_tok_per_s") is None


def test_attach_live_rates_missing_metrics_are_nil_not_zero():
    info = {"nodes": [{"id": "a", "state": "serving"}]}
    cluster.attach_live_rates(info, {})
    assert info["nodes"][0]["gen_tok_per_s"] is None
    assert info["nodes"][0]["prompt_tok_per_s"] is None


def test_remote_probe_script_emits_temp_and_usage(monkeypatch):
    import io
    import urllib.request

    def fake_check_output(cmd, text=True, stderr=None, timeout=8):
        if cmd[0] == "nvidia-smi":
            return "NVIDIA GB10, 52, 91, 44.2, [N/A], [N/A]\n"
        if cmd[:2] == ["docker", "ps"]:
            return ""
        if cmd[0] == "free":
            return (
                "              total        used        free      shared  buff/cache   available\n"
                "Mem:            120          10          20           0          90          80\n"
            )
        return ""

    class FakeResp:
        def __init__(self, body: bytes):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(url, timeout=2.5):
        return FakeResp(b'{"data": []}')

    monkeypatch.setattr(cluster.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    real_open = open

    def fake_open(path, *a, **kw):
        p = str(path)
        if p == "/proc/meminfo":
            return io.StringIO("MemTotal:       126000000 kB\n")
        if "/sys/class/net/" in p:
            raise FileNotFoundError(p)
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)
    script = cluster._remote_probe_script(
        {"qsfp_if": "", "vllm_url": "http://127.0.0.1:8000"}
    )
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    exec(compile(script, "<remote-probe>", "exec"), {})
    import json

    data = json.loads(buf.getvalue().strip().splitlines()[-1])
    node = cluster.apply_gpu_telemetry({}, data)
    assert node["temperature_c"] == 52
    assert node["gpu_util_pct"] == 91
    assert node["power_w"] == 44.2
