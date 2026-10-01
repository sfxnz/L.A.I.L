"""SGLang: `python3 -m sglang.launch_server`.

Verified upstream (sgl-project/sglang @ b51d4a04): --tp-size (not --tp), --nnodes /
--node-rank / --dist-init-addr for multi-node (rank ≥ 1 runs no HTTP server: a headless
worker), /health 503 while starting, /metrics only with --enable-metrics, owned_by
"sglang", `sglang:token_usage` a 0-1 fraction, server version on /server_info.
"""
from __future__ import annotations

from typing import Any

from . import Engine, Rank, ServeSpec, parse_extra, strip_flags

MASTER_PORT = 50000

# Flags this adapter always sets itself; a duplicate in extra_flags is dropped.
_OWNED = (
    "--model-path", "--model", "--host", "--port", "--tp-size", "--tensor-parallel-size",
    "--nnodes", "--node-rank", "--dist-init-addr", "--nccl-init-addr", "--enable-metrics",
)


def _kv_dtype(v: str) -> str:
    # SGLang spells vLLM's "fp8" (e4m3) explicitly.
    return "fp8_e4m3" if v == "fp8" else v


def argv(spec: ServeSpec, rank: Rank | None) -> list[str]:
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", spec.model,
        # Single node: inside the bridge network, published on 127.0.0.1 by docker.
        # Multi-node (host network): the head binds loopback; ranks talk over the fabric.
        "--host", "0.0.0.0" if rank is None else "127.0.0.1",
        "--port", str(spec.port),
        "--tp-size", str(rank.nnodes if rank else max(1, int(spec.tensor_parallel_size or 1))),
    ]
    if rank is not None:
        cmd += [
            "--nnodes", str(rank.nnodes),
            "--node-rank", str(rank.rank),
            "--dist-init-addr", f"{rank.master_addr}:{rank.master_port}",
        ]
    owned = list(_OWNED)
    for flag, value in (
        ("--mem-fraction-static", None if spec.util is None else str(spec.util)),
        ("--context-length", str(spec.max_model_len) if spec.max_model_len else None),
        ("--quantization", spec.quantization.strip() or None),
        ("--kv-cache-dtype", _kv_dtype(spec.kv_cache_dtype.strip()) or None),
        ("--max-running-requests", str(spec.max_num_seqs) if spec.max_num_seqs else None),
        ("--load-format", spec.load_format.strip() or None),
        ("--tool-call-parser", spec.tool_call_parser.strip() or None),
        ("--reasoning-parser", spec.reasoning_parser.strip() or None),
    ):
        if value is not None:
            cmd += [flag, value]
            owned.append(flag)
    if spec.trust_remote_code:
        cmd.append("--trust-remote-code")
        owned.append("--trust-remote-code")
    cmd.append("--enable-metrics")
    return [*cmd, *strip_flags(parse_extra(spec.extra_flags), owned)]


def container_args(image: str, cmd: list[str], multi: bool) -> list[str]:
    return ["--entrypoint", cmd[0], image, *cmd[1:]]


def derive(m: dict[str, Any], _health: Any) -> dict[str, Any]:
    """Per-stream decode from the histograms that advance per streamed chunk.

    Upstream @ b51d4a04: `generation_tokens_total` is incremented only in
    observe_one_finished_request (metrics_collector.py:1808), so it cannot drive a live
    rate. tokenizer_manager.collect_metrics (:3090-3115) observes TTFT once on a request's
    first chunk, then for each later chunk calls observe_inter_token_latency(interval,
    num_new_tokens), which adds `num_new_tokens` to the bucket counts and the chunk's
    interval to _sum (metrics_collector.py:1859-1881). So Δitl_count ÷ Δitl_sum is tokens
    per second of streaming time — the per-stream decode rate. A first chunk carrying
    several tokens (a non-streaming reply arrives as one chunk) counts once, so this token
    count is streamed tokens only: aggregate throughput comes from the scheduler's
    `gen_throughput` gauge instead (see live_gauges)."""
    if m.get("ttft_count") is not None and m.get("itl_count") is not None:
        m["generation_tokens_total"] = m["ttft_count"] + m["itl_count"]
    if m.get("kv_used_tokens") is not None and m.get("kv_available_tokens") is not None:
        m["kv_cache_size_tokens"] = m["kv_used_tokens"] + m["kv_available_tokens"]
    return m


def live_gauges(m: dict[str, Any]) -> dict[str, Any]:
    """Gauges the scheduler computes for the batch now running (not counters).

    `gen_throughput` is generated tokens ÷ wall time over the last decode_log_interval
    steps, every request streamed or not (metrics_reporter.py:900); it holds for up to 30 s
    after decode stops, so it is read only while requests run. When requests run but no
    chunk streamed this window (non-streaming replies), there is no per-stream reading."""
    if (m.get("requests_running") or 0) > 0:
        gen = m.get("gen_throughput_gauge")
        if gen is not None:
            m["throughput_tok_per_s"] = round(gen, 2)
            if gen > 0 and m.get("decode_tok_per_s") == 0.0:
                m["decode_tok_per_s"] = None
        if m.get("spec_accept_rate_gauge"):
            m["spec_accept_rate"] = round(m["spec_accept_rate_gauge"], 4)
        if m.get("spec_accept_length_gauge"):
            m["spec_tokens_per_step"] = round(m["spec_accept_length_gauge"], 2)
    return m


PROM_KEYS: tuple[tuple[str, str], ...] = (
    ("sglang:num_running_reqs", "requests_running"),
    ("sglang:num_queue_reqs", "requests_waiting"),
    ("sglang:gen_throughput", "gen_throughput_gauge"),
    ("sglang:token_usage", "gpu_kv_cache_usage"),
    ("sglang:kv_used_tokens", "kv_used_tokens"),
    ("sglang:kv_available_tokens", "kv_available_tokens"),
    ("sglang:prompt_tokens_total", "prompt_tokens_total"),
    ("sglang:time_to_first_token_seconds_sum", "ttft_sum"),
    ("sglang:time_to_first_token_seconds_count", "ttft_count"),
    ("sglang:inter_token_latency_seconds_sum", "itl_sum"),
    ("sglang:inter_token_latency_seconds_count", "itl_count"),
    ("sglang:spec_accept_rate", "spec_accept_rate_gauge"),
    ("sglang:spec_accept_length", "spec_accept_length_gauge"),
)

ENGINE = Engine(
    name="sglang",
    label="SGLang",
    default_port=30000,
    argv=argv,
    container_args=container_args,
    image_env="LAIL_SGLANG_IMAGE",
    default_image="lmsysorg/sglang:v0.5.20-cu130",
    fields=frozenset({
        "util", "max_model_len", "tensor_parallel_size", "quantization", "kv_cache_dtype",
        "trust_remote_code", "tool_call_parser", "reasoning_parser", "max_num_seqs", "load_format",
    }),
    container="lail-sglang",
    metrics_prefix="sglang:",
    ready_path="/health",
    version=("/server_info", "version"),
    master_port=MASTER_PORT,
    prom_keys=PROM_KEYS,
    prom_hists=(
        ("sglang:inter_token_latency_seconds", "itl"),
        ("sglang:time_to_first_token_seconds", "ttft"),
    ),
    # Scheduler gauges are per TP rank / mostrecent: the same batch seen by each rank.
    prom_max_keys=frozenset({
        "requests_running", "requests_waiting", "gen_throughput_gauge", "gpu_kv_cache_usage", "kv_used_tokens",
        "kv_available_tokens", "spec_accept_rate_gauge", "spec_accept_length_gauge",
    }),
    derive=derive,
    live_gauges=live_gauges,
    notes="util → --mem-fraction-static, max-model-len → --context-length; /metrics via --enable-metrics.",
)
