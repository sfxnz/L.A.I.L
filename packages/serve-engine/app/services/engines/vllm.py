"""vLLM: `vllm serve`. The reference engine — every other adapter maps onto its contract."""
from __future__ import annotations

import json
from typing import Any

from ...config import DEFAULT_IMAGE_MAX, DEFAULT_PORT, WORKFLOW_MAX_LEN, WORKFLOW_UTIL
from . import Engine, Rank, ServeSpec, bash_wrap, parse_extra, strip_flag

MASTER_PORT = 25000


def _mtp_speculative_json(
    *,
    num_tokens: int,
    moe_backend: str = "",
    extras: list[str] | None = None,
) -> str:
    """Structured MTP JSON, including Spark playbook keys (e.g. moe_backend:triton)."""
    spec: dict[str, Any] = {
        "method": "mtp",
        "num_speculative_tokens": int(num_tokens),
    }
    extra_keys: dict[str, Any] = {}
    args = extras or []
    i = 0
    while i < len(args):
        a = args[i]
        raw = ""
        if a == "--speculative-config" and i + 1 < len(args):
            raw = str(args[i + 1])
        elif a.startswith("--speculative-config="):
            raw = a.split("=", 1)[1]
        if raw:
            try:
                obj = json.loads(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                obj = None
            if isinstance(obj, dict):
                extra_keys.update(
                    {k: v for k, v in obj.items() if k not in ("method", "num_speculative_tokens")}
                )
        i += 1
    moe = (moe_backend or "").strip() or extra_keys.pop("moe_backend", None)
    if moe:
        spec["moe_backend"] = str(moe)
    spec.update(extra_keys)
    return json.dumps(spec, separators=(",", ":"))


def build_args(
    *,
    util: float,
    max_model_len: int,
    port: int,
    quantization: str = "",
    kv_cache_dtype: str = "",
    moe_backend: str = "",
    trust_remote_code: bool = False,
    enable_auto_tool_choice: bool = False,
    tool_call_parser: str = "",
    reasoning_parser: str = "",
    max_num_seqs: int | None = None,
    mtp: bool = False,
    mtp_num_tokens: int = 2,
    mtp_moe_backend: str = "",
    load_format: str = "",
    enable_chunked_prefill: bool = False,
    enable_prefix_caching: bool = False,
    extra_flags: str = "",
    tensor_parallel_size: int = 1,
) -> list[str]:
    """Assemble the vLLM CLI from explicit fields + free-form extras. Nothing silent."""
    args: list[str] = [
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(max(1, int(tensor_parallel_size or 1))),
        "--gpu-memory-utilization",
        str(util),
        "--max-model-len",
        str(max_model_len),
    ]
    if trust_remote_code:
        args.append("--trust-remote-code")
    if quantization.strip():
        args += ["--quantization", quantization.strip()]
    if kv_cache_dtype.strip():
        args += ["--kv-cache-dtype", kv_cache_dtype.strip()]
    if moe_backend.strip():
        args += ["--moe-backend", moe_backend.strip()]
    if max_num_seqs is not None and max_num_seqs > 0:
        args += ["--max-num-seqs", str(max_num_seqs)]
    if enable_auto_tool_choice:
        args.append("--enable-auto-tool-choice")
    if tool_call_parser.strip():
        args += ["--tool-call-parser", tool_call_parser.strip()]
    if reasoning_parser.strip():
        args += ["--reasoning-parser", reasoning_parser.strip()]
    if load_format.strip():
        args += ["--load-format", load_format.strip()]
    if enable_chunked_prefill:
        args.append("--enable-chunked-prefill")
    if enable_prefix_caching:
        args.append("--enable-prefix-caching")
    extras = parse_extra(extra_flags)
    if mtp:
        args += [
            "--speculative-config",
            _mtp_speculative_json(
                num_tokens=mtp_num_tokens,
                moe_backend=mtp_moe_backend,
                extras=extras,
            ),
        ]
        extras = strip_flag(extras, "--speculative-config")
    if quantization.strip():
        extras = strip_flag(extras, "--quantization")
        extras = strip_flag(extras, "-q")
    if kv_cache_dtype.strip():
        extras = strip_flag(extras, "--kv-cache-dtype")
    if moe_backend.strip():
        extras = strip_flag(extras, "--moe-backend")
    if tool_call_parser.strip():
        extras = strip_flag(extras, "--tool-call-parser")
    if reasoning_parser.strip():
        extras = strip_flag(extras, "--reasoning-parser")
    if load_format.strip():
        extras = strip_flag(extras, "--load-format")
    # Launcher owns the model (passed positionally) and the lab envelope owns
    # host/port/tp/util/max-len.
    for f in (
        "--model",
        "--host",
        "--port",
        "--tensor-parallel-size",
        "--gpu-memory-utilization",
        "--max-model-len",
    ):
        extras = strip_flag(extras, f)
    args += extras
    return args


def spec_args(spec: ServeSpec) -> list[str]:
    """`build_args` for a ServeSpec, with the workflow envelope filling unset util / max-len."""
    return build_args(
        util=WORKFLOW_UTIL if spec.util is None else spec.util,
        max_model_len=WORKFLOW_MAX_LEN if spec.max_model_len is None else spec.max_model_len,
        port=spec.port,
        quantization=spec.quantization,
        kv_cache_dtype=spec.kv_cache_dtype,
        moe_backend=spec.moe_backend,
        trust_remote_code=spec.trust_remote_code,
        enable_auto_tool_choice=spec.enable_auto_tool_choice,
        tool_call_parser=spec.tool_call_parser,
        reasoning_parser=spec.reasoning_parser,
        max_num_seqs=spec.max_num_seqs,
        mtp=spec.mtp,
        mtp_num_tokens=spec.mtp_num_tokens,
        mtp_moe_backend=spec.mtp_moe_backend,
        load_format=spec.load_format,
        enable_chunked_prefill=spec.enable_chunked_prefill,
        enable_prefix_caching=spec.enable_prefix_caching,
        extra_flags=spec.extra_flags,
        tensor_parallel_size=spec.tensor_parallel_size,
    )


_STRUCTURED = (
    "--tensor-parallel-size",
    "--pipeline-parallel-size",
    "--nnodes",
    "--node-rank",
    "--master-addr",
    "--master-port",
    "--distributed-executor-backend",
    "--host",
    "--port",
)


def strip_structured_flags(args: list[str]) -> list[str]:
    """Remove flags the multi-node launcher sets per rank, so one extra_flags blob is
    reused verbatim for head and workers without duplicates."""
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in _STRUCTURED:
            if i + 1 < len(args) and not str(args[i + 1]).startswith("-"):
                i += 2
            else:
                i += 1
            continue
        if any(a.startswith(f + "=") for f in _STRUCTURED):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def needs_bash_entrypoint(image: str) -> bool:
    """Anemll / DSpark images ship ENTRYPOINT=vllm; clear it for our bash wrapper."""
    img_l = (image or "").lower()
    return "anemll" in img_l or "dspark-vllm" in img_l or "gx10" in img_l


def argv(spec: ServeSpec, rank: Rank | None) -> list[str]:
    args = spec_args(spec)
    if rank is None:
        return ["vllm", "serve", spec.model, *args]
    suffix = [
        "vllm", "serve", spec.model,
        "--tensor-parallel-size", str(rank.nnodes),
        "--pipeline-parallel-size", "1",
        "--nnodes", str(rank.nnodes),
        "--node-rank", str(rank.rank),
        "--master-addr", rank.master_addr,
        "--master-port", str(rank.master_port),
        "--distributed-executor-backend", "mp",
    ]
    if rank.rank == 0:
        # Host network: the API binds loopback like a single-node serve (published on
        # 127.0.0.1 only). Rank traffic uses --master-addr on the fabric, not this socket.
        suffix += ["--host", "127.0.0.1", "--port", str(spec.port)]
    else:
        suffix += ["--headless"]
    return [*suffix, *strip_structured_flags(args)]


def container_args(image: str, cmd: list[str], multi: bool) -> list[str]:
    # Runtime images (Anemll) ship ENTRYPOINT=vllm: clear it for the bash wrapper. Multi-node
    # always wraps. The stock vllm-openai ENTRYPOINT already is `vllm serve`.
    if multi or needs_bash_entrypoint(image):
        return bash_wrap(image, cmd)
    return [image, *cmd[2:]]


# Prometheus name → normalized key. Series of one name that differ only by labels
# (engine="0", engine="1", … under data parallel) are SUMMED; when several names map
# to one key (old and new vLLM spellings) the first name present wins.
PROM_KEYS: tuple[tuple[str, str], ...] = (
    ("vllm:kv_cache_usage_perc", "gpu_kv_cache_usage"),
    ("vllm:gpu_cache_usage_perc", "gpu_kv_cache_usage"),
    # vLLM ≥ 0.10 exposes the counters with the Prometheus `_total` suffix.
    ("vllm:prefix_cache_hits_total", "prefix_cache_hits"),
    ("vllm:prefix_cache_hits", "prefix_cache_hits"),
    ("vllm:prefix_cache_queries_total", "prefix_cache_queries"),
    ("vllm:prefix_cache_queries", "prefix_cache_queries"),
    ("vllm:num_preemptions_total", "preemptions_total"),
    ("vllm:num_requests_running", "requests_running"),
    ("vllm:num_requests_waiting", "requests_waiting"),
    ("vllm:prompt_tokens_total", "prompt_tokens_total"),
    ("vllm:generation_tokens_total", "generation_tokens_total"),
    ("vllm:time_to_first_token_seconds_sum", "ttft_sum"),
    ("vllm:time_to_first_token_seconds_count", "ttft_count"),
    # Recorded once per engine step per running request: Σ is the busy decode time.
    ("vllm:inter_token_latency_seconds_sum", "itl_sum"),
    # Recorded when a request FINISHES (not at first token).
    ("vllm:request_prefill_time_seconds_sum", "prefill_time_s_sum"),
    ("vllm:request_prefill_time_seconds_count", "prefill_time_s_count"),
    ("vllm:request_prefill_kv_computed_tokens_sum", "prefill_tokens_sum"),
    ("vllm:spec_decode_num_drafts_total", "spec_drafts"),
    ("vllm:spec_decode_num_draft_tokens_total", "spec_draft_tokens"),
    ("vllm:spec_decode_num_accepted_tokens_total", "spec_accepted"),
)

ENGINE = Engine(
    name="vllm",
    label="vLLM",
    default_port=DEFAULT_PORT,
    argv=argv,
    container_args=container_args,
    # Workflow envelope: unset util / max-model-len fall back to WORKFLOW_UTIL / _MAX_LEN.
    image_env="LAB_VLLM_IMAGE_MAX",
    default_image=DEFAULT_IMAGE_MAX,
    fields=frozenset(ServeSpec.__dataclass_fields__),
    container="spark-vllm",
    metrics_prefix="vllm:",
    version=("/version", "version"),
    master_port=MASTER_PORT,
    prom_keys=PROM_KEYS,
    # A pool fraction per engine: summing engines would exceed 1 — report the fullest one.
    prom_max_keys=frozenset({"gpu_kv_cache_usage"}),
)
