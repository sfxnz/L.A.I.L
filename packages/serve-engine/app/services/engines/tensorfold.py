"""TensorFold: `tensorfold serve` (github.com/ashhart/TensorFold, CUDA backend on GB10).

Verified upstream (v0.6.0 @ c4646171): no PyPI package and no published image — it is
pip-installed from GitHub inside nvcr.io/nvidia/pytorch (--gpus all --ipc=host --network
host); the port opens only once the model is loaded (connection refused = loading); TP is
at most 2 ranks (--tp 2 --rank R --master HOST --master-port 29551, rank 1 runs no HTTP
server and must start first, devices /dev/infiniband + memlock + IPC_LOCK); owned_by
"tensorfold". Its /metrics token counters land only when a request finishes, while
/health `completion_tokens_total` includes replies still streaming — the live decode
rate is read from /health. `tensorfold:kv_cache_usage_ratio` is each stream's context
fill, not a KV pool, so it is never shown as KV %.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

from . import Engine, Rank, ServeSpec, bash_wrap, parse_extra, strip_flags

MASTER_PORT = 29551
# A source pip accepts; a prebuilt image that already has `tensorfold` skips the install.
# Pinned to the commit this adapter's flags and metric names were verified against (a full
# sha, so pip also caches the wheel it builds).
PIP_SOURCE = os.environ.get(
    "LAIL_TENSORFOLD_PIP", "git+https://github.com/ashhart/TensorFold.git@c4646171139ee8a3c38103eaa1699dad226ec12b"
)
# Host cache mounted over the container's /root/.cache: pip's wheel cache and torch's JIT
# extension builds (~/.cache/torch_extensions) survive the `docker rm -f` of every Stop,
# so only the first start pays the install + CUDA kernel compile.
CACHE_DIR = os.environ.get("LAIL_TENSORFOLD_CACHE", str(Path.home() / ".cache" / "lail-tensorfold"))
KV_DTYPES = ("bf16", "int8", "int4")
_OWNED = ("--host", "--port", "--name", "--no-update-check", "--tp", "--rank", "--master", "--master-port")


def argv(spec: ServeSpec, rank: Rank | None) -> list[str]:
    cmd = [
        "tensorfold", "serve", spec.model,
        # Host network: bind loopback (rank 1 opens no HTTP server at all).
        "--host", "127.0.0.1",
        "--port", str(spec.port),
        # Serve under the id the operator launched, like vLLM (default: the repo basename).
        "--name", spec.model,
        "--no-update-check",
    ]
    owned = list(_OWNED)
    if spec.max_model_len:
        # Unset: CUDA sizes the affordable native window itself.
        cmd += ["--context", str(spec.max_model_len)]
        owned.append("--context")
    if spec.kv_cache_dtype.strip():
        # bf16 | int8 | int4 (KV_DTYPES); anything else is refused by tensorfold at startup.
        cmd += ["--kv-dtype", spec.kv_cache_dtype.strip()]
        owned.append("--kv-dtype")
    if rank is not None:
        cmd += [
            "--tp", str(rank.nnodes),
            "--rank", str(rank.rank),
            "--master", rank.master_addr,
            "--master-port", str(rank.master_port),
        ]
    return [*cmd, *strip_flags(parse_extra(spec.extra_flags), owned)]


def container_args(image: str, cmd: list[str], multi: bool) -> list[str]:
    # The install resolves TensorFold's PyPI dependencies at run time: run it without the
    # HF token the container carries for `tensorfold serve`, so a bad dependency cannot read it.
    prelude = (
        "command -v tensorfold >/dev/null 2>&1 || "
        f"env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN pip install {shlex.quote(PIP_SOURCE)} || exit 1"
    )
    return bash_wrap(image, cmd, prelude=prelude)


def derive(m: dict[str, Any], health: Any) -> dict[str, Any]:
    """Counters from /health: completion tokens are live; prefill/decode seconds, cached
    tokens and draft counts are folded in when a request finishes. `rounds_total` is not
    read: upstream does not say whether it counts drafting rounds only, so no tokens/step."""
    if not isinstance(health, dict):
        return m
    num = lambda k: float(health[k]) if isinstance(health.get(k), (int, float)) and not isinstance(health.get(k), bool) else None  # noqa: E731
    if num("completion_tokens_total") is not None:
        m["generation_tokens_total"] = num("completion_tokens_total")
    if m.get("requests_running") is None and num("requests_running") is not None:
        m["requests_running"] = num("requests_running")
    for src, dst in (
        ("decode_seconds_total", "itl_sum"),
        ("prefill_seconds_total", "prefill_time_s_sum"),
        ("drafted_total", "spec_draft_tokens"),
        ("accepted_total", "spec_accepted"),
        ("context_length", "context_length"),
    ):
        if num(src) is not None:
            m[dst] = num(src)
    prompt, cached = num("prompt_tokens_total"), num("cached_tokens_total")
    if prompt is not None:
        m["prompt_tokens_total"] = prompt
        m["prefill_tokens_sum"] = prompt - (cached or 0.0)
    streams = health.get("streams")
    if isinstance(streams, dict) and isinstance(streams.get("decoding"), int):
        m["decode_streams"] = float(streams["decoding"])
    return m


PROM_KEYS: tuple[tuple[str, str], ...] = (
    ("tensorfold:requests_running", "requests_running"),
    ("tensorfold:requests_waiting", "requests_waiting"),
    ("tensorfold:time_to_first_token_seconds_sum", "ttft_sum"),
    ("tensorfold:time_to_first_token_seconds_count", "ttft_count"),
)

ENGINE = Engine(
    name="tensorfold",
    label="TensorFold",
    # Upstream default is 8080 — llama.cpp's too. L.A.I.L always passes --port; 8090 is free.
    default_port=8090,
    argv=argv,
    container_args=container_args,
    image_env="LAIL_TENSORFOLD_IMAGE",
    default_image="nvcr.io/nvidia/pytorch:26.07-py3",
    fields=frozenset({"max_model_len", "tensor_parallel_size", "kv_cache_dtype"}),
    container="lail-tensorfold",
    metrics_prefix="tensorfold:",
    rate_mode="wall",
    # First start pip-installs TensorFold and compiles its CUDA kernels before loading.
    ready_timeout_s=30 * 60,
    max_tp=2,
    master_port=MASTER_PORT,
    host_network=True,
    prom_keys=PROM_KEYS,
    prom_hists=(("tensorfold:time_to_first_token_seconds", "ttft"),),
    derive=derive,
    docker_opts=("-v", f"{CACHE_DIR}:/root/.cache"),
    notes=(
        "Installs from GitHub at container start unless the image already has `tensorfold` "
        "(set LAIL_TENSORFOLD_IMAGE to a prebuilt one); pip and kernel-build caches persist in "
        "~/.cache/lail-tensorfold. TP ≤ 2; GLM-5.3-Flash needs 2 ranks."
    ),
)
