"""The serving engines L.A.I.L launches and reads — one small adapter per engine.

An adapter owns only what differs between engines:

  launch    default image + port, how the engine-neutral `ServeSpec` (the placement
            result: util, context, TP, parsers, extras) becomes that engine's CLI for a
            single node or for one TP rank, and what the head binds to
  readiness the path that answers 200 only once the model is loaded
  telemetry which Prometheus names feed the shared `serve.metrics` contract, plus an
            optional hook for what an engine publishes differently (SGLang's live
            token counts, TensorFold's /health running totals)
  identity  `/v1/models` owned_by and the metrics prefix it is detected from

Docker / ssh / readiness / stop plumbing stays in `serve.py`; the live-rate maths
stays in `metadata.py`. Nothing here runs a process.
"""
from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ServeSpec:
    """One serve request, engine-neutral (what the Serve page and recommend produce).

    None / "" / False mean "not set": the engine's own default applies.
    """

    model: str
    port: int
    util: float | None = None
    max_model_len: int | None = None
    tensor_parallel_size: int = 1
    quantization: str = ""
    kv_cache_dtype: str = ""
    moe_backend: str = ""
    trust_remote_code: bool = False
    enable_auto_tool_choice: bool = False
    tool_call_parser: str = ""
    reasoning_parser: str = ""
    max_num_seqs: int | None = None
    mtp: bool = False
    mtp_num_tokens: int = 2
    mtp_moe_backend: str = ""
    load_format: str = ""
    enable_chunked_prefill: bool = False
    enable_prefix_caching: bool = False
    extra_flags: str = ""


# Fields every engine honours (they are docker / plumbing, not CLI flags).
COMMON_FIELDS = frozenset({"model", "port", "image", "docker_env", "extra_flags"})


@dataclass(frozen=True)
class Rank:
    """Where one TP rank sits: its index, the rank count, and rank 0's fabric address."""

    rank: int
    nnodes: int
    master_addr: str
    master_port: int


@dataclass(frozen=True)
class Engine:
    name: str
    label: str
    default_port: int
    # The engine process's command line (program first). rank None = single node; the
    # head of a multi-node serve binds loopback.
    argv: Callable[[ServeSpec, Rank | None], list[str]]
    # `docker run` tail after the shared options for that argv: [--entrypoint …] image
    # args… (image, argv, multi_node).
    container_args: Callable[[str, list[str], bool], list[str]]
    image_env: str
    default_image: str
    # ServeSpec fields this engine translates; any other field that is set is reported
    # as ignored in the job log (never silently dropped).
    fields: frozenset[str]
    container: str
    metrics_prefix: str
    # Path that answers 200 only once the model is loaded (SGLang's /v1/models may answer
    # while /health is still 503; TensorFold refuses connections until loaded).
    ready_path: str = "/v1/models"
    # Single-node load budget (multi-node serves get serve.MULTI_READY_TIMEOUT_S).
    ready_timeout_s: int = 10 * 60
    # Where the version lives: (path, JSON key); None = not exposed.
    version: tuple[str, str] | None = None
    # How live counters move (see metadata.live_token_rates):
    #   step    token + busy-time counters advance every engine step (vLLM, SGLang)
    #   wall    tokens advance live, busy time only when a request ends (TensorFold)
    #   finish  every generation counter lands when a request ends (llama.cpp)
    rate_mode: str = "step"
    max_tp: int | None = None
    master_port: int | None = None
    # Single node on the host network (bound to loopback) instead of a published port.
    host_network: bool = False
    # Prometheus name → normalized key (summed across label sets; *max_keys take the max).
    prom_keys: tuple[tuple[str, str], ...] = ()
    prom_max_keys: frozenset[str] = frozenset()
    # (metrics, /health JSON or None) → metrics: derived counters before the rate maths.
    derive: Callable[[dict[str, Any], Any], dict[str, Any]] | None = None
    # metrics → metrics: live gauges the engine computes itself, applied after the rates.
    live_gauges: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    # Extra container-level docker options (single node and every rank).
    docker_opts: tuple[str, ...] = field(default=())
    notes: str = ""

    def command(self, spec: ServeSpec, image: str, rank: Rank | None) -> list[str]:
        return self.container_args(image, self.argv(spec, rank), rank is not None)

    def image(self) -> str:
        return os.environ.get(self.image_env, "").strip() or self.default_image

    def rank_container(self, rank: int) -> str:
        return f"{self.container}-n{rank}"

    def describe(self) -> dict[str, Any]:
        """What the Serve page needs to offer this engine."""
        return {
            "name": self.name,
            "label": self.label,
            "default_port": self.default_port,
            "default_image": self.image(),
            "image_env": self.image_env,
            "max_tp": self.max_tp,
            "fields": sorted(self.fields | COMMON_FIELDS),
            "notes": self.notes,
        }


# ─── shared argv helpers ──────────────────────────────────────────────────────


def parse_extra(extra_flags: str) -> list[str]:
    s = (extra_flags or "").strip()
    return shlex.split(s) if s else []


def strip_flag(args: list[str], flag: str) -> list[str]:
    """Drop ``flag`` (+ value) from an argv-style list."""
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == flag:
            if i + 1 < len(args) and not str(args[i + 1]).startswith("-"):
                i += 2
            else:
                i += 1
            continue
        if a.startswith(flag + "="):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def strip_flags(args: list[str], flags: tuple[str, ...] | frozenset[str] | set[str]) -> list[str]:
    for f in flags:
        args = strip_flag(args, f)
    return args


# `--entrypoint bash` wrapper: bash's own flags follow the image (no second "bash"),
# `exec "$@"` runs the engine with "--" as $0. PATH covers CUDA + pip console scripts.
BASH_PATH = "export PATH=/usr/local/cuda/bin:/usr/local/bin:$PATH"


def bash_wrap(image: str, argv: list[str], prelude: str = "") -> list[str]:
    script = f"{BASH_PATH}; {prelude + '; ' if prelude else ''}exec \"$@\""
    return ["--entrypoint", "bash", image, "-lc", script, "--", *argv]


# ─── registry ─────────────────────────────────────────────────────────────────

from . import llamacpp, sglang, tensorfold, vllm  # noqa: E402  (adapters use the helpers above)

ENGINES: dict[str, Engine] = {
    e.name: e for e in (vllm.ENGINE, sglang.ENGINE, llamacpp.ENGINE, tensorfold.ENGINE)
}
VLLM = ENGINES["vllm"]

_ALIASES = {"llama.cpp": "llamacpp", "llama-cpp": "llamacpp", "gguf": "llamacpp", "sgl": "sglang", "sg-lang": "sglang"}


def get(name: str | None) -> Engine:
    """Adapter by name (aliases accepted); vLLM when None/empty. Unknown → ValueError."""
    key = (name or "vllm").strip().lower()
    key = _ALIASES.get(key, key)
    if key not in ENGINES:
        raise ValueError(f"unknown engine {name!r} (one of: {', '.join(ENGINES)})")
    return ENGINES[key]


def detect(models: list[Any] | None, metrics_text: str | None = None, hint: str | None = None) -> str | None:
    """Which engine is answering: /v1/models owned_by first, then the /metrics prefix,
    then `hint` (the serving container's engine). None when nothing says."""
    first = (models or [None])[0]
    owned = str(first.get("owned_by") or "").lower() if isinstance(first, dict) else ""
    if owned in ENGINES:
        return owned
    if metrics_text:
        for e in ENGINES.values():
            if f"\n{e.metrics_prefix}" in f"\n{metrics_text}":
                return e.name
    return hint if hint in ENGINES else None


def preview(engine: Engine, spec: ServeSpec, ranks: list[Rank] | None = None) -> list[dict[str, Any]]:
    """Dry-run command line per process (rank None = single node) for recommend / the UI."""
    if not ranks:
        return [{"rank": None, "argv": shlex.join(engine.argv(spec, None))}]
    return [{"rank": r.rank, "argv": shlex.join(engine.argv(spec, r))} for r in ranks]
