"""llama.cpp: `llama-server` (the image's ENTRYPOINT).

Verified upstream (ggml-org/llama.cpp @ 32dd62ee): /metrics only with --metrics,
`llamacpp:*` names with no labels, generation counters folded in when a request ends
(slot reset), prompt counters per batch, no KV-usage gauge any more, 503 on every route
while loading, owned_by "llamacpp", -hf resolves through HF_HOME/hub. No tensor parallel
across nodes here: single node only.
"""
from __future__ import annotations

from pathlib import Path

from . import Engine, Rank, ServeSpec, parse_extra, strip_flags

_OWNED = ("-m", "--model", "-hf", "-hfr", "--hf-repo", "--host", "--port", "--metrics")
_HOST_HF = str(Path.home() / ".cache" / "huggingface")


def _model_args(model: str) -> list[str]:
    """`org/repo[:QUANT]` → -hf (downloaded into the shared HF cache); a path → -m, with the
    host HF cache rewritten to its in-container mount."""
    if model.startswith("/") or model.endswith(".gguf"):
        if model.startswith(_HOST_HF):
            model = "/cache/huggingface" + model[len(_HOST_HF):]
        return ["-m", model]
    return ["-hf", model]


def argv(spec: ServeSpec, rank: Rank | None) -> list[str]:
    if rank is not None:
        raise ValueError("llama.cpp has no multi-node tensor parallel in L.A.I.L — serve it on one node")
    cmd = ["llama-server", *_model_args(spec.model), "--host", "0.0.0.0", "--port", str(spec.port), "--metrics"]
    owned = list(_OWNED)
    if spec.max_model_len:
        cmd += ["-c", str(spec.max_model_len)]
        owned += ["-c", "--ctx-size"]
    return [*cmd, *strip_flags(parse_extra(spec.extra_flags), owned)]


def container_args(image: str, cmd: list[str], multi: bool) -> list[str]:
    # The image's ENTRYPOINT is llama-server (/app/llama-server, not on PATH).
    return [image, *cmd[1:]]


PROM_KEYS: tuple[tuple[str, str], ...] = (
    ("llamacpp:requests_processing", "requests_running"),
    ("llamacpp:requests_deferred", "requests_waiting"),
    ("llamacpp:tokens_predicted_total", "generation_tokens_total"),
    # Σ per-request generation time: the busy decode time (lands with the request).
    ("llamacpp:tokens_predicted_seconds_total", "itl_sum"),
    ("llamacpp:prompt_tokens_total", "prompt_tokens_total"),
    # Prompt tokens exclude cache hits: computed prefill tokens / prompt time.
    ("llamacpp:prompt_tokens_total", "prefill_tokens_sum"),
    ("llamacpp:prompt_seconds_total", "prefill_time_s_sum"),
    ("llamacpp:spec_decode_num_drafts_total", "spec_drafts"),
    ("llamacpp:spec_decode_num_draft_tokens_total", "spec_draft_tokens"),
    ("llamacpp:spec_decode_num_accepted_tokens_total", "spec_accepted"),
)

ENGINE = Engine(
    name="llamacpp",
    label="llama.cpp",
    default_port=8080,
    argv=argv,
    container_args=container_args,
    image_env="LAIL_LLAMACPP_IMAGE",
    # Multi-arch (amd64 + arm64) CUDA 13 server build, ENTRYPOINT /app/llama-server.
    default_image="ghcr.io/ggml-org/llama.cpp:server-cuda13",
    fields=frozenset({"max_model_len"}),
    container="lail-llamacpp",
    metrics_prefix="llamacpp:",
    version=("/props", "build_info"),
    rate_mode="finish",
    max_tp=1,
    prom_keys=PROM_KEYS,
    notes="Model is org/repo[:QUANT] (-hf) or a GGUF path; GPU offload etc. go in extra flags (-ngl 99 …).",
)
