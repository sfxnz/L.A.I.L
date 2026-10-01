"""Collect model / engine / hardware / metrics metadata for Run Envelopes."""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from ..config import DEFAULT_BASE_URL, MODEL_PRESETS
from . import node_probe


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    h = hashlib.sha1(f"{stamp}{time.time()}".encode()).hexdigest()[:6]
    return f"{stamp}_{h}"


def _run(cmd: list[str], timeout: float = 10) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=timeout)
    except Exception:
        return ""


def available_gib() -> float | None:
    """MemAvailable in GiB from /proc/meminfo (0.01 GiB precision)."""
    return node_probe.parse_meminfo(node_probe._read("/proc/meminfo"))["available_gib"]


@functools.lru_cache(maxsize=1)
def _host_facts() -> dict[str, Any]:
    """Static host facts, read once: real CPU model (lscpu), hostname, platform."""
    return {"cpu": node_probe.cpu_model(), "hostname": platform.node(), "platform": platform.platform()}


def collect_hardware(tel: node_probe.Telemetry | None = None) -> dict[str, Any]:
    """This host's telemetry + static facts. Pass the sampler's `Telemetry` so CPU % has a
    previous /proc/stat sample to diff against; a one-shot call reports cpu_util_pct None."""
    sample = (tel or node_probe.Telemetry()).sample()
    return {**sample, "gpu_sku": sample.get("gpu_sku") or "unknown", **_host_facts()}


def list_vllm_containers() -> list[dict[str, Any]]:
    return node_probe.list_serve_containers()


def docker_inspect_flags(name: str) -> dict[str, Any]:
    raw = _run(["docker", "inspect", name], timeout=15)
    if not raw:
        return {}
    try:
        data = json.loads(raw)[0]
    except Exception:
        return {}
    cfg = data.get("Config") or {}
    ns = data.get("NetworkSettings") or {}
    hc = data.get("HostConfig") or {}
    ports = sorted({*node_probe.ports_from_bindings(ns.get("Ports")), *node_probe.ports_from_bindings(hc.get("PortBindings"))})
    return {
        "image": cfg.get("Image"),
        "cmd": cfg.get("Cmd") or [],
        "args": data.get("Args") or [],
        "env": [e for e in (cfg.get("Env") or []) if not e.startswith("HF_TOKEN") and "TOKEN" not in e],
        "state": (data.get("State") or {}).get("Status"),
        "started_at": (data.get("State") or {}).get("StartedAt"),
        "ports": ports,
        "network_mode": hc.get("NetworkMode"),
    }


# Flag *names* ending in a secret word: `--hf-token`, `--api-key`; not `--max-num-batched-tokens`.
_SECRET_FLAG_RE = re.compile(r"(?:^|[-_])(?:token|secret|api[-_]?key|password)$", re.I)
_SECRET_VALUE_RE = re.compile(r"^(hf_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9_-]{10,})$")


def redact_flags(cmd: list[Any]) -> list[str]:
    """Container Cmd args with token-like values replaced. Flag names are kept."""
    out: list[str] = []
    hide_next = False
    for raw in cmd:
        arg = str(raw)
        if hide_next:
            out.append("<redacted>")
            hide_next = False
            continue
        if arg.startswith("-") and "=" in arg:
            name, _sep, _val = arg.partition("=")
            if _SECRET_FLAG_RE.search(name):
                out.append(f"{name}=<redacted>")
                continue
        elif arg.startswith("-") and _SECRET_FLAG_RE.search(arg):
            hide_next = True
        if _SECRET_VALUE_RE.match(arg):
            out.append("<redacted>")
            continue
        out.append(arg)
    return out


def flags_fingerprint(flags: list[str]) -> str | None:
    """First 8 hex of sha256 over the sorted (redacted) args; None when there are none."""
    if not flags:
        return None
    return hashlib.sha256("\n".join(sorted(flags)).encode()).hexdigest()[:8]


def _uptime_s(started_at: Any, now: datetime | None = None) -> float | None:
    if not isinstance(started_at, str) or not started_at:
        return None
    s = started_at.strip()
    # Docker prints nanoseconds; fromisoformat takes at most 6 fractional digits.
    m = re.match(r"^(.*?\.\d{1,6})\d*(Z|[+-]\d\d:\d\d)?$", s)
    if m:
        s = m.group(1) + (m.group(2) or "")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        started = datetime.fromisoformat(s)
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    up = (now - started).total_seconds()
    return round(up, 1) if up >= 0 else None


def _num_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_engine(
    probe: dict[str, Any] | None,
    inspect: dict[str, Any] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """`serve.engine` for /api/status: vLLM gauges, KV geometry, container flags.

    Every value is None when its source is absent (llama.cpp has no vllm:* metrics,
    a bare process has no container) — nothing is fabricated.
    """
    probe = probe or {}
    metrics = probe.get("metrics") or {}
    inspect = inspect or {}
    kv = metrics.get("gpu_kv_cache_usage")
    block_size = metrics.get("block_size")
    num_blocks = metrics.get("num_gpu_blocks")
    # vLLM publishes the exact token capacity for hybrid (Mamba/attention) caches, where
    # block_size × num_gpu_blocks over-counts; fall back to the product when it is absent.
    capacity = metrics.get("kv_cache_size_tokens")
    if capacity is None and block_size is not None and num_blocks is not None:
        capacity = block_size * num_blocks
    models = probe.get("models") or []
    max_len = models[0].get("max_model_len") if models and isinstance(models[0], dict) else None
    version = probe.get("version")
    if isinstance(version, dict):
        version = version.get("version")
    flags = redact_flags(inspect.get("cmd") or [])
    return {
        "kv_usage_pct": None if kv is None else round(kv * 100, 2),
        "requests_running": _int_or_none(metrics.get("requests_running")),
        "requests_waiting": _int_or_none(metrics.get("requests_waiting")),
        "block_size": _int_or_none(block_size),
        "num_gpu_blocks": _int_or_none(num_blocks),
        "kv_capacity_tokens": _int_or_none(capacity),
        "max_model_len": _int_or_none(max_len),
        "version": str(version) if version else None,
        "prefix_cache_hit_rate": _num_or_none(metrics.get("prefix_cache_hit_rate_live")),
        "preemptions_total": _int_or_none(metrics.get("preemptions_total")),
        "sleep_state": metrics.get("sleep_state"),
        "uptime_s": _uptime_s(inspect.get("started_at"), now),
        "flags_fingerprint": flags_fingerprint(flags),
        "flags": flags,
    }


def _int_or_none(v: Any) -> int | None:
    f = _num_or_none(v)
    return None if f is None else int(f)


async def probe_endpoint(
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 5.0,
    *,
    client: httpx.AsyncClient | None = None,
    version: bool = True,
) -> dict[str, Any]:
    """Probe /health, /v1/models, /version (when asked), /metrics concurrently.

    Parsing /metrics feeds the live tok/s counter deltas (`_LIVE_RATE`), so this
    must have one caller at a fixed cadence: the status sampler. Bench threads
    read the sampler's cached probe instead of calling this. The sampler passes
    its long-lived `client` and skips /version until the serving container changes.
    """
    base = base_url.rstrip("/")
    result: dict[str, Any] = {
        "base_url": base,
        "healthy": False,
        "models": [],
        "version": None,
        "metrics": {},
        "error": None,
    }
    health_ok = False
    errors: dict[str, str] = {}

    async def run(c: httpx.AsyncClient) -> None:
        async def health() -> None:
            nonlocal health_ok
            try:
                h = await c.get(f"{base}/health", timeout=timeout)
                health_ok = h.status_code == 200
            except Exception as e:
                errors["health"] = f"health: {e}"

        async def models() -> None:
            try:
                m = await c.get(f"{base}/v1/models", timeout=timeout)
                if m.status_code == 200:
                    body = m.json()
                    result["models"] = body.get("data") or []
                    result["healthy"] = True
            except Exception as e:
                errors["models"] = f" models: {e}"

        async def version_() -> None:
            try:
                v = await c.get(f"{base}/version", timeout=timeout)
                if v.status_code == 200:
                    result["version"] = v.json() if "application/json" in v.headers.get("content-type", "") else v.text
            except Exception:
                pass

        async def metrics() -> None:
            try:
                met = await c.get(f"{base}/metrics", timeout=timeout)
                if met.status_code == 200:
                    mono, wall_ms = time.monotonic(), int(time.time() * 1000)
                    # ~70 KB of text: parse off the event loop. One writer → rate state is safe.
                    parsed = await asyncio.to_thread(parse_prometheus, met.text, mono)
                    parsed["sampled_at"] = wall_ms
                    result["metrics"] = parsed
            except Exception:
                pass

        await asyncio.gather(health(), models(), *([version_()] if version else []), metrics())

    if client is not None:
        await run(client)
    else:
        async with httpx.AsyncClient(timeout=timeout) as c:
            await run(c)
    result["healthy"] = result["healthy"] or health_ok
    if errors:
        result["error"] = errors.get("health", "") + errors.get("models", "")
    return result


# Prometheus name → normalized key. Series of one name that differ only by labels
# (engine="0", engine="1", … under data parallel) are SUMMED; when several names map
# to one key (old and new vLLM spellings) the first name present wins.
_PROM_KEYS: tuple[tuple[str, str], ...] = (
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
_PROM_NAMES = {name for name, _ in _PROM_KEYS}
# A pool fraction per engine: summing engines would exceed 1 — report the fullest one.
_PROM_MAX_KEYS = {"gpu_kv_cache_usage"}
_PROM_LINE_RE = re.compile(r"^([a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+|NaN)")


def parse_prometheus(text: str, now: float | None = None) -> dict[str, Any]:
    """Extract the vLLM gauges/counters LAIL uses, then the live window rates."""
    by_name: dict[str, list[float]] = {}
    out: dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith("vllm:cache_config_info{"):
            # KV geometry lives in the labels of this info gauge (value is always 1).
            for label in ("block_size", "num_gpu_blocks", "kv_cache_size_tokens"):
                lm = re.search(rf'\b{label}="(\d+)"', line)
                if lm and label not in out:
                    out[label] = float(lm.group(1))
            continue
        if line.startswith("vllm:engine_sleep_state{"):
            sm = re.search(r'sleep_state="([^"]+)"\}\s+1(?:\.0)?\s*$', line)
            if sm:
                out["sleep_state"] = sm.group(1)
            continue
        m = _PROM_LINE_RE.match(line)
        if not m or m.group(1) not in _PROM_NAMES:
            continue
        val = float(m.group(2))
        if val == val:  # drop NaN
            by_name.setdefault(m.group(1), []).append(val)
    for name, key in _PROM_KEYS:
        if key in out or name not in by_name:
            continue
        vals = by_name[name]
        out[key] = max(vals) if key in _PROM_MAX_KEYS else sum(vals)
    if out.get("prefix_cache_queries"):
        out["prefix_cache_hit_rate"] = out.get("prefix_cache_hits", 0.0) / out["prefix_cache_queries"]
    return live_token_rates(out, now=now)


# Counters whose window deltas drive the live rates.
_RATE_COUNTERS = (
    "generation_tokens_total",
    "ttft_sum",
    "ttft_count",
    "itl_sum",
    "prefill_time_s_sum",
    "prefill_time_s_count",
    "prefill_tokens_sum",
    "spec_drafts",
    "spec_draft_tokens",
    "spec_accepted",
    "prefix_cache_hits",
    "prefix_cache_queries",
)
# Every live rate key is always present; None means "no reading this window".
LIVE_RATE_KEYS = (
    "decode_tok_per_s",
    "throughput_tok_per_s",
    "prefill_tok_per_s",
    "ttft_s",
    "spec_accept_rate",
    "spec_tokens_per_step",
    "prefix_cache_hit_rate_live",
    "rate_window_s",
)

# Single writer (the sampler's fast tick): previous counters, the burst in progress,
# and the last finished burst.
_LIVE_RATE: dict[str, Any] = {"t": None, "prev": None, "burst": None, "last_burst": None, "last_prefill": None}


def reset_live_rate_state() -> None:
    """Forget counters and bursts — the endpoint or serving container changed."""
    _LIVE_RATE.update(t=None, prev=None, burst=None, last_burst=None, last_prefill=None)


def _ratio(num: float | None, den: float | None, digits: int = 2) -> float | None:
    if num is None or den is None or den <= 0:
        return None
    return round(num / den, digits)


def live_token_rates(metrics: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
    """Live rates from counter deltas over the window since the previous scrape.

    decode_tok_per_s      per-stream decode speed over BUSY time: tokens after each
                          request's first ÷ Σ inter-token latency. A short burst inside
                          a long window is not diluted. With spec-decode one engine
                          step emits several tokens; this is tokens per second of step
                          time, i.e. what one stream sees. 0 while requests run but no
                          token moved (prefill, stall); None when no request is running at
                          the scrape — the window's last tokens are in last_burst instead.
    throughput_tok_per_s  aggregate Δgeneration_tokens / Δwall — all streams together.
    prefill_tok_per_s     computed prompt tokens ÷ prefill time of requests that FINISHED
                          in the window (vLLM records it at finish; cache hits excluded).
    ttft_s                mean time to first token of requests that got one this window.
    spec_accept_rate      accepted ÷ drafted tokens; spec_tokens_per_step = 1 + accepted ÷ drafts.
    last_prefill          the latest non-null prefill_tok_per_s with its time (`at`, epoch ms).
    last_burst            the previous busy period: per-stream decode rate and tokens,
                          `ended_at` epoch ms. Closed at the first scrape with no request
                          running, so ended_at is at most one window after the last token.
                          Both "last" values are labelled, never live.

    Never a lifetime average; a counter that goes backwards (engine restart) re-baselines.
    """
    out = dict(metrics)
    now = time.monotonic() if now is None else now
    for key in LIVE_RATE_KEYS:
        out[key] = None
    out["spec_accept_rate_lifetime"] = _ratio(metrics.get("spec_accepted"), metrics.get("spec_draft_tokens"), 4)

    cur = {k: metrics.get(k) for k in _RATE_COUNTERS}
    prev, prev_t = _LIVE_RATE["prev"], _LIVE_RATE["t"]
    _LIVE_RATE.update(t=now, prev=cur)
    out["last_burst"] = _LIVE_RATE["last_burst"]
    out["last_prefill"] = _LIVE_RATE["last_prefill"]
    if prev is None or prev_t is None or now - prev_t < 0.2:
        return out
    d = {k: cur[k] - prev[k] for k in _RATE_COUNTERS if cur[k] is not None and prev.get(k) is not None}
    if any(v < 0 for v in d.values()):
        # Counters went backwards: the engine restarted. Re-baseline, show nothing stale.
        _LIVE_RATE.update(burst=None, last_burst=None, last_prefill=None)
        out["last_burst"] = out["last_prefill"] = None
        return out

    dt = now - prev_t
    running = (metrics.get("requests_running") or 0) > 0
    gen = d.get("generation_tokens_total")
    first = d.get("ttft_count") or 0.0
    itl = d.get("itl_sum") or 0.0
    out["rate_window_s"] = round(dt, 2)
    if gen is not None:
        out["throughput_tok_per_s"] = round(gen / dt, 2)
    if running and gen is not None:
        out["decode_tok_per_s"] = round(max(0.0, gen - first) / itl, 2) if itl > 0 else 0.0
    if d.get("prefill_time_s_count"):
        out["prefill_tok_per_s"] = _ratio(d.get("prefill_tokens_sum"), d.get("prefill_time_s_sum"))
        if out["prefill_tok_per_s"] is not None:
            _LIVE_RATE["last_prefill"] = out["last_prefill"] = {
                "tok_per_s": out["prefill_tok_per_s"],
                "at": int(time.time() * 1000),
            }
    if first:
        out["ttft_s"] = _ratio(d.get("ttft_sum"), first, 3)
    out["spec_accept_rate"] = _ratio(d.get("spec_accepted"), d.get("spec_draft_tokens"), 4)
    if d.get("spec_drafts") and d.get("spec_accepted") is not None:
        out["spec_tokens_per_step"] = round(1 + d["spec_accepted"] / d["spec_drafts"], 2)
    out["prefix_cache_hit_rate_live"] = _ratio(d.get("prefix_cache_hits"), d.get("prefix_cache_queries"), 4)

    burst = _LIVE_RATE["burst"]
    if running or gen or d.get("prefill_time_s_count"):
        burst = burst or {"tokens": 0.0, "decode_tokens": 0.0, "itl": 0.0}
        burst["tokens"] += gen or 0.0
        burst["decode_tokens"] += max(0.0, (gen or 0.0) - first) if itl > 0 else 0.0
        burst["itl"] += itl
        _LIVE_RATE["burst"] = burst
    if burst and not running:
        # Nothing runs at this scrape: the burst ended inside this window (its last tokens,
        # if any, were just added). Keep it, labelled as such — never shown as live.
        _LIVE_RATE["last_burst"] = out["last_burst"] = {
            "decode_tok_per_s": _ratio(burst["decode_tokens"], burst["itl"]),
            "tokens": int(burst["tokens"]),
            "ended_at": int(time.time() * 1000),
        }
        _LIVE_RATE["burst"] = None
    return out


def model_preset(model_id: str) -> dict[str, Any]:
    if model_id in MODEL_PRESETS:
        return MODEL_PRESETS[model_id]
    # heuristic
    info: dict[str, Any] = {
        "architecture": "unknown",
        "param_count": None,
        "active_moe_params": None,
        "weights": {
            "dtype": "unknown",
            "quant_format": "unknown",
            "group_size": None,
            "calibration": "unknown",
        },
    }
    mid = model_id.lower()
    if "nvfp4" in mid or "fp4" in mid:
        info["weights"]["dtype"] = "nvfp4"
        info["weights"]["quant_format"] = "compressed-tensors"
    elif "fp8" in mid:
        info["weights"]["dtype"] = "fp8"
    elif "gguf" in mid or "q4" in mid:
        info["weights"]["dtype"] = "gguf"
    m = re.search(r"(\d+)[Bb]", model_id)
    if m:
        info["param_count"] = f"{m.group(1)}B"
    if "a3b" in mid or "A3B" in model_id:
        info["active_moe_params"] = "3B"
        info["architecture"] = "MoE"
    return info


def build_envelope(
    *,
    run_id: str | None = None,
    intent: str = "attach",
    model_id: str | None = None,
    kind: str = "status",
    workload: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    agentic: dict[str, Any] | None = None,
    engine_extra: dict[str, Any] | None = None,
    probe: dict[str, Any] | None = None,
    flags: list[str] | None = None,
) -> dict[str, Any]:
    rid = run_id or make_run_id()
    preset = model_preset(model_id or "unknown")
    models = (probe or {}).get("models") or []
    if not model_id and models:
        model_id = models[0].get("id")
        preset = model_preset(model_id)
    max_len = None
    if models:
        max_len = models[0].get("max_model_len")
    version = (probe or {}).get("version")
    eng_ver = None
    eng_commit = None
    if isinstance(version, dict):
        eng_ver = version.get("version") or str(version)
        eng_commit = version.get("commit") or version.get("git_tag")
    elif isinstance(version, str):
        eng_ver = version

    containers = list_vllm_containers()
    running = [c for c in containers if "Up" in c.get("status", "")]
    image = running[0]["image"] if running else None
    cmd_flags: list[str] = list(flags or [])
    if running and not cmd_flags:
        insp = docker_inspect_flags(running[0]["name"])
        cmd_flags = [str(x) for x in (insp.get("cmd") or [])]
        image = insp.get("image") or image

    return {
        "schema_version": 1,
        "run_id": rid,
        "created_at": utc_now(),
        "kind": kind,
        "intent": intent,
        "model": {
            "id": model_id,
            "architecture": preset.get("architecture"),
            "param_count": preset.get("param_count"),
            "active_moe_params": preset.get("active_moe_params"),
            "max_model_len": max_len,
        },
        "weights": preset.get("weights") or {},
        "engine": {
            "name": "vllm",
            "version": eng_ver,
            "commit": eng_commit,
            "image": image,
            "backend": _flag_value(cmd_flags, "--attention-backend") or "unknown",
            "flags": cmd_flags,
            "system_fingerprint": None,
            "metrics_snapshot": (probe or {}).get("metrics") or {},
            **(engine_extra or {}),
        },
        "hardware": collect_hardware(),
        "workload": workload or {},
        "metrics": metrics or {},
        "agentic": agentic or {},
        "endpoint": {
            "base_url": (probe or {}).get("base_url") or DEFAULT_BASE_URL,
            "healthy": (probe or {}).get("healthy"),
            "models": [{"id": m.get("id"), "max_model_len": m.get("max_model_len")} for m in models],
        },
        "containers": containers,
    }


def _flag_value(flags: list[str], name: str) -> str | None:
    for i, f in enumerate(flags):
        if f == name and i + 1 < len(flags):
            return flags[i + 1]
        if f.startswith(f"{name}="):
            return f.split("=", 1)[1]
    return None
