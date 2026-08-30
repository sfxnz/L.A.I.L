"""Collect model / engine / hardware / metrics metadata for Run Envelopes."""
from __future__ import annotations

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
    out = _run(["free", "-g"])
    for line in out.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            if len(parts) >= 7:
                try:
                    return float(parts[6])
                except ValueError:
                    return None
    return None


def free_h() -> str:
    return _run(["free", "-h"]).strip()


GPU_SMI_QUERY = (
    "name,temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total"
)
_GPU_NA = {"[n/a]", "n/a", "na", ""}
GPU_TELEMETRY_FIELDS = (
    "temperature_c",
    "gpu_util_pct",
    "power_w",
    "memory_used_mib",
    "memory_total_mib",
)


def _smi_num(raw: str) -> float | None:
    s = (raw or "").strip().lower()
    if s in _GPU_NA:
        return None
    for suffix in (" mib", "°c", "w", "%", "c"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
            break
    try:
        return float(s)
    except ValueError:
        return None


def parse_gpu_telemetry(csv_text: str) -> dict[str, Any]:
    """Parse one nvidia-smi csv line. Missing and N/A fields stay None, never 0."""
    empty: dict[str, Any] = {
        "gpu_sku": None,
        "temperature_c": None,
        "gpu_util_pct": None,
        "power_w": None,
        "memory_used_mib": None,
        "memory_total_mib": None,
    }
    lines = (csv_text or "").strip().splitlines()
    if not lines:
        return empty
    parts = [p.strip() for p in lines[0].split(",")]
    sku = parts[0] or None
    if len(parts) >= 6:
        return {
            "gpu_sku": sku,
            "temperature_c": _smi_num(parts[1]),
            "gpu_util_pct": _smi_num(parts[2]),
            "power_w": _smi_num(parts[3]),
            "memory_used_mib": _smi_num(parts[4]),
            "memory_total_mib": _smi_num(parts[5]),
        }
    if len(parts) == 2:
        empty["gpu_sku"] = sku
        empty["memory_total_mib"] = _smi_num(parts[1])
        return empty
    empty["gpu_sku"] = sku
    return empty


def collect_gpu_telemetry() -> dict[str, Any]:
    raw = _run(
        [
            "nvidia-smi",
            f"--query-gpu={GPU_SMI_QUERY}",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_gpu_telemetry(raw)


def collect_hardware() -> dict[str, Any]:
    tel = collect_gpu_telemetry()
    gpu = tel.get("gpu_sku") or "unknown"
    cpu = platform.processor() or platform.machine()
    # better CPU model on Linux
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    mem_total = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                mem_total = round(kb / 1024 / 1024, 1)
                break
    except Exception:
        pass
    return {
        "gpu_sku": gpu,
        "temperature_c": tel.get("temperature_c"),
        "gpu_util_pct": tel.get("gpu_util_pct"),
        "power_w": tel.get("power_w"),
        "memory_used_mib": tel.get("memory_used_mib"),
        "memory_total_mib": tel.get("memory_total_mib"),
        "memory_capacity_gib": mem_total,
        "bandwidth": "UMA" if "GB10" in str(gpu) or "Spark" in str(gpu) else "unknown",
        "interconnect": "n/a",
        "cpu": cpu,
        "ram_gib": mem_total,
        "available_gib": available_gib(),
        "free_h": free_h(),
        "hostname": platform.node(),
        "platform": platform.platform(),
    }


_SERVE_CONTAINER_RE = re.compile(
    r"vllm|spark-vllm|qwen|brain|nemotron|deepseek|llama|dspark|glm",
    re.I,
)


def is_serve_container(name: str, image: str = "") -> bool:
    """True for a lab vLLM/llama.cpp-style serve container, including GLM image names."""
    blob = f"{name} {image}"
    if _SERVE_CONTAINER_RE.search(blob):
        return True
    img = image.lower()
    return "vllm" in img or "dspark" in img or "ray" in img


def list_vllm_containers() -> list[dict[str, Any]]:
    out = _run(
        [
            "docker",
            "ps",
            "-a",
            "--format",
            "{{.Names}}\t{{.Status}}\t{{.Image}}\t{{.ID}}",
        ]
    )
    containers = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name, status, image = parts[0], parts[1], parts[2]
        if not is_serve_container(name, image):
            continue
        containers.append(
            {
                "name": name,
                "status": status,
                "image": image,
                "id": parts[3] if len(parts) > 3 else "",
            }
        )
    return containers


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

    def _host_ports(obj: Any) -> list[int]:
        out: list[int] = []
        if not isinstance(obj, dict):
            return out
        for _k, bindings in obj.items():
            if not bindings:
                continue
            for b in bindings:
                if not isinstance(b, dict):
                    continue
                hp = b.get("HostPort")
                if hp is None:
                    continue
                try:
                    port = int(hp)
                except (TypeError, ValueError):
                    continue
                if 1 <= port <= 65535:
                    out.append(port)
        return out

    ports = sorted({*_host_ports(ns.get("Ports") or {}), *_host_ports(hc.get("PortBindings") or {})})
    return {
        "image": cfg.get("Image"),
        "cmd": cfg.get("Cmd") or [],
        "args": data.get("Args") or [],
        "env": [e for e in (cfg.get("Env") or []) if not e.startswith("HF_TOKEN") and "TOKEN" not in e],
        "state": (data.get("State") or {}).get("Status"),
        "ports": ports,
        "network_mode": hc.get("NetworkMode"),
    }


async def probe_endpoint(base_url: str = DEFAULT_BASE_URL, timeout: float = 5.0) -> dict[str, Any]:
    base = base_url.rstrip("/")
    result: dict[str, Any] = {
        "base_url": base,
        "healthy": False,
        "models": [],
        "version": None,
        "metrics": {},
        "error": None,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            h = await client.get(f"{base}/health")
            result["healthy"] = h.status_code == 200
        except Exception as e:
            result["error"] = f"health: {e}"
        try:
            m = await client.get(f"{base}/v1/models")
            if m.status_code == 200:
                body = m.json()
                result["models"] = body.get("data") or []
                result["healthy"] = True
        except Exception as e:
            result["error"] = (result.get("error") or "") + f" models: {e}"
        try:
            v = await client.get(f"{base}/version")
            if v.status_code == 200:
                result["version"] = v.json() if "application/json" in v.headers.get("content-type", "") else v.text
        except Exception:
            pass
        try:
            met = await client.get(f"{base}/metrics")
            if met.status_code == 200:
                result["metrics"] = parse_prometheus(met.text)
        except Exception:
            pass
    return result


def parse_prometheus(text: str) -> dict[str, float]:
    """Extract a few useful vLLM gauges/counters."""
    keys = {
        "vllm:gpu_cache_usage_perc": "gpu_kv_cache_usage",
        "vllm:prefix_cache_hits": "prefix_cache_hits",
        "vllm:prefix_cache_queries": "prefix_cache_queries",
        "vllm:num_requests_running": "requests_running",
        "vllm:num_requests_waiting": "requests_waiting",
        "vllm:prompt_tokens_total": "prompt_tokens_total",
        "vllm:generation_tokens_total": "generation_tokens_total",
        "vllm:avg_prompt_throughput_toks_per_s": "prompt_tok_per_s",
        "vllm:avg_generation_throughput_toks_per_s": "gen_tok_per_s",
        "vllm:kv_cache_usage_perc": "gpu_kv_cache_usage",
        "vllm:request_decode_time_seconds_sum": "decode_time_s_sum",
        "vllm:request_generation_tokens_sum": "generation_tokens_sum",
        "vllm:request_prefill_time_seconds_sum": "prefill_time_s_sum",
        "vllm:request_prefill_kv_computed_tokens_sum": "prefill_tokens_sum",
    }
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # metric{labels} value
        m = re.match(r"^([a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)", line)
        if not m:
            continue
        name, val = m.group(1), float(m.group(2))
        if name in keys:
            # prefer unlabeled or sum — last wins for gauges
            out[keys[name]] = val
    if "prefix_cache_hits" in out and "prefix_cache_queries" in out and out["prefix_cache_queries"] > 0:
        out["prefix_cache_hit_rate"] = out["prefix_cache_hits"] / out["prefix_cache_queries"]
    return live_token_rates(out)


_LIVE_RATE: dict[str, float | None] = {
    "t": None,
    "prompt": None,
    "gen": None,
    "last_gen_rate": None,
    "last_prompt_rate": None,
}


def reset_live_rate_state() -> None:
    _LIVE_RATE.update(
        t=None, prompt=None, gen=None, last_gen_rate=None, last_prompt_rate=None
    )


def _positive_rate(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return round(float(value), 2)


def _counter_delta(now_val: float | None, prev_val: float | None, dt: float) -> float | None:
    if now_val is None or prev_val is None or dt < 0.2:
        return None
    return (float(now_val) - float(prev_val)) / dt


def live_token_rates(metrics: dict[str, float], *, now: float | None = None) -> dict[str, float]:
    """Fill gen_tok_per_s / prompt_tok_per_s from gauges or counter deltas.

    Newer vLLM drops avg_*_throughput gauges. `generation_tokens_total` and
    `prompt_tokens_total` still move. Prefill is a burst at request start, so
    the last positive prefill rate sticks while decode tokens (or running
    requests) show the serve is still in use. Zero stays absent, never a fake 0.
    """
    out = dict(metrics)
    now = time.monotonic() if now is None else now
    prompt = out.get("prompt_tokens_total")
    gen = out.get("generation_tokens_total")
    prev_t = _LIVE_RATE.get("t")
    d_gen = None
    d_prompt = None
    if prev_t is not None:
        dt = now - float(prev_t)
        d_gen = _counter_delta(gen, _LIVE_RATE.get("gen"), dt)
        d_prompt = _counter_delta(prompt, _LIVE_RATE.get("prompt"), dt)

    if d_gen is not None and not out.get("gen_tok_per_s"):
        out["gen_tok_per_s"] = d_gen
    if d_prompt is not None and not out.get("prompt_tok_per_s"):
        out["prompt_tok_per_s"] = d_prompt

    running = (out.get("requests_running") or 0) > 0
    if not out.get("gen_tok_per_s") and running:
        decode_s = out.get("decode_time_s_sum")
        gen_sum = out.get("generation_tokens_sum")
        if decode_s and gen_sum and decode_s > 0:
            out["gen_tok_per_s"] = gen_sum / decode_s
    if not out.get("prompt_tok_per_s") and running:
        prefill_s = out.get("prefill_time_s_sum")
        prefill_tok = out.get("prefill_tokens_sum")
        if prefill_s and prefill_tok and prefill_s > 0:
            out["prompt_tok_per_s"] = prefill_tok / prefill_s

    gen_rate = _positive_rate(out.get("gen_tok_per_s"))
    prompt_rate = _positive_rate(out.get("prompt_tok_per_s"))
    in_flight = (
        running
        or (d_gen is not None and d_gen > 0)
        or (d_prompt is not None and d_prompt > 0)
    )
    if prompt_rate is None and in_flight:
        prompt_rate = _positive_rate(_LIVE_RATE.get("last_prompt_rate"))
    if gen_rate is None and in_flight:
        gen_rate = _positive_rate(_LIVE_RATE.get("last_gen_rate"))

    out["gen_tok_per_s"] = gen_rate
    out["prompt_tok_per_s"] = prompt_rate
    _LIVE_RATE.update(
        t=now,
        prompt=prompt,
        gen=gen,
        last_gen_rate=gen_rate if gen_rate is not None else (
            _LIVE_RATE.get("last_gen_rate") if in_flight else None
        ),
        last_prompt_rate=prompt_rate if prompt_rate is not None else (
            _LIVE_RATE.get("last_prompt_rate") if in_flight else None
        ),
    )
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


def cost_per_1m_tokens(
    decode_tok_per_s: float | None,
    dollars_per_hour: float,
) -> dict[str, float | None]:
    if not decode_tok_per_s or decode_tok_per_s <= 0:
        return {"cost_per_1m_output_tokens": None, "tokens_per_hour": None}
    tph = decode_tok_per_s * 3600
    cost = (dollars_per_hour / tph) * 1_000_000 if tph else None
    return {
        "cost_per_1m_output_tokens": round(cost, 4) if cost is not None else None,
        "tokens_per_hour": round(tph, 1),
        "assumed_dollars_per_hour": dollars_per_hour,
    }
