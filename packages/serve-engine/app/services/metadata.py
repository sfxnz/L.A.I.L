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


def collect_hardware() -> dict[str, Any]:
    gpu = "unknown"
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if smi.strip():
        gpu = smi.strip().split("\n")[0].strip()
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
        "memory_capacity_gib": mem_total,
        "bandwidth": "UMA" if "GB10" in gpu or "Spark" in gpu else "unknown",
        "interconnect": "n/a",
        "cpu": cpu,
        "ram_gib": mem_total,
        "available_gib": available_gib(),
        "free_h": free_h(),
        "hostname": platform.node(),
        "platform": platform.platform(),
    }


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
        if not (
            re.search(r"vllm|spark-vllm|qwen|brain|nemotron|deepseek|llama", name, re.I)
            or "vllm" in image.lower()
        ):
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
    return {
        "image": cfg.get("Image"),
        "cmd": cfg.get("Cmd") or [],
        "env": [e for e in (cfg.get("Env") or []) if not e.startswith("HF_TOKEN") and "TOKEN" not in e],
        "state": (data.get("State") or {}).get("Status"),
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
