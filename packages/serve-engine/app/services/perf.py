"""Performance benchmarks — workflow concurrency + optional external scripts."""
from __future__ import annotations

import concurrent.futures
import json
import statistics
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import BENCH_CONCURRENCY, BENCH_PREFILL, BENCH_WORKFLOW, DEFAULT_BASE_URL, RUNS_DIR
from .. import db
from .metadata import (
    build_envelope,
    cost_per_1m_tokens,
    make_run_id,
    probe_endpoint,
    utc_now,
)
import asyncio


def pct(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


@dataclass
class ReqResult:
    ok: bool
    wall_s: float
    ttft_s: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    label: str
    error: str | None = None
    snippet: str = ""


def stream_one(
    base: str,
    model: str,
    user_content: str,
    max_tokens: int,
    label: str,
    thinking: bool = False,
) -> ReqResult:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft: float | None = None
    usage: dict[str, Any] | None = None
    pieces: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    pieces.append(piece)
        wall = time.perf_counter() - t0
        text = "".join(pieces)
        if text and set(text.strip()) <= {"!", "?", ".", " "}:
            return ReqResult(
                False,
                wall,
                ttft,
                (usage or {}).get("prompt_tokens"),
                (usage or {}).get("completion_tokens"),
                label,
                error="garbage_output",
                snippet=text[:80],
            )
        return ReqResult(
            ok=True,
            wall_s=wall,
            ttft_s=ttft,
            prompt_tokens=(usage or {}).get("prompt_tokens"),
            completion_tokens=(usage or {}).get("completion_tokens"),
            label=label,
            snippet=text[:120].replace("\n", " "),
        )
    except Exception as e:
        return ReqResult(
            False, time.perf_counter() - t0, None, None, None, label, error=str(e)
        )


WORKLOAD_KINDS = ("structured", "prose", "code", "json")

_WORKLOAD_FAMILY: dict[str, tuple[str, str, int]] = {
    "structured": (
        "structured_fields",
        (
            "Fill every field. Use the labels exactly. No preamble.\n"
            "model: \nquant: \ncontext_len: \ntp: \nprefill_tok_s: \n"
            "decode_tok_s: \nkv_policy: \nheadroom_gib: \nnotes: \n"
            "Repeat the block three times with different plausible Spark-lab values."
        ),
        256,
    ),
    "prose": (
        "prose_essay",
        (
            "Write a flowing essay on how decode throughput and time-to-first-token "
            "feel different when a coding agent shares a long system prompt across "
            "tabs on a DGX Spark with unified memory. Be concrete about KV cache, "
            "continuous batching, and what the operator should watch. "
            + ("Stay in prose; no bullets, no headings, no lists. " * 8)
        ),
        384,
    ),
    "code": (
        "code_impl",
        (
            "Write a complete Python module (no markdown fences) that implements "
            "an in-memory token-bucket rate limiter with acquire(n), available(), "
            "and a background refill. Include type hints, a docstring on each "
            "public function, and a tiny self-check under if __name__ == '__main__'."
        ),
        384,
    ),
    "json": (
        "json_object",
        (
            "Return only a JSON object, no markdown, no commentary, with keys "
            "spark_id, hostname, serving, model_id, temperature_c, gpu_util_pct, "
            "power_w, decode_tok_per_s, prefill_tok_per_s, notes. Invent realistic "
            "values for two-node TP on GB10. Nested key 'peers' is an array of two "
            "objects with the same fields except peers."
        ),
        256,
    ),
}


def jobs_for_kind(kind: str) -> list[tuple[str, str, int]]:
    if kind not in _WORKLOAD_FAMILY:
        raise ValueError(f"unknown workload kind: {kind}")
    job = _WORKLOAD_FAMILY[kind]
    return [job]


def normalize_concurrencies(raw: list[int] | None) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for n in raw or [1]:
        if not isinstance(n, int) or isinstance(n, bool):
            continue
        if 1 <= n <= 32 and n not in seen:
            seen.add(n)
            out.append(n)
    out.sort()
    return out or [1]


def expand_wave_jobs(
    jobs: list[tuple[str, str, int]], concurrency: int
) -> list[tuple[str, str, int]]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if not jobs:
        raise ValueError("jobs must be non-empty")
    work: list[tuple[str, str, int]] = []
    i = 0
    while len(work) < concurrency:
        work.append(jobs[i % len(jobs)])
        i += 1
    return work


def run_concurrency_levels(
    concurrencies: list[int], run_level: Callable[[int], dict[str, Any]]
) -> list[dict[str, Any]]:
    """Run each concurrency in order. Each call is one wave."""
    return [run_level(c) for c in concurrencies]


def summarize(results: list[ReqResult], concurrency: int) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    walls = sorted(r.wall_s for r in ok)
    ttfts = sorted(r.ttft_s for r in ok if r.ttft_s is not None)
    decode_rates: list[float] = []
    prefill_rates: list[float] = []
    tpot_ms: list[float] = []
    for r in ok:
        if r.completion_tokens and r.ttft_s and r.wall_s > r.ttft_s:
            ds = r.wall_s - r.ttft_s
            if ds > 0 and r.completion_tokens > 0:
                decode_rates.append(r.completion_tokens / ds)
                tpot_ms.append((ds / r.completion_tokens) * 1000)
        if r.prompt_tokens and r.ttft_s and r.ttft_s > 0:
            prefill_rates.append(r.prompt_tokens / r.ttft_s)

    return {
        "concurrency": concurrency,
        "requests": len(results),
        "ok": len(ok),
        "errors": [{"label": r.label, "error": r.error} for r in results if not r.ok],
        "latency_s": {
            "p50": round(pct(walls, 50) or 0, 3),
            "p95": round(pct(walls, 95) or 0, 3),
            "p99": round(pct(walls, 99) or 0, 3),
            "mean": round(statistics.mean(walls), 3) if walls else None,
        },
        "ttft_s": {
            "p50": round(pct(ttfts, 50) or 0, 3) if ttfts else None,
            "p95": round(pct(ttfts, 95) or 0, 3) if ttfts else None,
            "p99": round(pct(ttfts, 99) or 0, 3) if ttfts else None,
        },
        "tpot_ms": {
            "p50": round(pct(sorted(tpot_ms), 50) or 0, 2) if tpot_ms else None,
            "p95": round(pct(sorted(tpot_ms), 95) or 0, 2) if tpot_ms else None,
            "mean": round(statistics.mean(tpot_ms), 2) if tpot_ms else None,
        },
        "prefill_tok_per_s_median": round(statistics.median(prefill_rates), 2)
        if prefill_rates
        else None,
        "decode_tok_per_s_median": round(statistics.median(decode_rates), 2)
        if decode_rates
        else None,
        "snippets": [r.snippet for r in ok[:3]],
        "per_request": [asdict(r) for r in results],
    }


def run_wave(
    base: str,
    model: str,
    concurrency: int,
    *,
    kind: str = "prose",
    stream_fn: Callable[..., ReqResult] | None = None,
) -> dict[str, Any]:
    jobs = jobs_for_kind(kind)
    work = expand_wave_jobs(jobs, concurrency)
    send = stream_fn or stream_one

    t_batch0 = time.perf_counter()
    results: list[ReqResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [
            ex.submit(send, base, model, prompt, mt, label)
            for label, prompt, mt in work
        ]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
    batch_s = time.perf_counter() - t_batch0
    summary = summarize(results, concurrency)
    ok = [r for r in results if r.ok]
    total_comp = sum(r.completion_tokens or 0 for r in ok)
    summary["batch_wall_s"] = round(batch_s, 3)
    summary["aggregate_tok_per_s"] = (
        round(total_comp / batch_s, 2) if batch_s > 0 else None
    )
    summary["request_per_s"] = round(len(ok) / batch_s, 4) if batch_s > 0 else None
    return summary


def resolve_model(base: str, model: str | None) -> str:
    if model:
        return model
    with urllib.request.urlopen(f"{base.rstrip('/')}/v1/models", timeout=30) as r:
        data = json.loads(r.read().decode())
    ids = [m["id"] for m in data.get("data", [])]
    if not ids:
        raise RuntimeError("No models served")
    return ids[0]


def run_workflow_bench(
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = None,
    concurrencies: list[int] | None = None,
    workload: str = "prose",
    intent: str = "attach",
    dollars_per_hour: float = 0.5,
    log: Any = None,
    progress: Callable | None = None,
) -> dict[str, Any]:
    base = base_url.rstrip("/")
    model_id = resolve_model(base, model)
    kind = workload if workload in WORKLOAD_KINDS else "prose"
    concs = normalize_concurrencies(concurrencies)

    def run_level(c: int) -> dict[str, Any]:
        if log:
            log.write(f"=== concurrency {c} · {kind} ===")
        arm = run_wave(base, model_id, c, kind=kind)
        if log:
            log.write(
                f"decode {arm.get('decode_tok_per_s_median')} tok/s · "
                f"prefill {arm.get('prefill_tok_per_s_median')} tok/s · "
                f"ok {arm.get('ok')}/{arm.get('requests')}"
            )
        return arm

    idx = {"n": 0}

    def run_level_tracked(c: int) -> dict[str, Any]:
        if progress:
            progress(idx["n"] / max(len(concs), 1), f"concurrency={c}")
        idx["n"] += 1
        return run_level(c)

    raw_arms = run_concurrency_levels(concs, run_level_tracked)

    arms = []
    for arm in raw_arms:
        slim = {k: v for k, v in arm.items() if k != "per_request"}
        slim["per_request"] = arm.get("per_request")
        arms.append(slim)

    # headline metrics from c=1 arm
    c1 = next((a for a in arms if a["concurrency"] == 1), arms[0] if arms else {})
    decode = c1.get("decode_tok_per_s_median")
    cost = cost_per_1m_tokens(decode, dollars_per_hour)

    run_id = make_run_id()
    metrics = {
        "arms": [{k: v for k, v in a.items() if k != "per_request"} for a in arms],
        "headline": {
            "workload": kind,
            "concurrencies": concs,
            "decode_tok_per_s_median_c1": decode,
            "ttft_p50_s_c1": (c1.get("ttft_s") or {}).get("p50"),
            "ttft_p95_s_c1": (c1.get("ttft_s") or {}).get("p95"),
            "latency_p95_s_c1": (c1.get("latency_s") or {}).get("p95"),
            "tpot_ms_p50_c1": (c1.get("tpot_ms") or {}).get("p50"),
            "prefill_tok_per_s_median_c1": c1.get("prefill_tok_per_s_median"),
            **cost,
        },
        "full_arms": arms,
    }

    # async probe via sync bridge
    probe = asyncio.run(probe_endpoint(base, timeout=10))
    envelope = build_envelope(
        run_id=run_id,
        intent=intent,
        model_id=model_id,
        kind="perf_workflow",
        workload={
            "type": f"decode_{kind}",
            "kind": kind,
            "prompts": [jobs_for_kind(kind)[0][0]],
            "concurrencies": concs,
            "streaming": True,
            "thinking": False,
            "client": "localhost_on_spark",
        },
        metrics=metrics,
        probe=probe,
    )

    out = RUNS_DIR / f"{run_id}.json"
    out.write_text(json.dumps(envelope, indent=2))
    db.insert_run(
        run_id=run_id,
        created_at=envelope["created_at"],
        kind="perf_workflow",
        intent=intent,
        model_id=model_id,
        summary=metrics["headline"],
        path=str(out),
    )
    if progress:
        progress(1.0, f"saved {out}")
    if log:
        log.write(f"wrote {out}")
    return envelope


def run_external_prefill(
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = None,
    intent: str = "attach",
    log: Any = None,
    progress: Callable | None = None,
) -> dict[str, Any]:
    base = base_url.rstrip("/")
    model_id = resolve_model(base, model)
    if not BENCH_PREFILL.exists():
        raise FileNotFoundError(str(BENCH_PREFILL))
    run_id = make_run_id()
    out = RUNS_DIR / f"{run_id}_prefill_decode.json"
    cmd = ["python3", str(BENCH_PREFILL), base, model_id, str(out)]
    if log:
        log.write(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if log:
        log.write(r.stdout or "")
        log.write(r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError(r.stderr or "prefill bench failed")
    raw = json.loads(out.read_text()) if out.exists() else {"stdout": r.stdout}
    probe = asyncio.run(probe_endpoint(base, timeout=10))
    envelope = build_envelope(
        run_id=run_id,
        intent=intent,
        model_id=model_id,
        kind="perf_prefill_decode",
        workload={"type": "prefill_decode"},
        metrics=raw,
        probe=probe,
    )
    path = RUNS_DIR / f"{run_id}.json"
    path.write_text(json.dumps(envelope, indent=2))
    db.insert_run(
        run_id=run_id,
        created_at=envelope["created_at"],
        kind="perf_prefill_decode",
        intent=intent,
        model_id=model_id,
        summary=raw if isinstance(raw, dict) else {},
        path=str(path),
    )
    if progress:
        progress(1.0, "done")
    return envelope


def run_external_concurrency(
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = None,
    concurrency: int = 4,
    intent: str = "attach",
    log: Any = None,
    progress: Callable | None = None,
) -> dict[str, Any]:
    base = base_url.rstrip("/")
    model_id = resolve_model(base, model)
    if not BENCH_CONCURRENCY.exists():
        raise FileNotFoundError(str(BENCH_CONCURRENCY))
    run_id = make_run_id()
    out = RUNS_DIR / f"{run_id}_concurrency.json"
    cmd = [
        "python3",
        str(BENCH_CONCURRENCY),
        "--base",
        base,
        "--model",
        model_id,
        "--arm",
        "before",
        "--out",
        str(out),
        "--concurrency",
        str(concurrency),
    ]
    if log:
        log.write(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if log:
        log.write(r.stdout or "")
        log.write(r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError(r.stderr or "concurrency bench failed")
    raw = json.loads(out.read_text()) if out.exists() else {}
    probe = asyncio.run(probe_endpoint(base, timeout=10))
    envelope = build_envelope(
        run_id=run_id,
        intent=intent,
        model_id=model_id,
        kind="perf_concurrency",
        workload={"type": "concurrency_tiers", "concurrency": concurrency},
        metrics=raw,
        probe=probe,
    )
    path = RUNS_DIR / f"{run_id}.json"
    path.write_text(json.dumps(envelope, indent=2))
    # headline from tiers
    tiers = raw.get("tiers") or []
    headline = tiers[0] if tiers else {}
    db.insert_run(
        run_id=run_id,
        created_at=envelope["created_at"],
        kind="perf_concurrency",
        intent=intent,
        model_id=model_id,
        summary=headline,
        path=str(path),
    )
    if progress:
        progress(1.0, "done")
    return envelope
