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


def jobs_for_workflow() -> list[tuple[str, str, int]]:
    short = "Reply in one sentence: what is NVFP4 on Blackwell?"
    med = (
        "You are a local AI lab assistant. Explain how continuous batching changes "
        "per-user vs aggregate tokens/s on a single DGX Spark. Be concrete. "
        + ("Include practical caveats for unified memory. " * 6)
    )
    long_notes = (
        "Summarize these agent lab notes into 6 bullets, then one recommended next experiment.\n\n"
        + (
            "Session log: user runs vLLM with MTP speculative decoding on Qwen3.6-27B NVFP4. "
            "Target: multi-turn coding agent with long system + tools + retrieved files. "
            "Measure TTFT under concurrent chat tabs, decode under sustained generation, "
            "and whether prefix caching helps when the system prompt is shared. "
            "Context budget is the product decision — do not starve KV for vanity util. "
        )
        * 40
    )
    return [
        ("short_chat", short, 128),
        ("med_explain", med, 384),
        ("long_prefill_agent", long_notes, 256),
        ("short_chat", short, 128),
        ("med_explain", med, 384),
        ("long_prefill_agent", long_notes, 256),
    ]


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


def run_wave(base: str, model: str, concurrency: int) -> dict[str, Any]:
    jobs = jobs_for_workflow()
    work: list[tuple[str, str, int]] = []
    while len(work) < max(concurrency * 2, len(jobs)):
        work.extend(jobs)
    work = work[: max(concurrency * 2, 6)]

    t_batch0 = time.perf_counter()
    results: list[ReqResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [
            ex.submit(stream_one, base, model, prompt, mt, label)
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
    intent: str = "attach",
    dollars_per_hour: float = 0.5,
    log: Any = None,
    progress: Callable | None = None,
) -> dict[str, Any]:
    base = base_url.rstrip("/")
    model_id = resolve_model(base, model)
    concs = concurrencies or [1, 2, 4]
    arms = []
    for i, c in enumerate(concs):
        if log:
            log.write(f"=== concurrency {c} ===")
        if progress:
            progress((i) / max(len(concs), 1), f"concurrency={c}")
        arm = run_wave(base, model_id, c)
        arms.append({k: v for k, v in arm.items() if k != "per_request"})
        arms[-1]["per_request"] = arm.get("per_request")
        if log:
            slim = {k: arm[k] for k in arm if k != "per_request"}
            log.write(json.dumps(slim, indent=2))

    # headline metrics from c=1 arm
    c1 = next((a for a in arms if a["concurrency"] == 1), arms[0] if arms else {})
    decode = c1.get("decode_tok_per_s_median")
    cost = cost_per_1m_tokens(decode, dollars_per_hour)

    run_id = make_run_id()
    metrics = {
        "arms": [{k: v for k, v in a.items() if k != "per_request"} for a in arms],
        "headline": {
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
            "type": "workflow_mixed",
            "prompts": ["short_chat", "med_explain", "long_prefill_agent"],
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
