"""Decode bench: structured / prose / code / JSON, sequential concurrency waves."""
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
    last_s: float | None = None


def output_piece(delta: dict[str, Any]) -> str:
    """Tokens the model actually emitted in this chunk, including reasoning."""
    return str(
        delta.get("content")
        or delta.get("reasoning")
        or delta.get("reasoning_content")
        or ""
    )


def decode_time_s(r: ReqResult) -> float | None:
    """Time from first emitted token to last. Trailing usage frames are not decode."""
    if r.ttft_s is None:
        return None
    end = r.last_s
    if end is None or end <= r.ttft_s:
        end = r.wall_s
    ds = end - r.ttft_s
    if ds <= 0:
        return None
    return ds


def decode_tok_per_s(r: ReqResult) -> float | None:
    ds = decode_time_s(r)
    n = r.completion_tokens
    if ds is None or not n or n <= 0:
        return None
    return n / ds


def prefill_tok_per_s(r: ReqResult) -> float | None:
    n = r.prompt_tokens
    if not n or n <= 0 or not r.ttft_s or r.ttft_s <= 0:
        return None
    return n / r.ttft_s


def completion_body(
    *,
    model: str,
    user_content: str,
    max_tokens: int,
    thinking: bool = False,
) -> dict[str, Any]:
    """Decode-bench chat body. Force `max_tokens` output so EOS cannot end the wave early."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }


def stream_one(
    base: str,
    model: str,
    user_content: str,
    max_tokens: int,
    label: str,
    thinking: bool = False,
) -> ReqResult:
    body = completion_body(
        model=model,
        user_content=user_content,
        max_tokens=max_tokens,
        thinking=thinking,
    )
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft: float | None = None
    last_s: float | None = None
    usage: dict[str, Any] | None = None
    pieces: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                now = time.perf_counter() - t0
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
                piece = output_piece(choices[0].get("delta") or {})
                if piece:
                    if ttft is None:
                        ttft = now
                    last_s = now
                    pieces.append(piece)
        wall = time.perf_counter() - t0
        prompt_n = (usage or {}).get("prompt_tokens")
        comp_n = (usage or {}).get("completion_tokens")
        if usage is None or comp_n is None:
            return ReqResult(
                False, wall, ttft, prompt_n, comp_n, label, error="no_usage", last_s=last_s
            )
        text = "".join(pieces)
        if text and set(text.strip()) <= {"!", "?", ".", " "}:
            return ReqResult(
                False,
                wall,
                ttft,
                prompt_n,
                comp_n,
                label,
                error="garbage_output",
                snippet=text[:80],
                last_s=last_s,
            )
        return ReqResult(
            ok=True,
            wall_s=wall,
            ttft_s=ttft,
            prompt_tokens=prompt_n,
            completion_tokens=comp_n,
            label=label,
            snippet=text[:120].replace("\n", " "),
            last_s=last_s,
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
            "Repeat the following labeled fields over and over. Do not stop.\n\n"
            "model: Qwen3.6-27B-NVFP4\nquant: nvfp4\ncontext_len: 32768\ntp: 2\n"
            "prefill_tok_s: 410\ndecode_tok_s: 61\nkv_policy: prefix_cache\n"
            "headroom_gib: 18\nnotes: GB10 UMA, QSFP RoCE up\n"
        ),
        256,
    ),
    "prose": (
        "prose_essay",
        (
            "Continue this essay in the same voice. Do not stop.\n\n"
            "Decode throughput and time-to-first-token feel different when a coding "
            "agent shares a long system prompt across tabs on a DGX Spark with unified "
            "memory. The KV cache is the product, not a leftover after util. "
        ),
        256,
    ),
    "code": (
        "code_impl",
        (
            "Continue this Python module. No markdown fences. Do not stop.\n\n"
            "from __future__ import annotations\n\n"
            "class TokenBucket:\n"
            "    def __init__(self, rate: float, burst: int) -> None:\n"
            "        self.rate = rate\n"
            "        self.burst = burst\n"
            "        self.tokens = float(burst)\n"
        ),
        256,
    ),
    "json": (
        "json_object",
        (
            "Continue this JSON array. Valid JSON only. Do not stop.\n\n"
            '[{"spark_id":"spark1","serving":true,"temperature_c":47,'
            '"gpu_util_pct":12,"decode_tok_per_s":61.2},'
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
        rate = decode_tok_per_s(r)
        ds = decode_time_s(r)
        if rate is not None and ds is not None and r.completion_tokens:
            decode_rates.append(rate)
            tpot_ms.append((ds / r.completion_tokens) * 1000)
        pre = prefill_tok_per_s(r)
        if pre is not None:
            prefill_rates.append(pre)

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
            log.write(f"{kind} · {c}")
        arm = run_wave(base, model_id, c, kind=kind)
        if log:
            log.write(
                f"{kind} · {c} · {arm.get('decode_tok_per_s_median')} decode tok/s · "
                f"{arm.get('prefill_tok_per_s_median')} prefill tok/s"
            )
        return arm

    idx = {"n": 0}

    def run_level_tracked(c: int) -> dict[str, Any]:
        if progress:
            progress(idx["n"] / max(len(concs), 1), f"{kind} · {c}")
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
        kind="decode",
        workload={
            "type": kind,
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
        kind="decode",
        intent=intent,
        model_id=model_id,
        summary=metrics["headline"],
        path=str(out),
    )
    if progress:
        progress(1.0, "done")
    if log:
        log.write("done")
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
