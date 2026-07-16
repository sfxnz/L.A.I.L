"""REST + SSE API for Local AI Lab."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field  # Field used by ServeRequest
from sse_starlette.sse import EventSourceResponse

from .. import db
from ..config import DEFAULT_BASE_URL, MODEL_PRESETS, RUNS_DIR, SERVE_EXAMPLES
from ..services import agentic, autoconfig, jobs, metadata, perf, serve

router = APIRouter()


# ─── Status / dashboard ───────────────────────────────────────────────────────


@router.get("/status")
async def status(base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    probe = await metadata.probe_endpoint(base_url)
    hw = metadata.collect_hardware()
    containers = metadata.list_vllm_containers()
    avail = hw.get("available_gib")
    headroom = "ok"
    if avail is not None:
        if avail < 15:
            headroom = "critical"
        elif avail < 60:
            headroom = "tight"
    model_id = None
    if probe.get("models"):
        model_id = probe["models"][0].get("id")
    return {
        "healthy": probe.get("healthy"),
        "base_url": base_url,
        "model_id": model_id,
        "models": probe.get("models"),
        "version": probe.get("version"),
        "metrics": probe.get("metrics"),
        "hardware": hw,
        "containers": containers,
        "headroom": headroom,
        "error": probe.get("error"),
        "presets": list(MODEL_PRESETS.keys()),
        "serve_examples": SERVE_EXAMPLES,
        "tool_eval": agentic.tool_eval_available(),
    }


@router.get("/hardware")
async def hardware() -> dict[str, Any]:
    return metadata.collect_hardware()


# ─── Serve ────────────────────────────────────────────────────────────────────


class ServeRequest(BaseModel):
    """Fully explicit serve config — nothing is injected server-side beyond mode envelope."""

    model: str
    mode: Literal["lab_safe", "workflow_max"] = "lab_safe"
    util: Optional[float] = None
    max_model_len: Optional[int] = None
    port: int = 8000
    image: Optional[str] = None
    docker_env: list[str] = Field(default_factory=list)
    quantization: str = ""
    kv_cache_dtype: str = ""
    moe_backend: str = ""
    trust_remote_code: bool = False
    enable_auto_tool_choice: bool = False
    tool_call_parser: str = ""
    reasoning_parser: str = ""
    max_num_seqs: Optional[int] = None
    mtp: bool = False
    mtp_num_tokens: int = 2
    load_format: str = ""
    enable_chunked_prefill: bool = False
    enable_prefix_caching: bool = False
    extra_flags: str = ""
    stop_first: bool = True
    download: bool = False


@router.post("/serve/start")
async def serve_start(body: ServeRequest) -> dict[str, str]:
    def work(log, progress, **kw):
        if body.download:
            progress(0.02, "downloading…")
            serve.download_model(model=body.model, log=log, progress=progress)
        return serve.serve_model(
            model=body.model,
            mode=body.mode,
            util=body.util,
            max_model_len=body.max_model_len,
            port=body.port,
            image=body.image,
            docker_env=body.docker_env,
            quantization=body.quantization,
            kv_cache_dtype=body.kv_cache_dtype,
            moe_backend=body.moe_backend,
            trust_remote_code=body.trust_remote_code,
            enable_auto_tool_choice=body.enable_auto_tool_choice,
            tool_call_parser=body.tool_call_parser,
            reasoning_parser=body.reasoning_parser,
            max_num_seqs=body.max_num_seqs,
            mtp=body.mtp,
            mtp_num_tokens=body.mtp_num_tokens,
            load_format=body.load_format,
            enable_chunked_prefill=body.enable_chunked_prefill,
            enable_prefix_caching=body.enable_prefix_caching,
            extra_flags=body.extra_flags,
            stop_first=body.stop_first,
            log=log,
            progress=progress,
        )

    job_id = await jobs.start_job("serve", work)
    return {"job_id": job_id}


@router.get("/serve/examples")
async def serve_examples() -> dict[str, Any]:
    return {"examples": SERVE_EXAMPLES}


@router.get("/serve/recommend")
async def serve_recommend(
    model: str = Query(..., description="HF model id or local path"),
    mode: Literal["lab_safe", "workflow_max"] = "lab_safe",
    fetch_remote: bool = Query(
        True,
        description="If local cache misses config/README, fetch from huggingface.co",
    ),
) -> dict[str, Any]:
    """Recommend best-effort vLLM flags from HF card/config, lab recipes, Spark heuristics.

    Does not start a server — only returns a config the GUI (or client) can apply.
    """
    try:
        return autoconfig.recommend(model, mode=mode, fetch_remote=fetch_remote)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/serve/recipes")
async def serve_recipes() -> dict[str, Any]:
    """List lab-proven / curated model recipes used by auto-config."""
    return {"recipes": autoconfig.list_known_recipes()}


@router.post("/serve/stop")
async def serve_stop() -> dict[str, str]:
    job_id = await jobs.start_job("stop", serve.stop_all)
    return {"job_id": job_id}


@router.post("/serve/agent-restore")
async def serve_agent_restore() -> dict[str, str]:
    job_id = await jobs.start_job("agent_restore", serve.agent_restore)
    return {"job_id": job_id}


# ─── Jobs / logs ──────────────────────────────────────────────────────────────


@router.get("/jobs")
async def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    return db.list_jobs(limit)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return j


@router.get("/jobs/{job_id}/logs")
async def job_logs_sse(job_id: str) -> EventSourceResponse:
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    log_path = j.get("log_path")

    async def gen() -> AsyncIterator[dict[str, str]]:
        offset = 0
        while True:
            job = db.get_job(job_id) or j
            if log_path:
                chunk, offset = jobs.read_log_tail(log_path, offset)
                if chunk:
                    yield {"event": "log", "data": chunk}
            yield {
                "event": "status",
                "data": json.dumps(
                    {
                        "status": job.get("status"),
                        "progress": job.get("progress"),
                        "message": job.get("message"),
                    }
                ),
            }
            if job.get("status") in ("completed", "failed"):
                if job.get("result"):
                    yield {"event": "result", "data": json.dumps(job["result"])}
                break
            await asyncio.sleep(0.5)

    return EventSourceResponse(gen())


# ─── Chat / smoke ─────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    system: str = ""
    model: Optional[str] = None
    base_url: str = DEFAULT_BASE_URL
    max_tokens: int = 512
    temperature: float = 0.2
    thinking: bool = False
    stream: bool = True


@router.post("/chat")
async def chat(body: ChatRequest):
    """Proxy chat to vLLM. Client must stay open for the whole stream."""
    import time

    base = body.base_url.rstrip("/")
    timeout = httpx.Timeout(600.0, connect=30.0)

    # Resolve model with a short-lived client first
    async with httpx.AsyncClient(timeout=timeout) as probe:
        if not body.model:
            m = await probe.get(f"{base}/v1/models")
            m.raise_for_status()
            ids = [x["id"] for x in m.json().get("data", [])]
            if not ids:
                raise HTTPException(503, "no model loaded")
            model = ids[0]
        else:
            model = body.model

    messages = []
    if body.system:
        messages.append({"role": "system", "content": body.system})
    messages.append({"role": "user", "content": body.message})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
        "stream": body.stream,
        "chat_template_kwargs": {"enable_thinking": body.thinking},
    }

    if not body.stream:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{base}/v1/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or ""
            return {
                "content": content,
                "model": model,
                "usage": data.get("usage"),
                "system_fingerprint": data.get("system_fingerprint"),
            }

    payload["stream_options"] = {"include_usage": True}

    async def stream_gen():
        # Own client for the full stream lifetime (do not close early)
        t0 = time.perf_counter()
        ttft = None
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", f"{base}/v1/chat/completions", json=payload
                ) as resp:
                    if resp.status_code >= 400:
                        err = await resp.aread()
                        yield f"data: {json.dumps({'error': err.decode(errors='replace')[:500]})}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload_s = line[5:].strip()
                        if payload_s == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload_s)
                        except json.JSONDecodeError:
                            continue
                        usage = chunk.get("usage")
                        choices = chunk.get("choices") or []
                        piece = ""
                        if choices:
                            delta = choices[0].get("delta") or {}
                            # Qwen reasoning models may stream into reasoning fields
                            piece = (
                                delta.get("content")
                                or delta.get("reasoning")
                                or delta.get("reasoning_content")
                                or ""
                            )
                        if piece and ttft is None:
                            ttft = time.perf_counter() - t0
                        if piece:
                            yield f"data: {json.dumps({'content': piece})}\n\n"
                        if usage:
                            wall = time.perf_counter() - t0
                            comp = usage.get("completion_tokens") or 0
                            tps = (
                                (comp / (wall - ttft))
                                if ttft and wall > ttft and comp
                                else None
                            )
                            yield f"data: {json.dumps({'usage': usage, 'ttft_s': ttft, 'wall_s': wall, 'tok_per_s': tps, 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.post("/smoke")
async def smoke(base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    base = base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=180.0) as client:
        m = await client.get(f"{base}/v1/models")
        m.raise_for_status()
        ids = [x["id"] for x in m.json().get("data", [])]
        if not ids:
            raise HTTPException(503, "no model")
        model = ids[0]
        r = await client.post(
            f"{base}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "What is 12*17? Reply with only the number."}],
                "max_tokens": 64,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        r.raise_for_status()
        data = r.json()
        content = (data["choices"][0]["message"].get("content") or "").strip()
        ok = "204" in content and not ("!" * 5 in content)
        garbage = set(content) <= set("!?. \n") and "204" not in content
        return {
            "ok": ok and not garbage,
            "content": content[:500],
            "model": model,
            "system_fingerprint": data.get("system_fingerprint"),
            "expected": "204",
        }


# ─── Perf ─────────────────────────────────────────────────────────────────────


class PerfRequest(BaseModel):
    base_url: str = DEFAULT_BASE_URL
    model: Optional[str] = None
    intent: str = "attach"
    runner: Literal["workflow", "prefill", "concurrency"] = "workflow"
    concurrencies: list[int] = Field(default_factory=lambda: [1, 2, 4])
    concurrency: int = 4
    dollars_per_hour: float = 0.5


@router.post("/bench/perf")
async def bench_perf(body: PerfRequest) -> dict[str, str]:
    if body.runner == "workflow":

        def work(log, progress, **kw):
            return perf.run_workflow_bench(
                base_url=body.base_url,
                model=body.model,
                concurrencies=body.concurrencies,
                intent=body.intent,
                dollars_per_hour=body.dollars_per_hour,
                log=log,
                progress=progress,
            )

    elif body.runner == "prefill":

        def work(log, progress, **kw):
            return perf.run_external_prefill(
                base_url=body.base_url,
                model=body.model,
                intent=body.intent,
                log=log,
                progress=progress,
            )

    else:

        def work(log, progress, **kw):
            return perf.run_external_concurrency(
                base_url=body.base_url,
                model=body.model,
                concurrency=body.concurrency,
                intent=body.intent,
                log=log,
                progress=progress,
            )

    job_id = await jobs.start_job(f"perf_{body.runner}", work)
    return {"job_id": job_id}


# ─── Agentic ──────────────────────────────────────────────────────────────────


class AgenticRequest(BaseModel):
    base_url: str = DEFAULT_BASE_URL
    model: Optional[str] = None
    intent: str = "attach"
    suite: Literal["golden", "tool_eval"] = "golden"
    preset: Literal["short", "full", "hardmode", "coding"] = "short"
    seed: int = 42
    context_pressure: Optional[float] = None


@router.post("/bench/agentic")
async def bench_agentic(body: AgenticRequest) -> dict[str, str]:
    if body.suite == "golden":

        def work(log, progress, **kw):
            return agentic.run_golden_tools(
                base_url=body.base_url,
                model=body.model,
                intent=body.intent,
                log=log,
                progress=progress,
            )

    else:

        def work(log, progress, **kw):
            return agentic.run_tool_eval_bench(
                base_url=body.base_url,
                model=body.model,
                preset=body.preset,
                intent=body.intent,
                seed=body.seed,
                context_pressure=body.context_pressure,
                log=log,
                progress=progress,
            )

    job_id = await jobs.start_job(f"agentic_{body.suite}", work)
    return {"job_id": job_id}


@router.get("/bench/tool-eval-status")
async def tool_eval_status() -> dict[str, Any]:
    return agentic.tool_eval_available()


# ─── Runs / compare ───────────────────────────────────────────────────────────


@router.get("/runs")
async def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    return db.list_runs(limit)


@router.get("/runs/compare/{a}/{b}")
async def compare_runs(a: str, b: str, force: bool = False) -> dict[str, Any]:
    ra, rb = db.get_run(a), db.get_run(b)
    if not ra or not rb:
        raise HTTPException(404, "run not found")
    ea, eb = db.load_envelope(ra["path"]), db.load_envelope(rb["path"])
    warnings = []
    if (ea or {}).get("intent") != (eb or {}).get("intent"):
        warnings.append(
            f"intent mismatch: {(ea or {}).get('intent')} vs {(eb or {}).get('intent')}"
        )
    img_a = ((ea or {}).get("engine") or {}).get("image")
    img_b = ((eb or {}).get("engine") or {}).get("image")
    if img_a and img_b and img_a != img_b:
        warnings.append(f"image mismatch: {img_a} vs {img_b}")
    if warnings and not force:
        return {"comparable": False, "warnings": warnings, "a": ea, "b": eb}
    return {"comparable": True, "warnings": warnings, "a": ea, "b": eb}


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    row = db.get_run(run_id)
    if not row:
        raise HTTPException(404, "run not found")
    env = db.load_envelope(row["path"])
    return {"index": row, "envelope": env}


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str) -> dict[str, str]:
    row = db.get_run(run_id)
    if not row:
        raise HTTPException(404, "run not found")
    p = Path(row["path"])
    if p.exists():
        p.unlink()
    with db._conn() as c:
        c.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        c.commit()
    return {"deleted": run_id}
