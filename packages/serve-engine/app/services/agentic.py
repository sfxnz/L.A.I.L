"""Agentic / tool-calling evaluation: golden tools + tool-eval-bench."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Callable

from ..config import DEFAULT_BASE_URL, RUNS_DIR
from .. import db
from .metadata import build_envelope, make_run_id, probe_endpoint
import asyncio


GOLDEN_TOOLS_DEF = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command on the host",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

# (prompt, allowed tool names or None for no-tool)
GOLDEN_CASES: list[tuple[str, set[str] | None]] = [
    ("List the files in /home/sfxnz", {"run_command"}),
    ("What's inside /etc/hostname?", {"read_file", "run_command"}),
    ("Create a file at /tmp/hello.txt containing the word hi", {"write_file", "run_command"}),
    ("How much disk space is free?", {"run_command"}),
    ("Search the web for the latest vLLM release", {"web_search"}),
    ("What processes are using the most memory right now?", {"run_command"}),
    ("Save a note saying 'buy milk' to /tmp/note.txt", {"write_file", "run_command"}),
    ("Show me the first 10 lines of /var/log/syslog", {"read_file", "run_command"}),
    ("Look up today's Bitcoin price", {"web_search"}),
    ("What's the capital of France?", None),
    ("Explain what a KV cache is in one paragraph.", None),
    ("Thanks, that's all for now.", None),
]

INSTALL_HINT = (
    "uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git"
)

# Prefer uv-tool install location; serve-engine may not inherit interactive shell PATH.
_EXTRA_PATHS = [
    Path.home() / ".local" / "bin",
    Path("/usr/local/bin"),
]


def _augmented_env() -> dict[str, str]:
    env = dict(os.environ)
    parts = [str(p) for p in _EXTRA_PATHS if p.is_dir()]
    if parts:
        env["PATH"] = os.pathsep.join(parts + [env.get("PATH", "")])
    return env


def _resolve_tool_eval_bin() -> str | None:
    env = _augmented_env()
    which = shutil.which("tool-eval-bench", path=env.get("PATH"))
    if which:
        return which
    for p in _EXTRA_PATHS:
        cand = p / "tool-eval-bench"
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def _chat_tools(base: str, model: str, prompt: str) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": GOLDEN_TOOLS_DEF,
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def _probe_tools_enabled(base: str, model_id: str) -> str | None:
    """Return error message if tools are not supported on this serve."""
    try:
        _chat_tools(base, model_id, "Reply with no tools: ping")
        return None
    except Exception as e:
        msg = str(e)
        try:
            import urllib.error

            if isinstance(e, urllib.error.HTTPError):
                body = e.read().decode(errors="replace")
                msg = body or msg
                if "enable-auto-tool-choice" in body or "tool-call-parser" in body:
                    return (
                        "Server rejects tools: restart with --enable-auto-tool-choice "
                        "and --tool-call-parser (e.g. qwen3_coder / qwen3_xml). "
                        f"Detail: {body[:240]}"
                    )
        except Exception:
            pass
        if "400" in msg or "tool" in msg.lower():
            return (
                "Tool-calling not enabled on this endpoint. "
                "Serve with --enable-auto-tool-choice --tool-call-parser <parser>. "
                f"({msg[:200]})"
            )
        return None


def run_golden_tools(
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = None,
    intent: str = "attach",
    log: Any = None,
    progress: Callable | None = None,
) -> dict[str, Any]:
    base = base_url.rstrip("/")
    with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as r:
        models = json.loads(r.read().decode())
    model_id = model or models["data"][0]["id"]

    probe_err = _probe_tools_enabled(base, model_id)
    if probe_err:
        if log:
            log.write(probe_err)
        raise RuntimeError(probe_err)

    results = []
    passes = 0
    fingerprint = None
    for i, (prompt, allowed) in enumerate(GOLDEN_CASES):
        if progress:
            progress(i / len(GOLDEN_CASES), prompt[:40])
        try:
            data = _chat_tools(base, model_id, prompt)
            msg = data["choices"][0]["message"]
            tcs = msg.get("tool_calls") or []
            if allowed is None:
                ok = not tcs
            else:
                ok = False
                if tcs:
                    fn = tcs[0]["function"]
                    try:
                        ok = fn["name"] in allowed and isinstance(
                            json.loads(fn["arguments"]), dict
                        )
                    except Exception:
                        ok = False
            got = tcs[0]["function"]["name"] if tcs else "no-tool"
            fingerprint = data.get("system_fingerprint")
        except Exception as e:
            ok, got = False, f"ERROR {e}"
        results.append({"prompt": prompt, "ok": ok, "got": got})
        if log:
            log.write(f"{'PASS' if ok else 'FAIL'} | {prompt[:50]} -> {got}")
        passes += int(ok)

    score = round(100 * passes / len(GOLDEN_CASES))
    agentic = {
        "suite": "golden_tools",
        "passed": passes,
        "total": len(GOLDEN_CASES),
        "score": score,
        "cases": results,
    }
    run_id = make_run_id()
    probe = asyncio.run(probe_endpoint(base, timeout=10))
    envelope = build_envelope(
        run_id=run_id,
        intent=intent,
        model_id=model_id,
        kind="agentic_golden",
        workload={"type": "golden_tools", "n": len(GOLDEN_CASES)},
        metrics={},
        agentic=agentic,
        probe=probe,
        engine_extra={"system_fingerprint": fingerprint} if results else {},
    )
    path = RUNS_DIR / f"{run_id}.json"
    path.write_text(json.dumps(envelope, indent=2))
    db.insert_run(
        run_id=run_id,
        created_at=envelope["created_at"],
        kind="agentic_golden",
        intent=intent,
        model_id=model_id,
        summary={"score": score, "passed": passes, "total": len(GOLDEN_CASES)},
        path=str(path),
    )
    if progress:
        progress(1.0, f"{passes}/{len(GOLDEN_CASES)}")
    return envelope


def tool_eval_available() -> dict[str, Any]:
    bin_path = _resolve_tool_eval_bin()
    if not bin_path:
        return {
            "available": False,
            "install": INSTALL_HINT,
            "repo": "https://github.com/SeraphimSerapis/tool-eval-bench",
        }

    version: str | None = None
    try:
        out = subprocess.run(
            [bin_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            env=_augmented_env(),
        )
        # e.g. "tool-eval-bench 2.3.2.dev1+g910777110"
        version = (out.stdout or out.stderr or "").strip().split()[-1] or None
    except Exception:
        pass

    return {
        "available": True,
        "path": bin_path,
        "via": "cli",
        "version": version,
        "repo": "https://github.com/SeraphimSerapis/tool-eval-bench",
    }


def run_tool_eval_bench(
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = None,
    preset: str = "short",
    intent: str = "attach",
    seed: int = 42,
    context_pressure: float | None = None,
    no_think: bool = True,
    log: Any = None,
    progress: Callable | None = None,
) -> dict[str, Any]:
    info = tool_eval_available()
    if not info.get("available"):
        raise RuntimeError(
            "tool-eval-bench not installed. " + info.get("install", INSTALL_HINT)
        )

    bin_path = info.get("path") or _resolve_tool_eval_bin()
    if not bin_path:
        raise RuntimeError("tool-eval-bench binary not found on PATH")

    run_id = make_run_id()
    json_out = RUNS_DIR / f"{run_id}_tool_eval.json"
    base = base_url.rstrip("/")

    # v2.x CLI: `run` subcommand; --no-think for Qwen/Laguna reasoning models.
    cmd = [
        bin_path,
        "run",
        "--base-url",
        base,
        "--backend",
        "vllm",
        "--seed",
        str(seed),
        "--json-file",
        str(json_out),
        "--no-live",
    ]
    if no_think:
        cmd.append("--no-think")
    if model:
        cmd += ["--model", model]
    if preset == "short":
        cmd.append("--short")
    elif preset == "hardmode":
        cmd.append("--hardmode")
    elif preset == "coding":
        cmd += ["--categories", "J", "G", "M", "O"]
    # full = no extra scenario flags (full 69)
    if context_pressure is not None:
        cmd += ["--context-pressure", str(context_pressure)]

    if log:
        log.write(f"$ {' '.join(cmd)}")
    if progress:
        progress(0.02, "starting tool-eval-bench")

    env = _augmented_env()
    env["TOOL_EVAL_BASE_URL"] = base

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert proc.stdout
    n_lines = 0
    done_hint = 0
    for line in proc.stdout:
        line = line.rstrip()
        n_lines += 1
        if log:
            log.write(line)
        # Progress from JSON event stream (--no-live emits events as JSON lines)
        try:
            if line.startswith("{") and "scenario" in line.lower():
                ev = json.loads(line)
                idx = ev.get("index")
                total = ev.get("total")
                if isinstance(idx, int) and isinstance(total, int) and total > 0:
                    done_hint = max(done_hint, (idx + 1) / total)
                    if progress:
                        progress(
                            0.05 + 0.9 * done_hint,
                            f"{ev.get('scenario_id', 'scenario')} {ev.get('status', '')}".strip()[
                                :80
                            ],
                        )
                elif ev.get("event") == "benchmark_complete" and progress:
                    progress(0.98, f"score={ev.get('final_score')}")
        except Exception:
            if progress and n_lines % 5 == 0:
                progress(min(0.9, 0.1 + n_lines * 0.01), line[:80])

    code = proc.wait()
    if code != 0 and not json_out.exists():
        raise RuntimeError(f"tool-eval-bench exited {code}")

    raw: dict[str, Any] = {}
    if json_out.exists():
        raw = json.loads(json_out.read_text())

    cfg = raw.get("config") if isinstance(raw.get("config"), dict) else {}
    meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    model_id = (
        model
        or (cfg or {}).get("model")
        or (meta or {}).get("model")
        or raw.get("model")
    )

    agentic = {
        "suite": "tool-eval-bench",
        "preset": preset,
        "final_score": raw.get("final_score"),
        "rating": raw.get("rating"),
        "total_scenarios": raw.get("total_scenarios"),
        "safety_warnings": raw.get("safety_warnings"),
        "deployability": raw.get("deployability"),
        "responsiveness": raw.get("responsiveness"),
        "safety_gate": raw.get("safety_gate"),
        "scores": raw.get("scores"),
        "tool_eval_bench_version": raw.get("tool_eval_bench_version") or info.get("version"),
        "report_path": raw.get("report_path"),
        "raw_path": str(json_out),
        "exit_code": code,
    }
    probe = asyncio.run(probe_endpoint(base, timeout=10))
    envelope = build_envelope(
        run_id=run_id,
        intent=intent,
        model_id=model_id,
        kind="agentic_tool_eval",
        workload={
            "type": "tool-eval-bench",
            "preset": preset,
            "seed": seed,
            "context_pressure": context_pressure,
            "no_think": no_think,
        },
        metrics={},
        agentic=agentic,
        probe=probe,
    )
    # Prefer probed / envelope model id (probe may fill when CLI omitted --model)
    model_id = (
        model_id
        or ((envelope.get("model") or {}).get("id") if isinstance(envelope.get("model"), dict) else None)
        or ((raw.get("config") or {}).get("model") if isinstance(raw.get("config"), dict) else None)
    )
    if model_id and isinstance(envelope.get("model"), dict):
        envelope["model"]["id"] = model_id
    path = RUNS_DIR / f"{run_id}.json"
    path.write_text(json.dumps(envelope, indent=2))
    scores = agentic.get("scores") if isinstance(agentic.get("scores"), dict) else {}
    cat = scores.get("category_scores") if isinstance(scores, dict) else None
    db.insert_run(
        run_id=run_id,
        created_at=envelope["created_at"],
        kind="agentic_tool_eval",
        intent=intent,
        model_id=model_id,
        summary={
            "final_score": agentic.get("final_score"),
            "rating": agentic.get("rating"),
            "preset": preset,
            "total_scenarios": agentic.get("total_scenarios"),
            "deployability": agentic.get("deployability"),
            "responsiveness": agentic.get("responsiveness"),
            "safety_passed": (agentic.get("safety_gate") or {}).get("passed")
            if isinstance(agentic.get("safety_gate"), dict)
            else None,
            "pass_rate": scores.get("total_points") and scores.get("max_points")
            and round(100 * float(scores["total_points"]) / float(scores["max_points"]), 1)
            if isinstance(scores, dict) and scores.get("max_points")
            else None,
            "categories": [
                {"id": c.get("category"), "label": c.get("label"), "percent": c.get("percent")}
                for c in (cat or [])
                if isinstance(c, dict)
            ][:16],
            "engine": ((envelope.get("engine") or {}).get("image") if isinstance(envelope.get("engine"), dict) else None),
        },
        path=str(path),
    )
    if progress:
        progress(1.0, f"score={agentic.get('final_score')}")
    if log and agentic.get("final_score") is not None:
        log.write(
            f"=== tool-eval-bench done: score={agentic.get('final_score')} "
            f"rating={agentic.get('rating')} scenarios={agentic.get('total_scenarios')} ==="
        )
    return envelope
