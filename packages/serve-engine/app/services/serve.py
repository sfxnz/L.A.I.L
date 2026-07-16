"""Serve / stop vLLM — fully driven by explicit user config (no profiles)."""
from __future__ import annotations

import os
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from ..config import (
    CONTAINER_MAX,
    CONTAINER_SAFE,
    DEFAULT_IMAGE_MAX,
    DEFAULT_IMAGE_SAFE,
    DEFAULT_PORT,
    SAFE_MAX_LEN,
    SAFE_MIN_AVAIL_GIB,
    SAFE_UTIL,
    SERVE_EXAMPLES,
    SPARK_LAB,
    VLLM_RUNTIME,
    WORKFLOW_MAX_LEN,
    WORKFLOW_UTIL,
)
from .metadata import available_gib, list_vllm_containers


def serve_examples() -> dict[str, dict]:
    return SERVE_EXAMPLES


def stop_all(log: Any = None, progress: Callable | None = None, **_: Any) -> dict[str, Any]:
    def w(msg: str) -> None:
        if log:
            log.write(msg)
        if progress:
            progress(0.3, msg)

    if SPARK_LAB.exists():
        w(f"Running {SPARK_LAB} stop")
        r = subprocess.run(
            ["bash", str(SPARK_LAB), "stop"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if log:
            log.write(r.stdout or "")
            log.write(r.stderr or "")
        if progress:
            progress(1.0, "stopped")
        return {"ok": r.returncode == 0, "stdout": r.stdout, "stderr": r.stderr}

    names = [c["name"] for c in list_vllm_containers()]
    w(f"Stopping: {names}")
    for n in names:
        subprocess.run(["docker", "rm", "-f", n], capture_output=True, text=True)
    return {"ok": True, "stopped": names}


def _assert_lab_safe_util(util: float) -> None:
    if util > SAFE_UTIL + 1e-9:
        raise ValueError(
            f"Lab Safe mode refuses util={util} > {SAFE_UTIL}. "
            "Switch to Workflow Max or lower util."
        )


def _parse_extra_flags(extra_flags: str) -> list[str]:
    s = (extra_flags or "").strip()
    if not s:
        return []
    return shlex.split(s)


def _normalize_docker_env(user_env: list[str] | None) -> list[str]:
    """Only what the user (or GUI) sent — no silent defaults."""
    out: list[str] = []
    seen: set[str] = set()
    for item in user_env or []:
        item = item.strip()
        if not item or item.startswith("#") or "=" not in item:
            continue
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        if k in seen:
            # last wins
            out = [e for e in out if not e.startswith(k + "=")]
        seen.add(k)
        out.append(f"{k}={v}")
    return out


def _build_vllm_args(
    *,
    util: float,
    max_model_len: int,
    port: int,
    quantization: str = "",
    kv_cache_dtype: str = "",
    moe_backend: str = "",
    trust_remote_code: bool = False,
    enable_auto_tool_choice: bool = False,
    tool_call_parser: str = "",
    reasoning_parser: str = "",
    max_num_seqs: int | None = None,
    mtp: bool = False,
    mtp_num_tokens: int = 2,
    load_format: str = "",
    enable_chunked_prefill: bool = False,
    enable_prefix_caching: bool = False,
    extra_flags: str = "",
) -> list[str]:
    """Assemble the vLLM CLI from explicit fields + free-form extras. Nothing silent."""
    args: list[str] = [
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        str(util),
        "--max-model-len",
        str(max_model_len),
    ]
    if trust_remote_code:
        args.append("--trust-remote-code")
    if quantization.strip():
        args += ["--quantization", quantization.strip()]
    if kv_cache_dtype.strip():
        args += ["--kv-cache-dtype", kv_cache_dtype.strip()]
    if moe_backend.strip():
        args += ["--moe-backend", moe_backend.strip()]
    if max_num_seqs is not None and max_num_seqs > 0:
        args += ["--max-num-seqs", str(max_num_seqs)]
    if enable_auto_tool_choice:
        args.append("--enable-auto-tool-choice")
    if tool_call_parser.strip():
        args += ["--tool-call-parser", tool_call_parser.strip()]
    if reasoning_parser.strip():
        args += ["--reasoning-parser", reasoning_parser.strip()]
    if load_format.strip():
        args += ["--load-format", load_format.strip()]
    if enable_chunked_prefill:
        args.append("--enable-chunked-prefill")
    if enable_prefix_caching:
        args.append("--enable-prefix-caching")
    if mtp:
        # Compact JSON; shlex-safe single token after the flag
        cfg = f'{{"method":"mtp","num_speculative_tokens":{int(mtp_num_tokens)}}}'
        args += ["--speculative-config", cfg]
    args += _parse_extra_flags(extra_flags)
    return args


def serve_model(
    *,
    model: str,
    mode: str = "lab_safe",
    util: float | None = None,
    max_model_len: int | None = None,
    port: int = DEFAULT_PORT,
    image: str | None = None,
    docker_env: list[str] | None = None,
    quantization: str = "",
    kv_cache_dtype: str = "",
    moe_backend: str = "",
    trust_remote_code: bool = False,
    enable_auto_tool_choice: bool = False,
    tool_call_parser: str = "",
    reasoning_parser: str = "",
    max_num_seqs: int | None = None,
    mtp: bool = False,
    mtp_num_tokens: int = 2,
    load_format: str = "",
    enable_chunked_prefill: bool = False,
    enable_prefix_caching: bool = False,
    extra_flags: str = "",
    stop_first: bool = True,
    log: Any = None,
    progress: Callable | None = None,
    # legacy no-ops so old clients don't crash
    profile: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    def w(msg: str, p: float = 0.1) -> None:
        if log:
            log.write(msg)
        if progress:
            progress(p, msg)

    if mode == "lab_safe":
        util = SAFE_UTIL if util is None else util
        max_model_len = SAFE_MAX_LEN if max_model_len is None else max_model_len
        _assert_lab_safe_util(util)
        image = image or DEFAULT_IMAGE_SAFE
        container = CONTAINER_SAFE
    elif mode == "workflow_max":
        util = WORKFLOW_UTIL if util is None else util
        max_model_len = WORKFLOW_MAX_LEN if max_model_len is None else max_model_len
        image = image or DEFAULT_IMAGE_MAX
        container = CONTAINER_MAX
    else:
        raise ValueError(f"Unknown mode {mode}; use lab_safe or workflow_max")

    env_list = _normalize_docker_env(docker_env)

    # Guard: flashinfer_b12x crashes on mixed FP8+NVFP4 compressed-tensors MoE (e.g. Unsloth 35B).
    moe_backend = (moe_backend or "").strip()
    quant_l = (quantization or "").strip().lower()
    mid = (model or "").lower()
    if moe_backend == "flashinfer_b12x" and (
        quant_l in ("compressed-tensors", "compressed_tensors")
        or "unsloth" in mid
        or "mixed" in mid
    ):
        # Confirm via local HF config when available
        unsafe = quant_l in ("compressed-tensors", "compressed_tensors") or "unsloth" in mid
        try:
            from .autoconfig import analyze_config, load_local_fallback

            local = load_local_fallback(model)
            if local.get("config"):
                det = analyze_config(local["config"], model)
                unsafe = bool(
                    det.get("is_mixed_nvfp4_fp8")
                    or (det.get("is_moe") and det.get("has_fp8") and det.get("quant_flag") == "compressed-tensors")
                )
        except Exception:
            pass
        if unsafe:
            w(
                "SAFETY: dropping --moe-backend flashinfer_b12x "
                "(not supported for FP8 MoE on mixed compressed-tensors; leave auto)",
                0.15,
            )
            moe_backend = ""

    vllm_args = _build_vllm_args(
        util=util,
        max_model_len=max_model_len,
        port=port,
        quantization=quantization,
        kv_cache_dtype=kv_cache_dtype,
        moe_backend=moe_backend,
        trust_remote_code=trust_remote_code,
        enable_auto_tool_choice=enable_auto_tool_choice,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        max_num_seqs=max_num_seqs,
        mtp=mtp,
        mtp_num_tokens=mtp_num_tokens,
        load_format=load_format,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_prefix_caching=enable_prefix_caching,
        extra_flags=extra_flags,
    )

    if stop_first:
        w("Stopping existing vLLM containers…", 0.05)
        stop_all(log=log)

    w("Launching docker vLLM (manual config)…", 0.2)
    subprocess.run(["docker", "rm", "-f", container], capture_output=True)
    hf = str(Path.home() / ".cache" / "huggingface")
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container,
        "--restart",
        "no",
        "--gpus",
        "all",
        "--shm-size=4g",
        "-p",
        f"127.0.0.1:{port}:{port}",
        "-v",
        f"{hf}:/root/.cache/huggingface",
    ]
    for e in env_list:
        cmd += ["-e", e]
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if not hf_token:
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        if token_file.is_file():
            try:
                hf_token = token_file.read_text().strip()
            except OSError:
                hf_token = ""
    cmd += [
        "-e",
        f"HF_TOKEN={hf_token}",
        "-e",
        f"HUGGING_FACE_HUB_TOKEN={hf_token}",
        image,
        model,
        *vllm_args,
    ]

    def _redact_cmd(parts: list[str]) -> str:
        """Never log HF tokens / secrets in job logs."""
        out: list[str] = []
        for c in parts:
            if c.startswith("HF_TOKEN=") or c.startswith("HUGGING_FACE_HUB_TOKEN="):
                k = c.split("=", 1)[0]
                out.append(f"{k}=***")
            else:
                out.append(shlex.quote(c))
        return " ".join(out)

    w(f"image={image}", 0.25)
    w(f"docker_env={env_list}", 0.26)
    w(f"vllm_args={vllm_args}", 0.27)
    w(f"$ {_redact_cmd(cmd)}", 0.3)

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout or "docker run failed")
    if log:
        log.write((r.stdout or "").strip())
        log.write("Waiting for /v1/models (up to 10 min)…")

    def _container_dead() -> str | None:
        """Return docker status string if container is not running."""
        st = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", container],
            capture_output=True,
            text=True,
        )
        status = (st.stdout or "").strip()
        if not status:
            return "missing"
        if status.startswith("running"):
            return None
        return status

    def _tail_logs(n: int = 120) -> str:
        try:
            return subprocess.check_output(
                ["docker", "logs", "--tail", str(n), container],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except Exception as e:
            return str(e)

    ready = False
    for i in range(120):
        dead = _container_dead()
        if dead is not None:
            logs = _tail_logs()
            if log:
                log.write(f"container exited early ({dead})")
                log.write(logs)
            raise RuntimeError(
                f"vLLM container exited ({dead}) before /v1/models became ready.\n"
                f"--- docker logs ---\n{logs[-4000:]}"
            )
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3)
            ready = True
            break
        except Exception:
            time.sleep(5)
            if i % 6 == 0 and log:
                log.write(f"still loading… ({i * 5}s)")
    if not ready:
        logs = _tail_logs()
        if log:
            log.write(logs)
        raise RuntimeError(f"Timeout waiting for /v1/models\n--- docker logs ---\n{logs[-4000:]}")

    avail = available_gib()
    if mode == "lab_safe" and avail is not None and avail < SAFE_MIN_AVAIL_GIB:
        subprocess.run(["docker", "stop", container], capture_output=True)
        raise RuntimeError(
            f"ABORT: available {avail} GiB < {SAFE_MIN_AVAIL_GIB} GiB — container stopped"
        )
    w(f"API ready. available_gib={avail}", 1.0)
    return {
        "ok": True,
        "mode": mode,
        "model": model,
        "container": container,
        "util": util,
        "max_model_len": max_model_len,
        "available_gib": avail,
        "image": image,
        "docker_env": env_list,
        "vllm_args": vllm_args,
    }


def agent_restore(log: Any = None, progress: Callable | None = None, **_: Any) -> dict[str, Any]:
    if not SPARK_LAB.exists():
        script = VLLM_RUNTIME / "scripts" / "vllm-use-model.sh"
        if script.exists():
            if log:
                log.write(f"Using {script} 27b")
            r = subprocess.run(["bash", str(script), "27b"], capture_output=True, text=True)
            return {"ok": r.returncode == 0, "stdout": r.stdout, "stderr": r.stderr}
        raise FileNotFoundError("spark_lab.sh not found and no vllm-use-model.sh")
    r = subprocess.run(
        ["bash", str(SPARK_LAB), "agent-restore"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if log:
        log.write(r.stdout or "")
        log.write(r.stderr or "")
    if progress:
        progress(1.0, "agent-restore done")
    return {"ok": r.returncode == 0, "stdout": r.stdout, "stderr": r.stderr}


def ensure_hf_cache_writable(log: Any = None) -> None:
    """Host-side hf download fails if Docker left ~/.cache/huggingface owned by root."""
    cache = Path.home() / ".cache" / "huggingface"
    hub = cache / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    test = hub / ".lab_write_test"
    try:
        test.write_text("ok")
        test.unlink(missing_ok=True)
        return
    except PermissionError:
        pass
    if log:
        log.write(
            f"HF cache not writable by uid={os.getuid()} — fixing ownership via docker "
            f"(root-owned cache is common after vLLM containers write into the mount)…"
        )
    r = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{cache}:/hf",
            "alpine",
            "chown",
            "-R",
            f"{os.getuid()}:{os.getgid()}",
            "/hf",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if r.returncode != 0:
        raise RuntimeError(
            "HF cache is root-owned and auto-chown failed. "
            f"stderr={r.stderr or r.stdout}. "
            f"Fix manually: docker run --rm -v {cache}:/hf alpine chown -R "
            f"{os.getuid()}:{os.getgid()} /hf"
        )
    try:
        test.write_text("ok")
        test.unlink(missing_ok=True)
    except PermissionError as e:
        raise RuntimeError(f"HF cache still not writable after chown: {e}") from e
    if log:
        log.write("HF cache ownership fixed.")


def _hf_download_env() -> dict[str, str]:
    """Stable HF download env: Xet high-perf hangs on flaky links; disable it."""
    env = os.environ.copy()
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    env.pop("HF_XET_HIGH_PERFORMANCE", None)
    env["HF_HUB_DISABLE_XET"] = "1"
    # Prefer token from env or cached login (hf stores it; CLI picks it up)
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if not env.get("HF_TOKEN") and token_file.is_file():
        try:
            env["HF_TOKEN"] = token_file.read_text().strip()
            env["HUGGING_FACE_HUB_TOKEN"] = env["HF_TOKEN"]
        except OSError:
            pass
    return env


def download_model(
    model: str,
    log: Any = None,
    progress: Callable | None = None,
) -> dict[str, Any]:
    """Download model weights with live logs. Resumes partial cache. Idempotent if complete."""
    ensure_hf_cache_writable(log=log)
    env = _hf_download_env()

    # Resolve hf binary
    hf_bin = "hf"
    for cand in (
        Path.home() / ".local" / "bin" / "hf",
        Path("/usr/local/bin/hf"),
    ):
        if cand.is_file():
            hf_bin = str(cand)
            break

    if log:
        log.write(f"hf download {model} (HF_HUB_DISABLE_XET=1, resumes cache)")
        if env.get("HF_TOKEN"):
            log.write("HF_TOKEN: present")
        else:
            log.write("HF_TOKEN: missing — run `hf auth login` for higher rate limits")

    if progress:
        progress(0.05, "downloading…")

    cmd = [hf_bin, "download", model]
    # Stream so the GUI job log shows tqdm / errors live (capture_output hid hangs)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    assert proc.stdout is not None
    last_pct = 0.05
    lines: list[str] = []
    try:
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            lines.append(line)
            if log:
                log.write(line)
            # Rough progress from "Fetching N files:  42%|" style lines
            if "%" in line and "Fetching" in line:
                try:
                    # take last integer before %
                    part = line.rsplit("%", 1)[0]
                    num = ""
                    for ch in reversed(part):
                        if ch.isdigit():
                            num = ch + num
                        elif num:
                            break
                    if num:
                        pct = max(0.05, min(0.95, int(num) / 100.0))
                        if pct - last_pct >= 0.02 and progress:
                            progress(pct, f"download {int(num)}%")
                            last_pct = pct
                except Exception:
                    pass
        code = proc.wait(timeout=7200)
    except Exception:
        proc.kill()
        raise

    if code != 0:
        # Signal deaths (e.g. kill -9) often leave only tqdm on stdout
        tail = "\n".join(lines[-15:]) if lines else ""
        if code < 0:
            raise RuntimeError(
                f"hf download killed by signal {-code}. "
                "Re-run download; partial cache is kept and will resume. "
                f"Last output:\n{tail}"
            )
        raise RuntimeError(
            f"hf download failed (exit {code}). Last output:\n{tail or '(empty)'}"
        )

    if progress:
        progress(1.0, "downloaded")
    if log:
        log.write(f"download complete for {model}")
    return {"ok": True, "model": model}
