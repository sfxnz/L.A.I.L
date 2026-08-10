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
    DATA_DIR,
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


# ─── Multi-node launcher ──────────────────────────────────────────────────────

def _strip_structured_flags(args: list[str]) -> list[str]:
    """Remove flags the launcher already sets explicitly, so a free-form extra_flags
    blob can be reused verbatim for both head and worker without duplicates."""
    drop = {
        "--tensor-parallel-size",
        "--pipeline-parallel-size",
        "--nnodes",
        "--node-rank",
        "--master-addr",
        "--master-port",
        "--distributed-executor-backend",
        "--host",
        "--port",
    }
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in drop:
            if i + 1 < len(args) and not str(args[i + 1]).startswith("-"):
                i += 2
            else:
                i += 1
            continue
        if any(a.startswith(f + "=") for f in drop):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def build_multi_node_launch(
    *,
    image: str,
    model: str,
    vllm_args: list[str],
    env_list: list[str],
    head: dict[str, Any],
    workers: list[dict[str, Any]],
    nnodes: int,
    port: int,
    headless: bool = True,
) -> dict[str, Any]:
    """Build the per-node docker command set for a TP=nnodes multi-node serve.
    Pure function (no subprocess) so it is fully testable. Returns head + worker cmds."""
    hf = str(Path.home() / ".cache" / "huggingface")
    master_addr = head.get("qsfp_ip") or "127.0.0.1"
    base_args = _strip_structured_flags(vllm_args)

    # Environment that is identical across nodes (model/runtime knobs), minus host-IP keys.
    shared_env = [
        e
        for e in env_list
        if not e.startswith(("VLLM_HOST_IP=", "WORKER_VLLM_HOST_IP=", "NODE_RANK=", "MASTER_ADDR="))
    ]

    def docker_prefix(name: str, node_ip: str | None) -> list[str]:
        cmd = [
            "docker", "run", "-d", "--name", name, "--restart", "no",
            "--gpus", "all", "--network", "host", "--ipc", "host", "--shm-size=32g",
            "--device", "/dev/infiniband",
            # RDMA needs locked (pinned) memory + raw verbs access; without these
            # NCCL fails at init with "unhandled system error".
            "--cap-add", "IPC_LOCK",
            "--ulimit", "memlock=-1:-1",
            # Runtime images (e.g. Anemll dspark-vllm-gx10) ship ENTRYPOINT=vllm.
            # Clear it so our bash wrapper runs as the command instead of being
            # appended as arguments to vllm (=> "unrecognized arguments" exit 2).
            "--entrypoint", "bash",
            "-v", f"{hf}:/cache/huggingface",
        ]
        for e in shared_env:
            cmd += ["-e", e]
        if node_ip:
            cmd += ["-e", f"VLLM_HOST_IP={node_ip}"]
        cmd += ["-e", "HF_HOME=/cache/huggingface"]
        return cmd

    def serve_suffix(rank: int, is_head: bool) -> list[str]:
        args = [
            "vllm", "serve", model,
            "--tensor-parallel-size", str(nnodes),
            "--pipeline-parallel-size", "1",
            "--nnodes", str(nnodes),
            "--node-rank", str(rank),
            "--master-addr", master_addr,
            "--master-port", "25000",
            "--distributed-executor-backend", "mp",
        ]
        if is_head:
            args += ["--host", "0.0.0.0", "--port", str(port)]
        elif headless:
            args += ["--headless"]
        args += base_args
        return args

    # --entrypoint is bash, so args start at bash's own flags (no leading "bash").
    # `bash -lc 'exec "$@"' -- vllm serve …` makes "--" $0 and the rest "$@".
    entry = ["-lc", 'export PATH=/usr/local/cuda/bin:/usr/local/bin:$PATH; exec "$@"', "--"]

    head_cmd = docker_prefix(f"spark-vllm-n0", head.get("qsfp_ip")) + [
        "-e", "NODE_RANK=0", "-e", f"MASTER_ADDR={master_addr}",
        image, *entry, *serve_suffix(0, True),
    ]
    worker_cmds = []
    for idx, wnode in enumerate(workers, start=1):
        wc = docker_prefix(f"spark-vllm-n{idx}", wnode.get("qsfp_ip")) + [
            "-e", f"NODE_RANK={idx}", "-e", f"MASTER_ADDR={master_addr}",
            image, *entry, *serve_suffix(idx, False),
        ]
        worker_cmds.append({"node": wnode.get("id") or f"worker{idx}", "ssh_host": wnode.get("ssh_host") or wnode.get("id"), "rank": idx, "cmd": wc})

    return {"head": {"rank": 0, "cmd": head_cmd}, "workers": worker_cmds, "nnodes": nnodes, "port": port, "model": model, "image": image}


_MULTINODE_STATE = DATA_DIR / "multinode_serve.json"


def _redact(parts: list[str]) -> str:
    out = []
    for c in parts:
        if c.startswith(("HF_TOKEN=", "HUGGING_FACE_HUB_TOKEN=")):
            out.append(c.split("=", 1)[0] + "=***")
        else:
            out.append(shlex.quote(c))
    return " ".join(out)


def _launch_multi_node(
    launch: dict[str, Any],
    *,
    port: int,
    log: Any = None,
    progress: Callable | None = None,
) -> dict[str, Any]:
    """Start workers (ssh) then head, then wait for /v1/models on the head. Records a
    state file so stop_all can tear the whole cluster down."""

    def w(msg: str, p: float = 0.3) -> None:
        if log:
            log.write(msg)
        if progress:
            progress(p, msg)

    nnodes = launch["nnodes"]
    # 1) workers first (headless ranks), each over SSH
    for wc in launch["workers"]:
        host = wc["ssh_host"]
        w(f"Starting worker rank {wc['rank']} on {wc['node']} ({host})…")
        remote = _redact(wc["cmd"])
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, remote],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            raise RuntimeError(f"worker {wc['node']} failed to start: {r.stderr or r.stdout}")
        w(f"worker {wc['node']} up")

    # 2) head (serves the API)
    w("Starting head (rank 0, API)…", 0.5)
    hr = subprocess.run(launch["head"]["cmd"], capture_output=True, text=True)
    if hr.returncode != 0:
        raise RuntimeError(f"head failed to start: {hr.stderr or hr.stdout}")

    # 3) record state for stop_all
    try:
        _MULTINODE_STATE.write_text(
            __import__("json").dumps(
                {
                    "model": launch["model"],
                    "image": launch["image"],
                    "nnodes": nnodes,
                    "port": port,
                    "workers": [{"ssh_host": w["ssh_host"], "node": w["node"], "rank": w["rank"]} for w in launch["workers"]],
                },
                indent=2,
            )
        )
    except Exception:
        pass

    # 4) wait for readiness on the head endpoint
    w("Waiting for /v1/models on head (multi-node load can take 15–30+ min)…", 0.6)
    ready = False
    for i in range(360):  # up to 30 min
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3)
            ready = True
            break
        except Exception:
            time.sleep(5)
            if i % 12 == 0:
                w(f"still loading… ({i * 5}s)")
    if not ready:
        raise RuntimeError("Timeout waiting for multi-node /v1/models on head")
    w(f"Multi-node serve ready on :{port} ({nnodes} nodes)", 1.0)
    return {"ok": True, "multi_node": True, "nnodes": nnodes, "model": launch["model"], "port": port}


def stop_multi_node(log: Any = None) -> dict[str, Any]:
    """Tear down a recorded multi-node serve across head + workers (state file)."""
    stopped: list[str] = []
    try:
        import json as _json

        st = _json.loads(_MULTINODE_STATE.read_text())
    except Exception:
        return {"ok": False, "reason": "no multinode state"}
    for wr in st.get("workers") or []:
        host = wr.get("ssh_host")
        if host:
            name = f"spark-vllm-n{wr.get('rank', 1)}"
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
                 f"docker rm -f {shlex.quote(name)}"],
                capture_output=True, text=True, timeout=60,
            )
            stopped.append(f"{host}:{name}")
    subprocess.run(["docker", "rm", "-f", "spark-vllm-n0"], capture_output=True, text=True)
    stopped.append("spark-vllm-n0")
    try:
        _MULTINODE_STATE.unlink(missing_ok=True)
    except Exception:
        pass
    return {"ok": True, "stopped": stopped}


def _vllm_container_name(name: str) -> bool:
    n = (name or "").lower()
    return "vllm" in n or n.startswith("spark-vllm")


def stop_cluster_remote_vllm(log: Any = None) -> dict[str, Any]:
    """Stop vLLM containers on non-local cluster nodes via SSH.

    Ground truth is live docker on each node — not multinode_serve.json.
    That state file can be missing after engine restarts, manual launches,
    or a prior stop that removed the file while a remote rm failed.
    """
    from .cluster import _load_cluster_config

    stopped: list[str] = []
    errors: list[str] = []
    cfg = _load_cluster_config()
    for node in cfg.get("nodes") or []:
        if node.get("local"):
            continue
        host = node.get("ssh_host") or node.get("id")
        if not host:
            continue
        try:
            listed = subprocess.run(
                [
                    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", str(host),
                    "docker ps -a --format '{{.Names}}'",
                ],
                capture_output=True, text=True, timeout=30,
            )
        except Exception as e:
            errors.append(f"{host}: list failed: {e}")
            continue
        if listed.returncode != 0:
            errors.append(f"{host}: list exit {listed.returncode}: {(listed.stderr or '')[:200]}")
            continue
        names = [ln.strip() for ln in listed.stdout.splitlines() if _vllm_container_name(ln.strip())]
        if not names:
            continue
        if log:
            log.write(f"Remote {host}: stopping {names}")
        for name in names:
            try:
                rm = subprocess.run(
                    [
                        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", str(host),
                        f"docker rm -f {shlex.quote(name)}",
                    ],
                    capture_output=True, text=True, timeout=60,
                )
            except Exception as e:
                errors.append(f"{host}:{name}: {e}")
                continue
            if rm.returncode == 0:
                stopped.append(f"{host}:{name}")
            else:
                errors.append(f"{host}:{name}: rm exit {rm.returncode}")
    return {"ok": not errors, "stopped": stopped, "errors": errors}


def serve_examples() -> dict[str, dict]:
    return SERVE_EXAMPLES


def stop_all(log: Any = None, progress: Callable | None = None, **_: Any) -> dict[str, Any]:
    def w(msg: str) -> None:
        if log:
            log.write(msg)
        if progress:
            progress(0.3, msg)

    stopped: list[str] = []

    # 1) Optional state-file path (fast path when multinode_serve.json exists).
    try:
        mn = stop_multi_node(log=log)
        if mn.get("ok"):
            w(f"Stopped multi-node serve: {mn.get('stopped')}")
            stopped.extend(mn.get("stopped") or [])
    except Exception:
        pass

    # 2) Always live-discover remote workers. State file is not required.
    try:
        rem = stop_cluster_remote_vllm(log=log)
        if rem.get("stopped"):
            w(f"Stopped remote vLLM: {rem.get('stopped')}")
            for item in rem["stopped"]:
                if item not in stopped:
                    stopped.append(item)
        if rem.get("errors") and log:
            log.write(f"remote stop warnings: {rem['errors']}")
    except Exception as e:
        if log:
            log.write(f"remote cluster stop failed: {e}")

    # 3) Local containers (spark_lab helper if present, else docker rm).
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
        return {
            "ok": r.returncode == 0,
            "stopped": stopped,
            "stdout": r.stdout,
            "stderr": r.stderr,
        }

    names = [c["name"] for c in list_vllm_containers()]
    w(f"Stopping: {names}")
    for n in names:
        subprocess.run(["docker", "rm", "-f", n], capture_output=True, text=True)
        if n not in stopped:
            stopped.append(n)
    if progress:
        progress(1.0, "stopped")
    return {"ok": True, "stopped": stopped}


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


def _strip_flag(args: list[str], flag: str) -> list[str]:
    """Drop ``flag`` (+ value) from an argv-style list."""
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == flag:
            if i + 1 < len(args) and not str(args[i + 1]).startswith("-"):
                i += 2
            else:
                i += 1
            continue
        if a.startswith(flag + "="):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def _resolve_hf_token_for_container() -> str:
    """Return a Hub token only if it authenticates; never inject a known-bad token."""
    try:
        from .autoconfig import _hf_token, hf_token_usable

        tok = _hf_token()
        if not tok:
            return ""
        if hf_token_usable():
            return tok
        return ""
    except Exception:
        # Fallback: env only (legacy path)
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""


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
    tensor_parallel_size: int = 1,
) -> list[str]:
    """Assemble the vLLM CLI from explicit fields + free-form extras. Nothing silent."""
    args: list[str] = [
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(max(1, int(tensor_parallel_size or 1))),
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
    extras = _parse_extra_flags(extra_flags)
    # Drop free-form duplicates of structured fields we already set.
    if mtp:
        extras = _strip_flag(extras, "--speculative-config")
    if quantization.strip():
        extras = _strip_flag(extras, "--quantization")
        extras = _strip_flag(extras, "-q")
    if kv_cache_dtype.strip():
        extras = _strip_flag(extras, "--kv-cache-dtype")
    if moe_backend.strip():
        extras = _strip_flag(extras, "--moe-backend")
    if tool_call_parser.strip():
        extras = _strip_flag(extras, "--tool-call-parser")
    if reasoning_parser.strip():
        extras = _strip_flag(extras, "--reasoning-parser")
    if load_format.strip():
        extras = _strip_flag(extras, "--load-format")
    # Lab envelope owns host/port/tp/util/max-len
    for f in (
        "--host",
        "--port",
        "--tensor-parallel-size",
        "--gpu-memory-utilization",
        "--max-model-len",
    ):
        extras = _strip_flag(extras, f)
    args += extras
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
    tensor_parallel_size: int | None = None,
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

    # Guard: marlin crashes MoE on vLLM 0.25 (NVIDIA 35B-A3B-NVFP4 ModelOpt path)
    if moe_backend.lower() == "marlin":
        drop_marlin = True
        try:
            from .autoconfig import analyze_config, load_local_fallback, _marlin_unsafe_for_checkpoint

            local = load_local_fallback(model)
            if local.get("config"):
                det = analyze_config(local["config"], model)
                drop_marlin = _marlin_unsafe_for_checkpoint(det)
            elif "a3b" in mid or "moe" in mid or quant_l in ("modelopt", "compressed-tensors"):
                drop_marlin = True
        except Exception:
            drop_marlin = True
        if drop_marlin:
            w(
                "SAFETY: dropping --moe-backend marlin "
                "(vLLM 0.25: not supported for unquantized MoE on this checkpoint; leave auto)",
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
        tensor_parallel_size=int(tensor_parallel_size or 1),
    )

    # Multi-node (TP across Sparks): build per-node launch + orchestrate head/workers.
    tp_n = int(tensor_parallel_size or 1)
    if tp_n >= 2:
        from .autoconfig import _cluster_topology, plan_placement, estimate_weights_gib

        topo = _cluster_topology()
        weights = estimate_weights_gib(model, None)
        plan = plan_placement(weights, topo, mode=mode, overlay=None)
        if plan["nodes_available"] < tp_n:
            raise RuntimeError(
                f"TP={tp_n} requested but only {plan['nodes_available']} node(s) online. "
                "Bring the cluster up or lower tensor-parallel-size."
            )
        head = plan.get("head") or {}
        workers = (plan.get("planned_nodes") or [])[1:]
        # strip structured TP/nnodes from extras so the launcher can set them per rank
        clean_args = _strip_structured_flags(vllm_args)
        launch = build_multi_node_launch(
            image=image,
            model=model,
            vllm_args=clean_args,
            env_list=env_list,
            head=head,
            workers=workers,
            nnodes=tp_n,
            port=port,
        )
        if stop_first:
            w("Stopping existing vLLM containers (head + workers)…", 0.05)
            stop_multi_node(log=log)
            stop_all(log=log)
        w(f"Multi-node launch: TP={tp_n} across {tp_n} node(s) on QSFP RoCE", 0.2)
        return _launch_multi_node(launch, port=port, log=log, progress=progress)

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
    hf_token = _resolve_hf_token_for_container()
    if hf_token:
        cmd += [
            "-e",
            f"HF_TOKEN={hf_token}",
            "-e",
            f"HUGGING_FACE_HUB_TOKEN={hf_token}",
        ]
    else:
        w(
            "HF token missing or invalid (whoami failed) — container will fetch public models anonymously",
            0.28,
        )
    cmd += [
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
    # Only inject a token that actually authenticates (bad Bearer → 401 on public too)
    try:
        from .autoconfig import _hf_token, hf_token_usable

        tok = _hf_token() if hf_token_usable() else ""
    except Exception:
        tok = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or ""
        if not tok:
            token_file = Path.home() / ".cache" / "huggingface" / "token"
            if token_file.is_file():
                try:
                    tok = token_file.read_text().strip()
                except OSError:
                    tok = ""
    if tok:
        env["HF_TOKEN"] = tok
        env["HUGGING_FACE_HUB_TOKEN"] = tok
    else:
        env.pop("HF_TOKEN", None)
        env.pop("HUGGING_FACE_HUB_TOKEN", None)
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
