"""Serve / stop model engines (vLLM, SGLang, llama.cpp, TensorFold) from explicit config.

Engine specifics (CLI, image, port, readiness path) live in `engines/`; this module owns
the shared plumbing: docker run (single node, or one container per TP rank over ssh),
readiness with dead-container checks and cancel, and Stop.
"""
from __future__ import annotations

import json
import math
import os
import re
import select
import shlex
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from ..config import DATA_DIR, WORKFLOW_UTIL
from . import engines
from .engines import Engine, Rank, ServeSpec
from .jobs import JobCancelled
from .metadata import available_gib, list_vllm_containers
from .node_probe import DOCKER_PS_FORMAT, ENGINE_LABEL, parse_docker_ps

# In-container HF cache path shared by single-node and multi-node launches.
# Host ~/.cache/huggingface is bind-mounted here; HF_HOME must match.
HF_CACHE_IN_CONTAINER = "/cache/huggingface"
_MULTINODE_STATE = DATA_DIR / "multinode_serve.json"
_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]

READY_POLL_S = 5.0
MULTI_READY_TIMEOUT_S = 30 * 60
# A stalled `docker pull` (SGLang / NGC images are 20-30 GB) gives up after this.
PULL_TIMEOUT_S = 45 * 60
# Remote ranks are checked over ssh every this many polls (30 s), the local head every poll.
WORKER_CHECK_EVERY = 6
_TOKEN_ENV = ("HF_TOKEN=", "HUGGING_FACE_HUB_TOKEN=")


def redact_cmd(parts: list[str]) -> str:
    """A command for LOG LINES: shell-quoted, HF tokens masked. Never execute this string."""
    out = []
    for c in parts:
        if c.startswith(_TOKEN_ENV):
            out.append(c.split("=", 1)[0] + "=***")
        else:
            out.append(shlex.quote(c))
    return " ".join(out)


def _hf_env(hf_token: str | None) -> list[str]:
    """`-e NAME` without a value: docker copies it from its own environment, so the token
    is never on a command line (`ps` shows argv to every user). See _token_env / _ssh_run."""
    if not hf_token:
        return []
    return ["-e", "HF_TOKEN", "-e", "HUGGING_FACE_HUB_TOKEN"]


def _token_env(hf_token: str | None) -> dict[str, str] | None:
    """Environment for a local `docker run` whose argv carries `_hf_env` names."""
    if not hf_token:
        return None
    return {**os.environ, "HF_TOKEN": hf_token, "HUGGING_FACE_HUB_TOKEN": hf_token}


def _ssh_run(host: str, cmd: list[str], *, hf_token: str | None = None, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    """Run `cmd` on `host`. The token travels on stdin into the remote shell's environment,
    never on the ssh or remote command line."""
    remote = shlex.join(cmd)
    if hf_token:
        remote = f'IFS= read -r HF_TOKEN && export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" && {remote}'
    return subprocess.run(
        [*_SSH, host, remote], input=(hf_token + "\n") if hf_token else None,
        capture_output=True, text=True, timeout=timeout,
    )


def _hf_mount() -> list[str]:
    hf = str(Path.home() / ".cache" / "huggingface")
    return ["-v", f"{hf}:{HF_CACHE_IN_CONTAINER}"]


def build_single_node_docker_cmd(
    *,
    engine: Engine,
    spec: ServeSpec,
    image: str,
    env_list: list[str],
    container: str,
    hf_token: str | None = None,
) -> list[str]:
    """docker run argv for a single-node serve. Pure (no subprocess) for tests.

    The API is reachable on 127.0.0.1 only: published on loopback from the bridge
    network, or (host-network engines) bound to loopback by the engine itself. HF cache
    mount + HF_HOME match multi-node (/cache/huggingface).
    """
    cmd: list[str] = [
        "docker", "run", "-d",
        "--name", container,
        "--restart", "no",
        "--label", f"{ENGINE_LABEL}={engine.name}",
        "--gpus", "all",
    ]
    if engine.host_network:
        cmd += ["--network", "host", "--ipc", "host"]
    else:
        cmd += ["--shm-size=32g", "-p", f"127.0.0.1:{spec.port}:{spec.port}"]
    cmd += [*engine.docker_opts, *_hf_mount(), "-e", f"HF_HOME={HF_CACHE_IN_CONTAINER}"]
    for e in env_list:
        cmd += ["-e", e]
    cmd += _hf_env(hf_token)
    return cmd + engine.command(spec, image, None)


def build_multi_node_launch(
    *,
    engine: Engine,
    spec: ServeSpec,
    image: str,
    env_list: list[str],
    head: dict[str, Any],
    workers: list[dict[str, Any]],
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Per-rank docker commands for a TP serve across len(workers)+1 nodes, one GPU each.
    Pure function (no subprocess) so it is fully testable. Every rank gets the HF token."""
    nnodes = 1 + len(workers)
    master_addr = head.get("qsfp_ip") or "127.0.0.1"
    # Environment that is identical across nodes (model/runtime knobs), minus host-IP keys.
    shared_env = [
        e
        for e in env_list
        if not e.startswith(("VLLM_HOST_IP=", "WORKER_VLLM_HOST_IP=", "NODE_RANK=", "MASTER_ADDR="))
    ]

    def rank_cmd(rank: int, node_ip: str | None) -> list[str]:
        cmd = [
            "docker", "run", "-d", "--name", engine.rank_container(rank), "--restart", "no",
            "--label", f"{ENGINE_LABEL}={engine.name}",
            "--gpus", "all", "--network", "host", "--ipc", "host", "--shm-size=32g",
            "--device", "/dev/infiniband",
            # RDMA needs locked (pinned) memory + raw verbs access; without these
            # NCCL fails at init with "unhandled system error".
            "--cap-add", "IPC_LOCK",
            "--ulimit", "memlock=-1:-1",
            *engine.docker_opts,
            *_hf_mount(),
        ]
        for e in shared_env:
            cmd += ["-e", e]
        if node_ip:
            cmd += ["-e", f"VLLM_HOST_IP={node_ip}"]
        cmd += ["-e", f"HF_HOME={HF_CACHE_IN_CONTAINER}", *_hf_env(hf_token)]
        cmd += ["-e", f"NODE_RANK={rank}", "-e", f"MASTER_ADDR={master_addr}"]
        where = Rank(rank=rank, nnodes=nnodes, master_addr=master_addr, master_port=int(engine.master_port or 0))
        return cmd + engine.command(spec, image, where)

    worker_cmds = []
    for idx, wnode in enumerate(workers, start=1):
        worker_cmds.append({
            "node": wnode.get("id") or f"worker{idx}",
            "ssh_host": wnode.get("ssh_host") or wnode.get("id"),
            "rank": idx,
            "container": engine.rank_container(idx),
            "cmd": rank_cmd(idx, wnode.get("qsfp_ip")),
        })
    return {
        "engine": engine.name,
        "head": {"rank": 0, "container": engine.rank_container(0), "cmd": rank_cmd(0, head.get("qsfp_ip"))},
        "workers": worker_cmds,
        "nnodes": nnodes,
        "port": spec.port,
        "model": spec.model,
        "image": image,
    }


def _run_watched(
    argv: list[str], *, what: str, cancel: threading.Event | None = None, timeout_s: float = PULL_TIMEOUT_S,
) -> tuple[int, str]:
    """Run a long command (an image pull) that Cancel can stop: killed and JobCancelled when
    `cancel` is set, killed and TimeoutError after `timeout_s`. Returns (rc, output)."""
    p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            out, _ = p.communicate(timeout=1.0)
            return p.returncode, out or ""
        except subprocess.TimeoutExpired:
            cancelled = cancel is not None and cancel.is_set()
            if not cancelled and time.monotonic() < deadline:
                continue
            p.kill()
            p.communicate()
            if cancelled:
                raise JobCancelled(f"cancelled during {what}") from None
            raise TimeoutError(f"{what} gave up after {int(timeout_s // 60)} min") from None


def _ensure_image_present(
    image: str,
    *,
    log: Any = None,
    progress: Callable | None = None,
    cancel: threading.Event | None = None,
) -> None:
    """Pull the serve image if it is not already present locally."""
    image = (image or "").strip()
    if not image:
        return

    def w(msg: str, p: float = 0.12) -> None:
        if log:
            log.write(msg)
        if progress:
            progress(p, msg)

    probe = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        return
    w(f"Image {image} not local — pulling…", 0.12)
    rc, out = _run_watched(["docker", "pull", image], what=f"docker pull {image}", cancel=cancel)
    if rc != 0:
        raise RuntimeError(f"Failed to pull required image {image}: {out.strip()[-800:]}")
    w(f"Pulled {image}", 0.18)


def _ensure_images(
    launch: dict[str, Any],
    *,
    log: Any = None,
    progress: Callable | None = None,
    cancel: threading.Event | None = None,
) -> None:
    """The image on the head and on every worker rank, before anything is stopped."""
    image = (launch.get("image") or "").strip()
    if not image:
        return
    _ensure_image_present(image, log=log, progress=progress, cancel=cancel)
    iq = shlex.quote(image)
    for wc in launch["workers"]:
        host = wc.get("ssh_host")
        if not host:
            continue
        if log:
            log.write(f"Ensuring image on worker {wc.get('node')} ({host})…")
        if progress:
            progress(0.15, f"Ensuring image on worker {wc.get('node')} ({host})…")
        # Quote image so tags/repos with special shell chars are safe over SSH.
        rc, out = _run_watched(
            [*_SSH, host, f"docker image inspect {iq} >/dev/null 2>&1 || docker pull {iq}"],
            what=f"docker pull {image} on {host}", cancel=cancel,
        )
        if rc != 0:
            raise RuntimeError(f"Failed to pull {image} on {host}: {out.strip()[-800:]}")


# ─── readiness ────────────────────────────────────────────────────────────────


def _container_dead(name: str, host: str | None = None) -> str | None:
    """None while the container runs; else its docker state ("exited 1", "missing").
    An unanswered ssh probe is not a verdict (None): the next check decides."""
    inspect = ["docker", "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", name]
    try:
        if host:
            st = subprocess.run([*_SSH, host, shlex.join(inspect)], capture_output=True, text=True, timeout=20)
        else:
            st = subprocess.run(inspect, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return None
    if st.returncode != 0:
        # Only docker saying the container is gone is a verdict; an ssh failure (255), a
        # daemon hiccup or a permission error is not — the next check decides.
        return "missing" if "no such" in (st.stderr or "").lower() else None
    status = (st.stdout or "").strip()
    return None if not status or status.startswith("running") else status


def _tail_logs(name: str, n: int = 120, host: str | None = None) -> str:
    cmd = ["docker", "logs", "--tail", str(n), name]
    try:
        if host:
            return subprocess.check_output([*_SSH, host, shlex.join(cmd) + " 2>&1"], text=True, timeout=30)
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=30)
    except Exception as e:
        return str(e)


def _answers(url: str) -> bool:
    """200 from the readiness path. 503 (SGLang starting, llama.cpp loading) and a refused
    connection (TensorFold binds only once loaded) both mean "still loading"."""
    try:
        urllib.request.urlopen(url, timeout=3)
        return True
    except Exception:
        return False


def wait_ready(
    engine: Engine,
    port: int,
    *,
    container: str,
    remote: list[tuple[str, str]] | tuple = (),
    timeout_s: float,
    log: Any = None,
    progress: Callable | None = None,
    cancel: threading.Event | None = None,
) -> None:
    """Poll the engine's readiness path until it answers 200.

    Fails fast — with the container's last log lines — when the local container (every
    poll) or a remote rank (every WORKER_CHECK_EVERY polls) is no longer running. Raises
    JobCancelled when `cancel` is set, TimeoutError after `timeout_s`. Progress moves with
    the elapsed time and the container's latest log line.
    """
    url = f"http://127.0.0.1:{port}{engine.ready_path}"
    t0 = time.monotonic()
    tick = 0
    last_line = ""
    last_report = -math.inf
    while True:
        if cancel is not None and cancel.is_set():
            raise JobCancelled("cancelled while the model was loading")
        dead = _container_dead(container)
        if dead is not None:
            logs = _tail_logs(container)
            if log:
                log.write(f"container {container} exited early ({dead})")
                log.write(logs)
            raise RuntimeError(
                f"{engine.label} container {container} exited ({dead}) before {engine.ready_path} answered.\n"
                f"--- docker logs ---\n{logs[-4000:]}"
            )
        if remote and tick % WORKER_CHECK_EVERY == 0:
            for host, name in remote:
                dead = _container_dead(name, host)
                if dead is not None:
                    logs = _tail_logs(name, host=host)
                    if log:
                        log.write(f"worker {name} on {host} exited early ({dead})")
                        log.write(logs)
                    raise RuntimeError(
                        f"{engine.label} worker {name} on {host} exited ({dead}) before the head was ready.\n"
                        f"--- docker logs ({host}) ---\n{logs[-4000:]}"
                    )
        if _answers(url):
            return
        elapsed = time.monotonic() - t0
        if elapsed > timeout_s:
            logs = _tail_logs(container)
            if log:
                log.write(logs)
            raise TimeoutError(
                f"Timeout after {int(elapsed)} s waiting for {url}\n--- docker logs ---\n{logs[-4000:]}"
            )
        line = (_tail_logs(container, 1).strip().splitlines() or [""])[-1][:160]
        if progress and (line != last_line or elapsed - last_report >= 30):
            # 0.35 → ~0.95 over a typical load; the message carries the real elapsed time.
            p = 0.35 + 0.6 * (1 - math.exp(-elapsed / 300))
            progress(round(p, 3), f"loading · {int(elapsed)} s · {line or 'no output yet'}")
            last_line, last_report = line, elapsed
        tick += 1
        if cancel is not None:
            cancel.wait(READY_POLL_S)
        else:
            time.sleep(READY_POLL_S)


# ─── multi-node ───────────────────────────────────────────────────────────────


def _write_multinode_state(launch: dict[str, Any]) -> None:
    _MULTINODE_STATE.write_text(
        json.dumps(
            {
                "engine": launch.get("engine"),
                "model": launch["model"],
                "image": launch["image"],
                "nnodes": launch["nnodes"],
                "port": launch["port"],
                "head": launch["head"]["container"],
                "workers": [
                    {"ssh_host": w["ssh_host"], "node": w["node"], "rank": w["rank"], "container": w["container"]}
                    for w in launch["workers"]
                ],
            },
            indent=2,
        )
    )


def _rm_remote(host: str, name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*_SSH, host, f"docker rm -f {shlex.quote(name)}"], capture_output=True, text=True, timeout=60
    )


def _launch_multi_node(
    launch: dict[str, Any],
    *,
    engine: Engine = engines.VLLM,
    hf_token: str | None = None,
    log: Any = None,
    progress: Callable | None = None,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    """Start the worker ranks (ssh), then the head, then wait for the head's readiness.

    Images are already on every node (_ensure_images). The state file is written before
    anything starts, so Stop can always find every rank; any failure (a rank that will
    not start or dies, timeout, cancel) tears down whatever was started — a stray rank
    would otherwise pin its node's memory.
    """

    def w(msg: str, p: float = 0.3) -> None:
        if log:
            log.write(msg)
        if progress:
            progress(p, msg)

    nnodes = launch["nnodes"]
    port = launch["port"]
    head_name = launch["head"]["container"]
    try:
        _write_multinode_state(launch)
    except OSError as e:
        w(f"could not record multi-node state ({e}); Stop still finds ranks by label")
    try:
        # Stale per-rank containers (exited ranks persist: --restart no, no --rm) would make
        # `docker run --name` fail on a name conflict.
        subprocess.run(["docker", "rm", "-f", head_name], capture_output=True, text=True, timeout=60)
        for wc in launch["workers"]:
            _rm_remote(wc["ssh_host"], wc["container"])

        # 1) worker ranks first (TensorFold requires rank 1 up before rank 0), over ssh.
        #    Every rank gets the real token (on stdin / in the environment, never argv).
        for wc in launch["workers"]:
            host = wc["ssh_host"]
            w(f"Starting worker rank {wc['rank']} on {wc['node']} ({host})…")
            if log:
                log.write(f"$ ssh {host} {redact_cmd(wc['cmd'])}")
            r = _ssh_run(host, wc["cmd"], hf_token=hf_token)
            if r.returncode != 0:
                raise RuntimeError(f"worker {wc['node']} failed to start: {r.stderr or r.stdout}")
            w(f"worker {wc['node']} up")

        # 2) head (rank 0 serves the API)
        w("Starting head (rank 0, API)…", 0.5)
        if log:
            log.write(f"$ {redact_cmd(launch['head']['cmd'])}")
        hr = subprocess.run(launch["head"]["cmd"], capture_output=True, text=True, timeout=120, env=_token_env(hf_token))
        if hr.returncode != 0:
            raise RuntimeError(f"head failed to start: {hr.stderr or hr.stdout}")

        # 3) readiness on the head; ranks checked for crashes while it loads
        w(f"Waiting for {engine.ready_path} on head (multi-node load can take 15–30+ min)…", 0.6)
        wait_ready(
            engine, port,
            container=head_name,
            remote=[(wc["ssh_host"], wc["container"]) for wc in launch["workers"]],
            timeout_s=MULTI_READY_TIMEOUT_S,
            log=log, progress=progress, cancel=cancel,
        )
    except BaseException:
        w("Launch failed — removing every rank…", 0.9)
        _teardown_ranks(launch)
        raise
    w(f"Multi-node {engine.label} serve ready on :{port} ({nnodes} nodes)", 1.0)
    return {"ok": True, "multi_node": True, "engine": engine.name, "nnodes": nnodes, "model": launch["model"], "port": port}


def _teardown_ranks(launch: dict[str, Any]) -> None:
    """Best effort: remove every rank of `launch` (local head, workers over ssh) and its state."""
    for wc in launch["workers"]:
        try:
            _rm_remote(wc["ssh_host"], wc["container"])
        except Exception:
            pass
    subprocess.run(["docker", "rm", "-f", launch["head"]["container"]], capture_output=True, text=True, timeout=60)
    _MULTINODE_STATE.unlink(missing_ok=True)


def stop_multi_node(log: Any = None) -> dict[str, Any]:
    """Tear down a recorded multi-node serve across head + workers (state file)."""
    stopped: list[str] = []
    try:
        st = json.loads(_MULTINODE_STATE.read_text())
    except Exception:
        return {"ok": False, "reason": "no multinode state"}
    for wr in st.get("workers") or []:
        host = wr.get("ssh_host")
        if host:
            name = wr.get("container") or f"spark-vllm-n{wr.get('rank', 1)}"
            try:
                _rm_remote(host, name)
            except subprocess.TimeoutExpired:
                if log:
                    log.write(f"{host}: docker rm -f {name} timed out")
                continue
            stopped.append(f"{host}:{name}")
    head = st.get("head") or "spark-vllm-n0"
    subprocess.run(["docker", "rm", "-f", head], capture_output=True, text=True, timeout=60)
    stopped.append(head)
    try:
        _MULTINODE_STATE.unlink(missing_ok=True)
    except Exception:
        pass
    return {"ok": True, "stopped": stopped}


# ─── stop ─────────────────────────────────────────────────────────────────────


def _stoppable(c: dict[str, Any]) -> bool:
    """Every container L.A.I.L launched (any state); a serve launched by hand only while Up."""
    return bool(c.get("lail")) or "up" in str(c.get("status", "")).lower()


def stop_cluster_remote(log: Any = None) -> dict[str, Any]:
    """Stop serve containers (any engine) on the non-local cluster nodes over ssh.

    Ground truth is live docker on each node, judged by the same predicate as this
    host (`node_probe.parse_docker_ps`: label, serve command or engine image) — not the
    container name, and not multinode_serve.json (missing after engine restarts or
    manual launches).
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
                [*_SSH, str(host), f"docker ps -a --no-trunc --format {shlex.quote(DOCKER_PS_FORMAT)}"],
                capture_output=True, text=True, timeout=30,
            )
        except Exception as e:
            errors.append(f"{host}: list failed: {e}")
            continue
        if listed.returncode != 0:
            errors.append(f"{host}: list exit {listed.returncode}: {(listed.stderr or '')[:200]}")
            continue
        names = [c["name"] for c in parse_docker_ps(listed.stdout) if _stoppable(c)]
        if not names:
            continue
        if log:
            log.write(f"Remote {host}: stopping {names}")
        for name in names:
            try:
                rm = _rm_remote(str(host), name)
            except Exception as e:
                errors.append(f"{host}:{name}: {e}")
                continue
            if rm.returncode == 0:
                stopped.append(f"{host}:{name}")
            else:
                errors.append(f"{host}:{name}: rm exit {rm.returncode}")
    return {"ok": not errors, "stopped": stopped, "errors": errors}


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

    # 2) Always live-discover remote ranks. State file is not required.
    try:
        rem = stop_cluster_remote(log=log)
        if rem.get("stopped"):
            w(f"Stopped remote serves: {rem.get('stopped')}")
            for item in rem["stopped"]:
                if item not in stopped:
                    stopped.append(item)
        if rem.get("errors") and log:
            log.write(f"remote stop warnings: {rem['errors']}")
    except Exception as e:
        if log:
            log.write(f"remote cluster stop failed: {e}")

    # 3) This host.
    names = [c["name"] for c in list_vllm_containers() if _stoppable(c)]
    w(f"Stopping: {names}")
    for n in names:
        subprocess.run(["docker", "rm", "-f", n], capture_output=True, text=True, timeout=60)
        if n not in stopped:
            stopped.append(n)
    if progress:
        progress(1.0, "stopped")
    return {"ok": True, "stopped": stopped}


# ─── start ────────────────────────────────────────────────────────────────────


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


def ignored_fields(engine: Engine, spec: ServeSpec) -> list[str]:
    """Fields set in the request that `engine` has no translation for (reported, not dropped silently)."""
    default = ServeSpec(model=spec.model, port=spec.port)
    return [
        f for f in ServeSpec.__dataclass_fields__
        if f not in engine.fields and f not in engines.COMMON_FIELDS
        and getattr(spec, f) != getattr(default, f)
    ]


def serve_model(
    *,
    model: str,
    engine: str | None = None,
    mode: str | None = None,
    util: float | None = None,
    max_model_len: int | None = None,
    port: int | None = None,
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
    mtp_moe_backend: str = "",
    load_format: str = "",
    enable_chunked_prefill: bool = False,
    enable_prefix_caching: bool = False,
    extra_flags: str = "",
    tensor_parallel_size: int | None = None,
    stop_first: bool = True,
    log: Any = None,
    progress: Callable | None = None,
    cancel: threading.Event | None = None,
    # legacy no-ops so old clients don't crash
    profile: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    def w(msg: str, p: float = 0.1) -> None:
        if log:
            log.write(msg)
        if progress:
            progress(p, msg)

    eng = engines.get(engine)
    port = int(port or eng.default_port)
    image = (image or "").strip() or eng.image()

    # Resolve TP once — fits gate and launch must use the same value (no plan
    # fallback only in the gate, which previously greenlit multi-node fit while
    # launch still ran single-node).
    tp_n = max(1, int(tensor_parallel_size or 1))
    if eng.max_tp is not None and tp_n > eng.max_tp:
        raise RuntimeError(f"{eng.label} supports at most TP={eng.max_tp} here (requested {tp_n}).")

    # Hard gate: refuse Start when weights cannot fit the online cluster (P0.3).
    # Shares check_serve_loadability with recommend() so Start and Auto-configure agree.
    try:
        from .autoconfig import (
            _cluster_topology,
            _resolved_node_ram_gib,
            check_serve_loadability,
            estimate_weights_gib,
            load_local_fallback,
            plan_placement,
        )

        topo = _cluster_topology()
        local = load_local_fallback(model)
        weights = estimate_weights_gib(model, local.get("config"))
        plan = plan_placement(weights, topo, mode=mode, overlay=None)
        n_avail = int(plan.get("nodes_available") or 1)
        n_use = min(tp_n, n_avail)
        node_ram = _resolved_node_ram_gib(plan.get("node_ram_gib"))
        reserve = float(plan.get("reserve_gib") or 15.0)
        fits_ok, msg = check_serve_loadability(
            mode=mode,
            weights_gib=weights,
            node_ram_gib=node_ram,
            nodes_used=n_use,
            util=float(util if util is not None else WORKFLOW_UTIL),
            reserve_gib=reserve,
        )
        if not fits_ok:
            raise RuntimeError(
                f"SERVE BLOCKED: {msg or f'weights (~{weights} GiB) do not fit {n_avail} Spark(s)'}."
                " Add nodes or pick a smaller checkpoint."
            )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"SERVE BLOCKED: fit-gate probe failed ({e}). Refusing Start."
        ) from e

    env_list = _normalize_docker_env(docker_env)

    # Guard: flashinfer_b12x crashed mixed FP8+NVFP4 on vLLM 0.25.x. On ≥0.27
    # (lab default) Unsloth's Spark recipe *requires* it — share autoconfig policy.
    moe_backend = (moe_backend or "").strip()
    quant_l = (quantization or "").strip().lower()
    mid = (model or "").lower()
    if moe_backend == "flashinfer_b12x":
        drop_b12x = False
        try:
            from .autoconfig import (
                analyze_config,
                load_local_fallback,
                _flashinfer_b12x_unsafe_for_checkpoint,
            )

            local = load_local_fallback(model)
            det = analyze_config(local.get("config") or {}, model)
            drop_b12x = _flashinfer_b12x_unsafe_for_checkpoint(det, image)
        except Exception:
            drop_b12x = False
        if drop_b12x:
            w(
                "SAFETY: dropping --moe-backend flashinfer_b12x "
                "(not supported for FP8 MoE on this image; leave auto)",
                0.15,
            )
            moe_backend = ""

    # Guard: marlin crashes some MoE paths (Qwen ModelOpt on older vLLM). Nemotron
    # hybrid Spark recipes *require* marlin — never drop based on "a3b" alone.
    if moe_backend.lower() == "marlin":
        drop_marlin = True
        try:
            from .autoconfig import (
                analyze_config,
                load_local_fallback,
                _marlin_unsafe_for_checkpoint,
            )

            local = load_local_fallback(model)
            # Family/id detection works with empty config (Nemotron keep without cache).
            det = analyze_config(local.get("config") or {}, model)
            drop_marlin = _marlin_unsafe_for_checkpoint(det, image)
        except Exception:
            # Fail closed for unknown MoE-ish ids only when family is not Nemotron.
            drop_marlin = "nemotron" not in mid and (
                "a3b" in mid or "moe" in mid or quant_l in ("modelopt", "compressed-tensors")
            )
        if drop_marlin:
            w(
                "SAFETY: dropping --moe-backend marlin "
                "(unsupported for this MoE path on the selected image; leave auto)",
                0.15,
            )
            moe_backend = ""

    spec = ServeSpec(
        model=model,
        port=port,
        util=util,
        max_model_len=max_model_len,
        tensor_parallel_size=tp_n,
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
        mtp_moe_backend=mtp_moe_backend,
        load_format=load_format,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_prefix_caching=enable_prefix_caching,
        extra_flags=extra_flags,
    )
    ignored = ignored_fields(eng, spec)
    if ignored:
        w(f"{eng.label} has no flag for: {', '.join(ignored)} — not passed (use extra flags)", 0.16)

    hf_token = _resolve_hf_token_for_container()
    if not hf_token:
        w(
            "HF token missing or invalid (whoami failed) — container will fetch public models anonymously",
            0.19,
        )

    def not_cancelled() -> None:
        # Everything before this point (download, image pulls) leaves the running serve alone.
        if cancel is not None and cancel.is_set():
            raise JobCancelled("cancelled before the running serve was stopped")

    # Multi-node (TP across Sparks): one container per rank, workers over ssh.
    if tp_n >= 2:
        if eng.master_port is None:
            raise RuntimeError(f"{eng.label} has no multi-node tensor parallel here — serve it on one node.")
        from .autoconfig import (
            _cluster_topology,
            estimate_weights_gib,
            load_local_fallback,
            plan_placement,
        )

        topo = _cluster_topology()
        local = load_local_fallback(model)
        weights = estimate_weights_gib(model, local.get("config"))
        plan = plan_placement(weights, topo, mode=mode, overlay=None)
        if plan["nodes_available"] < tp_n:
            raise RuntimeError(
                f"TP={tp_n} requested but only {plan['nodes_available']} node(s) online. "
                "Bring the cluster up or lower tensor-parallel-size."
            )
        head = plan.get("head") or {}
        # The requested TP decides how many nodes take a rank (the plan's own
        # planned_nodes may be fewer: a small model "needs" one node).
        workers = (topo.get("workers") or [])[: tp_n - 1]
        launch = build_multi_node_launch(
            engine=eng, spec=spec, image=image, env_list=env_list,
            head=head, workers=workers, hf_token=hf_token,
        )
        _ensure_images(launch, log=log, progress=progress, cancel=cancel)
        not_cancelled()
        if stop_first:
            w("Stopping existing serve containers (head + workers)…", 0.05)
            stop_multi_node(log=log)
            stop_all(log=log)
        w(f"Multi-node {eng.label} launch: TP={tp_n} across {tp_n} node(s) on QSFP RoCE", 0.2)
        return _launch_multi_node(launch, engine=eng, hf_token=hf_token, log=log, progress=progress, cancel=cancel)

    _ensure_image_present(image, log=log, progress=progress, cancel=cancel)
    not_cancelled()
    if stop_first:
        w("Stopping existing serve containers…", 0.05)
        stop_all(log=log)

    container = eng.container
    w(f"Launching {eng.label} in docker ({container})…", 0.2)
    subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=60)
    cmd = build_single_node_docker_cmd(
        engine=eng, spec=spec, image=image, env_list=env_list, container=container, hf_token=hf_token,
    )
    w(f"image={image}", 0.25)
    w(f"docker_env={env_list}", 0.26)
    w(f"$ {redact_cmd(cmd)}", 0.3)

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=_token_env(hf_token))
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout or "docker run failed")
    if log:
        log.write((r.stdout or "").strip())
        log.write(f"Waiting for {eng.ready_path} (up to {eng.ready_timeout_s // 60} min)…")
    try:
        wait_ready(
            eng, port, container=container, timeout_s=eng.ready_timeout_s,
            log=log, progress=progress, cancel=cancel,
        )
    except (JobCancelled, TimeoutError):
        # Still loading and holding memory: remove it. An exited container is kept for `docker logs`.
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=60)
        raise

    avail = available_gib()
    w(f"{eng.label} API ready on :{port}. available_gib={avail}", 1.0)
    return {
        "ok": True,
        "engine": eng.name,
        "model": model,
        "container": container,
        "port": port,
        "util": util,
        "max_model_len": max_model_len,
        "available_gib": avail,
        "image": image,
        "docker_env": env_list,
        "args": eng.command(spec, image, None),
    }


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
    # hf is Python: unbuffered so tqdm \r frames reach the pipe before EOF.
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
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


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _split_cli_frames(buf: bytes) -> tuple[list[str], bytes]:
    """Split a byte buffer on newlines and tqdm carriage returns."""
    frames: list[str] = []
    while True:
        n_nl = buf.find(b"\n")
        n_cr = buf.find(b"\r")
        if n_nl < 0 and n_cr < 0:
            break
        if n_nl < 0:
            cut = n_cr
        elif n_cr < 0:
            cut = n_nl
        else:
            cut = min(n_nl, n_cr)
        raw, buf = buf[:cut], buf[cut + 1 :]
        if cut == n_cr and buf.startswith(b"\n"):
            buf = buf[1:]
        text = _ANSI_RE.sub("", raw.decode("utf-8", "replace")).strip()
        if text:
            frames.append(text)
    return frames, buf


def _download_pct_from_line(line: str) -> int | None:
    m = re.search(r"(\d+)\s*%", line)
    if not m:
        return None
    n = int(m.group(1))
    if 0 <= n <= 100:
        return n
    return None


def _pump_hf_output(
    proc: subprocess.Popen[bytes],
    *,
    log: Any = None,
    progress: Callable | None = None,
    timeout: float = 7200,
) -> tuple[int, list[str]]:
    """Read hf stdout live (tqdm uses \\r). Throttle progress-bar log lines."""
    assert proc.stdout is not None
    buf = b""
    lines: list[str] = []
    last_pct = 0.05
    last_log = 0.0
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() > deadline:
            proc.kill()
            raise TimeoutError("hf download timed out")
        ready, _, _ = select.select([proc.stdout], [], [], 0.5)
        chunk = b""
        if ready:
            chunk = proc.stdout.read(8192)
        if not chunk:
            if proc.poll() is not None:
                rest = proc.stdout.read() or b""
                frames, _ = _split_cli_frames(buf + rest + b"\n")
                for line in frames:
                    lines.append(line)
                    if log:
                        log.write(line)
                break
            continue
        buf += chunk
        frames, buf = _split_cli_frames(buf)
        now = time.monotonic()
        for line in frames:
            lines.append(line)
            pct = _download_pct_from_line(line)
            if pct is None or now - last_log >= 1.0:
                if log:
                    log.write(line)
                last_log = now
            if pct is not None and progress:
                p = max(0.05, min(0.95, pct / 100.0))
                if p - last_pct >= 0.02:
                    progress(p, f"download {pct}%")
                    last_pct = p
    return int(proc.wait()), lines


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
    # Binary + \r split: tqdm progress is carriage-return frames, not newlines.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=0,
    )
    try:
        code, lines = _pump_hf_output(proc, log=log, progress=progress, timeout=7200)
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
