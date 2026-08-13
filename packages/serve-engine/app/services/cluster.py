"""Cluster health for L.A.I.L Status (source of truth).

Default topology is this machine as probed (hostname, LAN, Tailscale, RoCE)
plus any RoCE L2 neighbors that answer ping. LAIL_CLUSTER_JSON or
gitignored data/cluster.json still override.

Probes:
  - local node via metadata + live NICs
  - remote nodes via passwordless SSH + a small JSON collector
  - QSFP fabric reachability between discovered interconnect IPs
  - whether a model is loaded on each node and whether multi-node looks aligned
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
from typing import Any

from . import metadata

log = logging.getLogger(__name__)

# Used only if live NIC/hostname probes fail.
_FALLBACK_CLUSTER = {
    "name": "local",
    "nodes": [
        {
            "id": "local",
            "label": "this host",
            "role": "head",
            "local": True,
            "vllm_url": "http://127.0.0.1:8000",
        },
    ],
}

_SKIP_IFACES = {"lo", "docker0", "tailscale0"}
_SKIP_IFACE_PREFIXES = ("br-", "veth", "virbr", "cni", "flannel", "wg")

_REMOTE_PROBE_PY = r"""
import json, platform, re, subprocess, urllib.request

def run(cmd, t=8):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=t)
    except Exception:
        return ""

def avail():
    out = run(["free", "-g"])
    for line in out.splitlines():
        if line.startswith("Mem:"):
            p = line.split()
            if len(p) >= 7:
                try: return float(p[6])
                except: return None
    return None

def mem_total():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except Exception:
        return None
    return None

gpu = "unknown"
smi = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
if smi.strip():
    gpu = smi.strip().split("\n")[0].strip()

containers = []
dout = run(["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}"])
for line in dout.splitlines():
    if not line.strip():
        continue
    parts = line.split("\t")
    if len(parts) < 3:
        continue
    name, status, image = parts[0], parts[1], parts[2]
    if re.search(r"vllm|spark-vllm|ray|deepseek|qwen|brain|llama", name, re.I) or "vllm" in image.lower() or "ray" in image.lower():
        containers.append({"name": name, "status": status, "image": image})

model_id = None
healthy = False
models = []
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2.5) as r:
        body = json.loads(r.read().decode())
        models = body.get("data") or []
        if models:
            model_id = models[0].get("id")
            healthy = True
except Exception:
    pass

# detect TP from running container cmd
tp = None
ray_like = False
for c in containers:
    if "ray" in c["name"].lower() or "ray" in c["image"].lower():
        ray_like = True
    insp = run(["docker", "inspect", "--format", "{{json .Config.Cmd}}", c["name"]], t=5)
    if not insp.strip():
        continue
    try:
        cmd = json.loads(insp)
    except Exception:
        cmd = []
    joined = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
    m = re.search(r"--tensor-parallel-size[=\s]+(\d+)", joined)
    if m:
        tp = int(m.group(1))

# fabric carrier/speed for the node-configured iface (injected as qsfp_if)
carrier = None
speed = None
try:
    if qsfp_if:
        carrier = int(open(f"/sys/class/net/{qsfp_if}/carrier").read().strip())
except Exception:
    pass
try:
    if qsfp_if:
        speed = int(open(f"/sys/class/net/{qsfp_if}/speed").read().strip())
except Exception:
    pass

ib = run(["ibdev2netdev"])
up_ifs = []
for line in ib.splitlines():
    if "(Up)" in line:
        parts = line.split()
        if len(parts) >= 5:
            up_ifs.append(parts[4].strip("()"))

print(json.dumps({
    "hostname": platform.node(),
    "reachable": True,
    "gpu_sku": gpu,
    "ram_gib": mem_total(),
    "available_gib": avail(),
    "model_id": model_id,
    "endpoint_healthy": healthy,
    "models": [{"id": m.get("id")} for m in models[:5]],
    "containers": containers,
    "tensor_parallel_size": tp,
    "ray_hint": ray_like,
    "qsfp_if": qsfp_if,
    "qsfp_carrier": carrier,
    "qsfp_speed_mbps": speed if speed and speed > 0 else None,
    "roce_up_ifs": up_ifs,
    "tailscale_ip": run(["tailscale", "ip", "-4"]).strip().split("\n")[0] if run(["tailscale", "ip", "-4"]).strip() else None,
}))
"""


def _run(cmd: list[str], timeout: float = 12) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as e:
        return 1, "", str(e)


def _parse_ip_addrs(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for line in (text or "").splitlines():
        m = re.match(r"^\d+:\s+(\S+)\s+inet\s+([\d.]+)/", line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def _parse_roce_up(text: str) -> list[str]:
    up: list[str] = []
    for line in (text or "").splitlines():
        m = re.search(r"==>\s+(\S+)\s+\(Up\)", line)
        if m:
            up.append(m.group(1))
    return up


def _parse_neigh(text: str) -> list[str]:
    ips: list[str] = []
    for line in (text or "").splitlines():
        parts = line.split()
        if not parts:
            continue
        ip = parts[0]
        if not re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", ip):
            continue
        blob = " ".join(parts[1:]).upper()
        if "FAILED" in blob or "INCOMPLETE" in blob:
            continue
        if ip.startswith("169.254."):
            continue
        ips.append(ip)
    return ips


def _iface_skipped(name: str) -> bool:
    n = (name or "").strip()
    if n in _SKIP_IFACES:
        return True
    return any(n.startswith(p) for p in _SKIP_IFACE_PREFIXES)


def _detect_local_net() -> dict[str, Any]:
    """LAN / Tailscale / RoCE from this host. No baked lab addresses."""
    code, addr_txt, _ = _run(["ip", "-4", "-o", "addr"], timeout=4)
    addrs = _parse_ip_addrs(addr_txt) if code == 0 else []
    code, ib_txt, _ = _run(["ibdev2netdev"], timeout=4)
    roce_up = _parse_roce_up(ib_txt) if code == 0 else []

    by_iface: dict[str, str] = {}
    for iface, ip in addrs:
        by_iface.setdefault(iface, ip)

    qsfp_if = next((i for i in roce_up if i in by_iface), None)
    qsfp_ip = by_iface.get(qsfp_if) if qsfp_if else None

    lan_ip = None
    for iface, ip in addrs:
        if _iface_skipped(iface) or iface == qsfp_if:
            continue
        lan_ip = ip
        break

    ts_ip = by_iface.get("tailscale0")
    if not ts_ip:
        code, ts_out, _ = _run(["tailscale", "ip", "-4"], timeout=4)
        if code == 0 and ts_out.strip():
            ts_ip = ts_out.strip().splitlines()[0]

    return {
        "lan_ip": lan_ip,
        "tailscale_ip": ts_ip,
        "qsfp_if": qsfp_if,
        "qsfp_ip": qsfp_ip,
    }


def _parse_ssh_config(text: str) -> dict[str, str]:
    """Map HostName/IP -> first Host alias."""
    out: dict[str, str] = {}
    aliases: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, val = line.partition(" ")
        key_l = key.lower()
        if key_l == "host":
            aliases = [a for a in val.split() if a and a != "*"]
            continue
        if key_l == "hostname" and aliases:
            host = val.strip()
            if host and host not in out:
                out[host] = aliases[0]
    return out


def _ssh_alias_for_ip(ip: str) -> str | None:
    path = os.path.expanduser("~/.ssh/config")
    try:
        with open(path, encoding="utf-8") as f:
            return _parse_ssh_config(f.read()).get(ip)
    except OSError:
        return None


def _hostname_for_ip(ip: str) -> str | None:
    code, out, _ = _run(["getent", "hosts", ip], timeout=3)
    if code == 0 and out.strip():
        parts = out.split()
        if len(parts) >= 2:
            return parts[1].split(".")[0]
    return _ssh_alias_for_ip(ip)


def _discover_fabric_peers(local: dict[str, Any]) -> list[dict[str, Any]]:
    """RoCE L2 neighbors that answer ping become remote nodes."""
    qsfp_if = local.get("qsfp_if")
    self_ip = local.get("qsfp_ip")
    if not qsfp_if:
        return []
    code, neigh_txt, _ = _run(["ip", "neigh", "show", "dev", str(qsfp_if)], timeout=4)
    if code != 0:
        return []
    peers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ip in _parse_neigh(neigh_txt):
        if ip == self_ip or ip in seen:
            continue
        seen.add(ip)
        ping = _ping_ok(ip, 1.0)
        if not ping.get("ok"):
            continue
        name = _hostname_for_ip(ip)
        if name == local.get("id"):
            continue
        ssh_host = name or ip
        node_id = name or ip.replace(".", "-")
        peers.append(
            {
                "id": node_id,
                "label": node_id,
                "role": "worker",
                "local": False,
                "ssh_host": ssh_host,
                "qsfp_ip": ip,
                "qsfp_if": qsfp_if,
                "vllm_url": "http://127.0.0.1:8000",
            }
        )
    return peers


def _default_cluster() -> dict[str, Any]:
    host = (platform.node() or "local").split(".")[0] or "local"
    net = _detect_local_net()
    local: dict[str, Any] = {
        "id": host,
        "label": host,
        "role": "head",
        "local": True,
        "vllm_url": "http://127.0.0.1:8000",
    }
    for k, v in net.items():
        if v:
            local[k] = v
    nodes = [local]
    try:
        for peer in _discover_fabric_peers(local):
            if peer.get("id") and peer["id"] != host:
                nodes.append(peer)
    except Exception as e:
        log.warning("fabric peer discovery failed (%s)", e)
    if not nodes:
        return json.loads(json.dumps(_FALLBACK_CLUSTER))
    name = host if len(nodes) == 1 else f"{host}-lab"
    return {"name": name, "nodes": nodes}


def _parse_cluster_dict(data: Any, *, source: str) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        log.warning("%s is not a JSON object; using single local node", source)
        return None
    nodes = data.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        log.warning("%s missing nodes; using single local node", source)
        return None
    return data


def _cluster_file_path():
    from ..config import DATA_DIR

    return DATA_DIR / "cluster.json"


def _load_cluster_config() -> dict[str, Any]:
    raw = os.environ.get("LAIL_CLUSTER_JSON", "").strip()
    if raw:
        try:
            parsed = _parse_cluster_dict(json.loads(raw), source="LAIL_CLUSTER_JSON")
        except json.JSONDecodeError as e:
            log.warning("LAIL_CLUSTER_JSON is invalid JSON (%s); using single local node", e)
            return _default_cluster()
        return parsed if parsed else _default_cluster()
    try:
        path = _cluster_file_path()
        if path.is_file():
            parsed = _parse_cluster_dict(json.loads(path.read_text()), source=str(path))
            if parsed:
                return parsed
    except Exception as e:
        log.warning("cluster.json unreadable (%s); using single local node", e)
    return _default_cluster()


def _node_is_local(node: dict[str, Any], n_cfg: int) -> bool:
    """Honor explicit local flags only. Hostname must not steal a remote's IPs."""
    if "local" in node:
        return bool(node.get("local"))
    return n_cfg == 1 and not node.get("ssh_host")


def _remote_probe_script(node: dict[str, Any]) -> str:
    prefix = f"qsfp_if = {json.dumps(str(node.get('qsfp_if') or ''))}\n"
    return prefix + _REMOTE_PROBE_PY


def _ping_ok(ip: str, timeout_s: float = 1.0) -> dict[str, Any]:
    if not ip:
        return {"ok": False, "error": "no_ip"}
    code, out, err = _run(["ping", "-c", "1", "-W", str(max(1, int(timeout_s))), ip], timeout=timeout_s + 2)
    rtt = None
    m = re.search(r"time[=<]([\d.]+)\s*ms", out)
    if m:
        try:
            rtt = float(m.group(1))
        except ValueError:
            rtt = None
    return {"ok": code == 0, "rtt_ms": rtt, "error": None if code == 0 else (err or out[-200:] or f"exit {code}")}


def _probe_local(node: dict[str, Any], base_url: str | None = None) -> dict[str, Any]:
    url = base_url or node.get("vllm_url") or "http://127.0.0.1:8000"
    hw = metadata.collect_hardware()
    containers = metadata.list_vllm_containers()
    # also catch ray containers
    code, dout, _ = _run(
        ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}"],
        timeout=8,
    )
    if code == 0:
        seen = {c["name"] for c in containers}
        for line in dout.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name, status, image = parts[0], parts[1], parts[2]
            if name in seen:
                continue
            if re.search(r"ray|vllm", name + image, re.I):
                containers.append({"name": name, "status": status, "image": image})

    import httpx

    probe: dict[str, Any] = {"healthy": False, "models": [], "error": None}
    try:
        with httpx.Client(timeout=2.5) as client:
            m = client.get(f"{url.rstrip('/')}/v1/models")
            if m.status_code == 200:
                body = m.json()
                probe["models"] = body.get("data") or []
                probe["healthy"] = True
    except Exception as e:
        probe["error"] = str(e)

    model_id = None
    models = probe.get("models") or []
    if models:
        model_id = models[0].get("id")

    tp = None
    ray_hint = False
    for c in containers:
        blob = f"{c.get('name','')} {c.get('image','')}"
        if "ray" in blob.lower():
            ray_hint = True
        insp = metadata.docker_inspect_flags(c.get("name") or "")
        cmd = insp.get("cmd") or []
        joined = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        m = re.search(r"--tensor-parallel-size[=\s]+(\d+)", joined)
        if m:
            tp = int(m.group(1))

    net = _detect_local_net()
    qsfp_if = node.get("qsfp_if") or net.get("qsfp_if") or None
    lan_ip = node.get("lan_ip") or net.get("lan_ip")
    qsfp_ip = node.get("qsfp_ip") or net.get("qsfp_ip")
    carrier = None
    speed = None
    if qsfp_if:
        try:
            carrier = int(open(f"/sys/class/net/{qsfp_if}/carrier").read().strip())
        except Exception:
            pass
        try:
            sp = int(open(f"/sys/class/net/{qsfp_if}/speed").read().strip())
            if sp > 0:
                speed = sp
        except Exception:
            pass

    code, ib, _ = _run(["ibdev2netdev"], timeout=5)
    up_ifs: list[str] = []
    if code == 0:
        for line in ib.splitlines():
            if "(Up)" in line:
                parts = line.split()
                if len(parts) >= 5:
                    up_ifs.append(parts[4].strip("()"))

    ts_ip = None
    code, ts_out, _ = _run(["tailscale", "ip", "-4"], timeout=4)
    if code == 0 and ts_out.strip():
        ts_ip = ts_out.strip().splitlines()[0]

    return {
        "id": node["id"],
        "label": node.get("label") or node["id"],
        "role": node.get("role") or "node",
        "local": True,
        "online": True,
        "probe_error": None,
        "hostname": hw.get("hostname") or platform.node(),
        "lan_ip": lan_ip,
        "tailscale_ip": ts_ip or node.get("tailscale_ip") or net.get("tailscale_ip"),
        "qsfp_ip": qsfp_ip,
        "gpu_sku": hw.get("gpu_sku"),
        "ram_gib": hw.get("ram_gib"),
        "available_gib": hw.get("available_gib"),
        "endpoint_healthy": bool(probe.get("healthy")),
        "model_id": model_id,
        "models": [{"id": m.get("id")} for m in (models or [])[:5] if isinstance(m, dict)],
        "containers": containers,
        "tensor_parallel_size": tp,
        "ray_hint": ray_hint,
        "qsfp_if": qsfp_if,
        "qsfp_carrier": carrier,
        "qsfp_speed_mbps": speed,
        "roce_up_ifs": up_ifs,
        "vllm_url": url,
    }


def _probe_remote_ssh(node: dict[str, Any]) -> dict[str, Any]:
    host = node.get("ssh_host") or node.get("id")
    base = {
        "id": node["id"],
        "label": node.get("label") or node["id"],
        "role": node.get("role") or "node",
        "local": False,
        "online": False,
        "probe_error": None,
        "hostname": None,
        "lan_ip": node.get("lan_ip"),
        "tailscale_ip": node.get("tailscale_ip"),
        "qsfp_ip": node.get("qsfp_ip"),
        "gpu_sku": None,
        "ram_gib": None,
        "available_gib": None,
        "endpoint_healthy": False,
        "model_id": None,
        "models": [],
        "containers": [],
        "tensor_parallel_size": None,
        "ray_hint": False,
        "qsfp_if": node.get("qsfp_if"),
        "qsfp_carrier": None,
        "qsfp_speed_mbps": None,
        "roce_up_ifs": [],
        "vllm_url": node.get("vllm_url"),
        "ssh_host": host,
    }

    # Prefer QSFP then Tailscale then SSH hostname for reachability ping
    for ip_key in ("qsfp_ip", "tailscale_ip", "lan_ip"):
        ip = node.get(ip_key)
        if not ip:
            continue
        p = _ping_ok(ip, 1.0)
        base[f"ping_{ip_key}"] = p
        if p.get("ok"):
            break

    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "StrictHostKeyChecking=accept-new",
        host,
        "python3",
        "-",
    ]
    try:
        p = subprocess.run(
            cmd,
            input=_remote_probe_script(node),
            text=True,
            capture_output=True,
            timeout=18,
        )
        code, out, err = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        code, out, err = 124, "", "timeout"
    except Exception as e:
        code, out, err = 1, "", str(e)
    if code != 0:
        base["probe_error"] = (err or out or f"ssh exit {code}")[-300:]
        return base
    try:
        data = json.loads(out.strip().splitlines()[-1])
    except Exception as e:
        base["probe_error"] = f"bad json: {e}; out={out[-200:]}"
        return base

    base.update(
        {
            "online": True,
            "hostname": data.get("hostname"),
            "gpu_sku": data.get("gpu_sku"),
            "ram_gib": data.get("ram_gib"),
            "available_gib": data.get("available_gib"),
            "endpoint_healthy": bool(data.get("endpoint_healthy")),
            "model_id": data.get("model_id"),
            "models": data.get("models") or [],
            "containers": data.get("containers") or [],
            "tensor_parallel_size": data.get("tensor_parallel_size"),
            "ray_hint": bool(data.get("ray_hint")),
            "qsfp_if": data.get("qsfp_if") or node.get("qsfp_if"),
            "qsfp_carrier": data.get("qsfp_carrier"),
            "qsfp_speed_mbps": data.get("qsfp_speed_mbps"),
            "roce_up_ifs": data.get("roce_up_ifs") or [],
            "tailscale_ip": data.get("tailscale_ip") or node.get("tailscale_ip"),
        }
    )
    return base


def _multinode_worker_rank(n: dict[str, Any]) -> int | None:
    """Rank of a running headless multi-node worker container (spark-vllm-nN, N>=1).

    Headless workers intentionally expose no /v1/models endpoint, so they can never
    be detected via endpoint health — they must be identified by their container.
    """
    for c in n.get("containers") or []:
        m = re.match(r"spark-vllm-n(\d+)$", str(c.get("name", "")))
        if m and "up" in str(c.get("status", "")).lower():
            rank = int(m.group(1))
            if rank >= 1:
                return rank
    return None


def _node_state(n: dict[str, Any]) -> str:
    if not n.get("online") and not n.get("local"):
        return "offline"
    if n.get("endpoint_healthy") and n.get("model_id"):
        return "serving"
    # Headless TP worker: container up, no endpoint by design → still serving.
    if _multinode_worker_rank(n) is not None:
        return "serving_worker"
    if n.get("containers"):
        # container present but endpoint not healthy yet
        up = any("up" in str(c.get("status", "")).lower() for c in n.get("containers") or [])
        if up:
            return "loading"
    if n.get("online") or n.get("local"):
        return "idle"
    return "offline"


def _summarize(nodes: list[dict[str, Any]], fabric: dict[str, Any]) -> dict[str, Any]:
    for n in nodes:
        n["state"] = _node_state(n)

    online = sum(1 for n in nodes if n.get("state") != "offline")
    head_serving = [n for n in nodes if n.get("state") == "serving"]
    workers_serving = [n for n in nodes if n.get("state") == "serving_worker"]
    # A headless worker serves the head's model — attribute it for display.
    if head_serving and workers_serving:
        head_model = head_serving[0].get("model_id")
        for w in workers_serving:
            if not w.get("model_id"):
                w["model_id"] = head_model
            w["headless_worker"] = True
    serving = head_serving + workers_serving
    models = [n.get("model_id") for n in serving if n.get("model_id")]
    unique_models = sorted({m for m in models if m})

    multi = {
        "mode": "none",  # none | single | multi_aligned | multi_mismatch | multi_partial
        "model_id": None,
        "nodes_serving": [n["id"] for n in serving],
        "tensor_parallel_hint": None,
        "fabric_ok": bool(fabric.get("ok")),
        "message": "No model loaded on the cluster.",
    }

    tps = [int(n["tensor_parallel_size"]) for n in nodes if n.get("tensor_parallel_size")]
    if tps:
        multi["tensor_parallel_hint"] = max(tps)
    # Headless workers don't publish a TP flag we can read; infer it from the
    # head + running worker ranks so a real 2-node serve reports TP=2.
    if workers_serving and head_serving:
        multi["tensor_parallel_hint"] = max(
            multi.get("tensor_parallel_hint") or 0, len(head_serving) + len(workers_serving)
        )

    if len(serving) == 0:
        loading = [n for n in nodes if n.get("state") == "loading"]
        if loading:
            multi["mode"] = "loading"
            multi["message"] = f"Container activity on {', '.join(n['id'] for n in loading)} — endpoint not ready yet."
        elif online == len(nodes):
            multi["message"] = "Cluster hosts online. No serve endpoint healthy."
        else:
            multi["message"] = "Some nodes offline or unreachable over SSH."
    elif len(serving) == 1:
        multi["mode"] = "single"
        multi["model_id"] = serving[0].get("model_id")
        multi["message"] = f"Single-node serve on {serving[0]['id']}: {serving[0].get('model_id')}"
    elif len(unique_models) == 1:
        multi["mode"] = "multi_aligned"
        multi["model_id"] = unique_models[0]
        tp = multi.get("tensor_parallel_hint")
        tp_bit = f" · TP={tp}" if tp else ""
        fabric_bit = " · fabric OK" if fabric.get("ok") else " · fabric check failed"
        multi["message"] = (
            f"Same model on {len(serving)} nodes: {unique_models[0]}{tp_bit}{fabric_bit}"
        )
    else:
        multi["mode"] = "multi_mismatch"
        multi["model_id"] = None
        multi["message"] = "Nodes serving different models — not a clean multi-node load."
        multi["models_by_node"] = {n["id"]: n.get("model_id") for n in serving}

    # partial: one serving one idle with TP expected (only when no headless worker is up)
    if (
        multi["mode"] == "single"
        and multi.get("tensor_parallel_hint")
        and multi["tensor_parallel_hint"] >= 2
        and not workers_serving
    ):
        multi["mode"] = "multi_partial"
        multi["message"] = (
            f"TP≥2 hinted but only {serving[0]['id']} is serving — worker may be down or still loading."
        )

    cluster_reachable = online == len(nodes)
    fabric_ok = bool(fabric.get("ok"))
    # Single-node: healthy if this host is up. Fabric is required only when links were checked.
    return {
        "nodes_total": len(nodes),
        "nodes_online": online,
        "nodes_serving": len(serving),
        "cluster_reachable": cluster_reachable,
        "fabric_ok": fabric_ok,
        "healthy": cluster_reachable and fabric_ok,
        "multi": multi,
    }


def _fabric_link(a: dict[str, Any], b: dict[str, Any], target: str) -> dict[str, Any]:
    link = {
        "from": a["id"],
        "to": b["id"],
        "via": "qsfp",
        "target_ip": target,
        **_ping_ok(target, 1.0),
    }
    link["from_carrier"] = a.get("qsfp_carrier")
    link["to_carrier"] = b.get("qsfp_carrier")
    link["from_speed_mbps"] = a.get("qsfp_speed_mbps")
    link["to_speed_mbps"] = b.get("qsfp_speed_mbps")
    return link


def _fabric_note(fabric_links: list[dict[str, Any]], probed: list[dict[str, Any]]) -> str:
    if not fabric_links:
        return "No multi-node fabric"
    parts: list[str] = []
    seen: set[str] = set()
    for n in probed:
        iface = n.get("qsfp_if")
        if iface and iface not in seen:
            seen.add(iface)
            parts.append(str(iface))
    for lnk in fabric_links:
        ip = lnk.get("target_ip")
        if ip and ip not in seen:
            seen.add(ip)
            parts.append(str(ip))
    if parts:
        return "QSFP RoCE path (" + " / ".join(parts) + ")"
    return "configured interconnect"


def collect_cluster() -> dict[str, Any]:
    cfg = _load_cluster_config()
    nodes_cfg = [n for n in (cfg.get("nodes") or []) if isinstance(n, dict) and n.get("id")]
    probed: list[dict[str, Any]] = []
    n_cfg = len(nodes_cfg)

    for node in nodes_cfg:
        if _node_is_local(node, n_cfg):
            probed.append(_probe_local(node))
        else:
            probed.append(_probe_remote_ssh(node))

    # Fabric: ping configured remote qsfp IPs only — never invent lab interconnects.
    fabric_links: list[dict[str, Any]] = []
    local_nodes = [n for n in probed if n.get("local")]
    remote_nodes = [n for n in probed if not n.get("local")]
    fabric_ok = True
    if local_nodes and remote_nodes:
        for a in local_nodes:
            for b in remote_nodes:
                target = b.get("qsfp_ip")
                if not target:
                    continue
                link = _fabric_link(a, b, str(target))
                fabric_links.append(link)
                if not link.get("ok"):
                    fabric_ok = False
    elif len(probed) >= 2:
        a, b = probed[0], probed[1]
        target = b.get("qsfp_ip")
        if target:
            link = _fabric_link(a, b, str(target))
            fabric_links.append(link)
            fabric_ok = bool(link.get("ok"))

    fabric = {
        "ok": fabric_ok,
        "links": fabric_links,
        "note": _fabric_note(fabric_links, probed),
    }

    summary = _summarize(probed, fabric)

    return {
        "name": cfg.get("name") or "lab-cluster",
        "updated_from": platform.node(),
        "nodes": probed,
        "fabric": fabric,
        "summary": summary,
    }
