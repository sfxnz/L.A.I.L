"""Cluster health for L.A.I.L Status (source of truth).

Default topology is this machine as probed (hostname, LAN, Tailscale, RoCE)
plus any RoCE peers that answer ping (ARP table, or a /24-or-tighter QSFP
scan when the table is cold after reboot). LAIL_CLUSTER_JSON or
gitignored data/cluster.json still override.

Probes:
  - local node via metadata + live NICs
  - remote nodes via passwordless SSH + a small JSON collector
  - QSFP fabric reachability between discovered interconnect IPs
  - whether a model is loaded on each node and whether multi-node looks aligned
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import platform
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
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
_DEFAULT_VLLM_URL = "http://127.0.0.1:8000"


def _node_vllm_url(node: dict[str, Any] | None = None) -> str:
    """Live serve URL for a node. LAB_BASE_URL, then :8000, when unset."""
    if node:
        url = str(node.get("vllm_url") or "").strip()
        if url:
            return url
    env = (os.environ.get("LAB_BASE_URL") or "").strip()
    return env or _DEFAULT_VLLM_URL


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
    with urllib.request.urlopen(vllm_url.rstrip("/") + "/v1/models", timeout=2.5) as r:
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
