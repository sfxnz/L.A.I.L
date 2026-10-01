"""Cluster health for L.A.I.L Status (source of truth).

Default topology is this machine as probed (hostname, LAN, Tailscale, RoCE)
plus every RoCE peer that has ever answered ping (ARP table, or a /24-or-tighter
QSFP scan when the table is cold after reboot) — a peer that goes down stays
listed as offline. LAIL_CLUSTER_JSON or gitignored data/cluster.json override.

Two cadences:
  - inventory (`collect_cluster`, slow tick): containers, endpoint, RoCE rails,
    addresses, reachability and fabric pings. Nodes are probed in parallel; the
    local node in-process and remotes over one multiplexed ssh call each, both
    running the SAME `node_probe` code.
  - telemetry (`PeerStream`, ~1 s): one long-lived `ssh host python3 -u -` per
    remote node streaming memory / temps / GPU / CPU lines. The local node's
    telemetry comes from the status sampler's fast tick.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import platform
import re
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import node_probe

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

# Container naming / rank parsing is shared with the probe that runs on every node.
_rank_from_container_name = node_probe.rank_from_container_name

# Per-node live readings (fast tick / peer stream). Copied onto node dicts as a set.
TELEMETRY_FIELDS = (
    "sampled_at",
    "temperature_c",
    "gpu_util_pct",
    "power_w",
    "memory_used_mib",
    "memory_total_mib",
    "engine_reserved_gib",
    "ram_gib",
    "available_gib",
    "swap_total_gib",
    "swap_used_gib",
    "mem_psi_full_avg10",
    "cpu_util_pct",
    "soc_temp_c",
    "nvme_temp_c",
    "nic_temp_c",
    "rail_rates",
)


def apply_gpu_telemetry(node: dict[str, Any], tel: dict[str, Any] | None) -> dict[str, Any]:
    """Copy probe GPU fields onto a node. Absent/N/A stay None (never a fake 0)."""
    tel = tel or {}
    sku = tel.get("gpu_sku")
    if sku:
        node["gpu_sku"] = sku
    for key in node_probe.GPU_TELEMETRY_FIELDS:
        node[key] = tel.get(key)
    # GiB view for the Status memory bar. GB10 unified memory reports [N/A] → None.
    for src, dst in (("memory_used_mib", "gpu_mem_used_gib"), ("memory_total_mib", "gpu_mem_total_gib")):
        mib = tel.get(src)
        node[dst] = None if mib is None else round(float(mib) / 1024, 2)
    return node


def apply_telemetry(node: dict[str, Any], tel: dict[str, Any] | None) -> dict[str, Any]:
    """Overwrite a node's live readings (and their `sampled_at`, epoch ms) from one sample."""
    tel = tel or {}
    for key in TELEMETRY_FIELDS:
        node[key] = tel.get(key)
    return apply_gpu_telemetry(node, tel)


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


def _container_serve_family(name: str) -> str | None:
    """Stable serve-family key for pairing head/worker containers (node container lists
    hold serve containers only — `node_probe.parse_docker_ps` filtered them)."""
    n = (name or "").strip()
    if not n:
        return None
    if node_probe._OFFICIAL_VLLM_NAME_RE.match(n):
        return re.sub(r"-n\d+$", "", n)
    m = node_probe._DSPARK_VLLM_NAME_RE.match(n)
    if m:
        return re.sub(r"[-_]\d+$", "", n).lower()
    return n.lower()


def _node_serve_families(n: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for c in n.get("containers") or []:
        if "up" not in str(c.get("status", "")).lower():
            continue
        fam = _container_serve_family(str(c.get("name", "")))
        if fam:
            out.add(fam)
    return out


def _subnet_hosts(ip: str, prefix: int) -> list[str]:
    """Usable hosts on a tight QSFP prefix. Refuse /16-and-wider scans."""
    if prefix < 24 or prefix > 30:
        return []
    try:
        net = ipaddress.ip_network(f"{ip}/{prefix}", strict=False)
    except ValueError:
        return []
    self = str(ipaddress.ip_address(ip))
    return [str(h) for h in net.hosts() if str(h) != self]


_PEER_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_PEER_CACHE_SEC = 90.0
# Every peer ever discovered, per local fabric key — a peer that stops answering stays
# listed (and probes as offline) instead of vanishing and leaving the cluster "healthy".
_KNOWN_PEERS: dict[str, dict[str, dict[str, Any]]] = {}


def _clear_peer_cache() -> None:
    _PEER_CACHE.clear()
    _KNOWN_PEERS.clear()


def _peer_cache_key(local: dict[str, Any]) -> str:
    return f"{local.get('qsfp_if')}|{local.get('qsfp_ip')}"


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


def _detect_local_net() -> dict[str, Any]:
    """LAN / Tailscale / RoCE rails from this host (one probe). No baked lab addresses."""
    net = node_probe.net_info()
    rail = net["rails"][0] if net["rails"] else None
    return {
        "lan_ip": net["lan_ip"],
        "tailscale_ip": net["tailscale_ip"],
        "qsfp_if": rail["if"] if rail else None,
        "qsfp_ip": rail["ip"] if rail else None,
        "rails": net["rails"],
        "roce_up_ifs": net["roce_up_ifs"],
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


def _candidate_qsfp_ips(qsfp_if: str, self_ip: str | None, neigh_txt: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ip in _parse_neigh(neigh_txt):
        if ip == self_ip or ip in seen:
            continue
        seen.add(ip)
        out.append(ip)
    if out:
        return out
    code, addr_txt, _ = _run(["ip", "-4", "-o", "addr", "show", "dev", str(qsfp_if)], timeout=4)
    if code != 0:
        return []
    for iface, ip, prefix in node_probe.parse_ip_cidrs(addr_txt):
        if iface != qsfp_if:
            continue
        for host in _subnet_hosts(ip, prefix):
            if host == self_ip or host in seen:
                continue
            seen.add(host)
            out.append(host)
    return out


def _peer_from_ip(local: dict[str, Any], ip: str) -> dict[str, Any] | None:
    name = _hostname_for_ip(ip)
    if name == local.get("id"):
        return None
    ssh_host = name or ip
    node_id = name or ip.replace(".", "-")
    return {
        "id": node_id,
        "label": node_id,
        "role": "worker",
        "local": False,
        "ssh_host": ssh_host,
        "qsfp_ip": ip,
        "qsfp_if": local.get("qsfp_if"),
        "vllm_url": "http://127.0.0.1:8000",
    }


def _discover_fabric_peers(local: dict[str, Any]) -> list[dict[str, Any]]:
    """RoCE peers become remote nodes once they answer ping, and stay known after.

    `ip neigh` is empty after a reboot until something talks on the link, so
    when the table is cold we ping the QSFP prefix (/24 or tighter) in parallel.
    """
    qsfp_if = local.get("qsfp_if")
    self_ip = local.get("qsfp_ip")
    if not qsfp_if:
        return []
    key = _peer_cache_key(local)
    hit = _PEER_CACHE.get(key)
    if hit and (time.monotonic() - hit[0]) < _PEER_CACHE_SEC:
        return [dict(p) for p in hit[1]]

    code, neigh_txt, _ = _run(["ip", "neigh", "show", "dev", str(qsfp_if)], timeout=4)
    candidates = _candidate_qsfp_ips(str(qsfp_if), str(self_ip) if self_ip else None, neigh_txt if code == 0 else "")
    known = _KNOWN_PEERS.setdefault(key, {})
    candidates = [ip for ip in candidates if ip not in known]
    if candidates:
        with ThreadPoolExecutor(max_workers=min(64, max(4, len(candidates)))) as pool:
            results = list(pool.map(lambda ip: (ip, _ping_ok(ip, 1.0)), candidates))
        for ip, ping in results:
            if not ping.get("ok"):
                continue
            peer = _peer_from_ip(local, ip)
            if peer:
                known[ip] = peer
    peers = [known[ip] for ip in sorted(known, key=ipaddress.ip_address)]
    _PEER_CACHE[key] = (time.monotonic(), [dict(p) for p in peers])
    return [dict(p) for p in peers]


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
    # The local probe reuses this detection instead of running ip/ibdev2netdev/tailscale again.
    local["net"] = {k: net.get(k) for k in ("lan_ip", "tailscale_ip", "rails", "roce_up_ifs")}
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


def _ping_ok(ip: str, timeout_s: float = 1.0) -> dict[str, Any]:
    return node_probe.ping(ip, timeout_s)


# ─── probes ───────────────────────────────────────────────────────────────────


def _probe_source() -> str:
    return Path(node_probe.__file__).read_text()


def _ssh_control_dir() -> str | None:
    """Private dir for ssh ControlMaster sockets (short path: unix sockets cap at ~104 bytes)."""
    path = f"/tmp/lail-ssh-{os.getuid()}"
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
        st = os.stat(path)
    except OSError:
        return None
    if st.st_uid != os.getuid() or st.st_mode & 0o077:
        return None
    return path


def _ssh_cmd(host: str, req: dict[str, Any]) -> list[str]:
    """`ssh host python3 -u - '<req>'` — the probe source goes on stdin.

    ControlMaster multiplexes every call to a host over one connection (the peer
    stream keeps it open), so an inventory call costs ~20 ms, not a handshake.
    """
    opts = [
        "BatchMode=yes",
        "ConnectTimeout=5",
        "StrictHostKeyChecking=accept-new",
        "ServerAliveInterval=5",
        "ServerAliveCountMax=2",
    ]
    ctl = _ssh_control_dir()
    if ctl:
        opts += ["ControlMaster=auto", f"ControlPath={ctl}/%C", "ControlPersist=60"]
    cmd = ["ssh"]
    for o in opts:
        cmd += ["-o", o]
    return [*cmd, host, "python3", "-u", "-", shlex.quote(json.dumps(req))]


def _probe_local(
    node: dict[str, Any],
    telemetry: dict[str, Any] | None = None,
    containers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """This host's inventory in-process; telemetry and `docker ps` from the sampler's
    fast tick when given (the slow tick then only adds one batched `docker inspect`)."""
    configured_url = node.get("vllm_url") or "http://127.0.0.1:8000"
    inv = node_probe.inventory(node.get("qsfp_if"), configured_url, net=node.get("net"), containers=containers)
    out = {
        "id": node["id"],
        "label": node.get("label") or node["id"],
        "role": node.get("role") or "node",
        "local": True,
        "online": True,
        "probe_error": None,
        "hostname": inv["hostname"],
        "cpu": inv["cpu"],
        "lan_ip": node.get("lan_ip") or inv["lan_ip"],
        "tailscale_ip": inv["tailscale_ip"] or node.get("tailscale_ip"),
        "qsfp_ip": node.get("qsfp_ip") or inv["qsfp_ip"],
        "endpoint_healthy": inv["endpoint_healthy"],
        "model_id": inv["model_id"],
        "models": inv["models"],
        "containers": inv["containers"],
        "tensor_parallel_size": inv["tensor_parallel_size"],
        "ray_hint": inv["ray_hint"],
        "qsfp_if": inv["qsfp_if"],
        "qsfp_carrier": inv["qsfp_carrier"],
        "qsfp_speed_mbps": inv["qsfp_speed_mbps"],
        "rails": inv["rails"],
        "roce_up_ifs": inv["roce_up_ifs"],
        "vllm_url": inv["vllm_url"],
        "inventory_at": int(time.time() * 1000),
    }
    return apply_telemetry(out, telemetry or node_probe.Telemetry().sample())


def _probe_remote_ssh(node: dict[str, Any], ping_targets: list[str] | None = None) -> dict[str, Any]:
    host = node.get("ssh_host") or node.get("id")
    base: dict[str, Any] = {
        "id": node["id"],
        "label": node.get("label") or node["id"],
        "role": node.get("role") or "node",
        "local": False,
        "online": False,
        "probe_error": None,
        "hostname": None,
        "cpu": None,
        "lan_ip": node.get("lan_ip"),
        "tailscale_ip": node.get("tailscale_ip"),
        "qsfp_ip": node.get("qsfp_ip"),
        "gpu_sku": None,
        "endpoint_healthy": False,
        "model_id": None,
        "models": [],
        "containers": [],
        "tensor_parallel_size": None,
        "ray_hint": False,
        "qsfp_if": node.get("qsfp_if"),
        "qsfp_carrier": None,
        "qsfp_speed_mbps": None,
        "rails": [],
        "roce_up_ifs": [],
        "vllm_url": node.get("vllm_url"),
        "ssh_host": host,
        "ping": None,
        "pings": {},
        "inventory_at": int(time.time() * 1000),
    }
    apply_telemetry(base, None)

    # Reachability: ping every known address at once; first answer in QSFP → Tailscale → LAN
    # order wins. Kept separate from ssh so "host down" and "ssh broken" read differently.
    addrs = [(k, node[k]) for k in ("qsfp_ip", "tailscale_ip", "lan_ip") if node.get(k)]
    if addrs:
        with ThreadPoolExecutor(max_workers=len(addrs)) as pool:
            results = list(pool.map(lambda a: _ping_ok(a[1], 1.0), addrs))
        hit = next(((a, r) for a, r in zip(addrs, results) if r.get("ok")), (addrs[0], results[0]))
        base["ping"] = {"via": hit[0][0], "ip": hit[0][1], **hit[1]}

    req = {
        "mode": "inventory",
        "qsfp_if": node.get("qsfp_if") or "",
        "vllm_url": node.get("vllm_url") or "http://127.0.0.1:8000",
        "ping_targets": list(ping_targets or []),
    }
    try:
        p = subprocess.run(
            _ssh_cmd(host, req), input=_probe_source(), text=True, capture_output=True, timeout=15
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
            "cpu": data.get("cpu"),
            "endpoint_healthy": bool(data.get("endpoint_healthy")),
            "model_id": data.get("model_id"),
            "models": data.get("models") or [],
            "containers": data.get("containers") or [],
            "tensor_parallel_size": data.get("tensor_parallel_size"),
            "ray_hint": bool(data.get("ray_hint")),
            "qsfp_if": data.get("qsfp_if") or node.get("qsfp_if"),
            "qsfp_carrier": data.get("qsfp_carrier"),
            "qsfp_speed_mbps": data.get("qsfp_speed_mbps"),
            "rails": data.get("rails") or [],
            "roce_up_ifs": data.get("roce_up_ifs") or [],
            "tailscale_ip": data.get("tailscale_ip") or node.get("tailscale_ip"),
            "vllm_url": data.get("vllm_url") or node.get("vllm_url"),
            "pings": data.get("pings") or {},
        }
    )
    return apply_telemetry(base, data.get("telemetry"))


class PeerStream:
    """One persistent `ssh host python3 -u -` printing a telemetry JSON line per interval.

    A reader thread keeps the latest line, stamped with THIS server's receive time
    (epoch ms) so remote clock skew never shows up as staleness, and drops it when
    the stream ends. The stream is restarted with exponential backoff (1 → 30 s)
    when ssh or the host drops.
    """

    def __init__(self, host: str, interval_s: float = 1.0) -> None:
        self.host = host
        self.interval_s = interval_s
        self.error: str | None = None
        self._latest: tuple[dict[str, Any], int] | None = None
        self._stop = threading.Event()
        self._proc: subprocess.Popen[str] | None = None
        self._thread = threading.Thread(target=self._run, name=f"peer-stream-{host}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._kill()
        self._thread.join(timeout=3)

    def _kill(self) -> None:
        p = self._proc
        if p is not None and p.poll() is None:
            p.terminate()

    def reading(self) -> dict[str, Any] | None:
        latest = self._latest
        if latest is None:
            return None
        return {**latest[0], "sampled_at": latest[1]}

    def _run(self) -> None:
        backoff = 1.0
        source = _probe_source()
        cmd = _ssh_cmd(self.host, {"mode": "stream", "interval_s": self.interval_s})
        while not self._stop.is_set():
            try:
                p = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
            except OSError as e:
                self.error = str(e)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            self._proc = p
            if self._stop.is_set():
                # stop() ran between Popen and the assignment above and killed nothing.
                p.terminate()
            try:
                assert p.stdin is not None and p.stdout is not None
                p.stdin.write(source)
                p.stdin.close()
                for line in p.stdout:
                    if self._stop.is_set():
                        break
                    try:
                        data = json.loads(line)
                    except ValueError:
                        if line.strip():
                            self.error = line.strip()[-300:]
                        continue
                    self._latest = (data, int(time.time() * 1000))
                    self.error = None
                    backoff = 1.0
            except (OSError, ValueError) as e:
                self.error = str(e)
            if self._stop.is_set():
                self._kill()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            # The stream is gone: its last line is no longer a current reading.
            self._latest = None
            if self._stop.is_set():
                break
            self.error = self.error or f"telemetry stream exited ({p.returncode})"
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 30.0)


# ─── state & summary ──────────────────────────────────────────────────────────


def _multinode_worker_rank(n: dict[str, Any]) -> int | None:
    """Rank of a running headless / TP worker container (rank >= 1), or None.

    Headless workers intentionally expose no /v1/models endpoint, so they can never
    be detected via endpoint health — they must be identified by a known worker
    name (spark-vllm-nN / …-vllm-dspark-N) or by --headless with an explicit rank >= 1.
    Never coerce unknown/0 rank to 1.
    """
    for c in n.get("containers") or []:
        if "up" not in str(c.get("status", "")).lower():
            continue
        name = str(c.get("name", ""))
        name_rank = _rank_from_container_name(name)
        if name_rank is not None and name_rank >= 1:
            return name_rank

        rank: int | None = None
        if c.get("node_rank") is not None:
            try:
                rank = int(c["node_rank"])
            except (TypeError, ValueError):
                rank = None
        headless = bool(c.get("headless"))

        # Explicit headless worker only when rank is known and >= 1.
        if headless and rank is not None and rank >= 1:
            return rank
    return None


def _worker_aligned_with_head(worker: dict[str, Any], head: dict[str, Any], rank: int) -> bool:
    """True when a TP worker plausibly belongs to the same live serve as head."""
    tp = head.get("tensor_parallel_size")
    if tp is not None:
        try:
            if rank >= int(tp):
                return False
        except (TypeError, ValueError):
            pass
    hk = _node_serve_families(head)
    wk = _node_serve_families(worker)
    if not hk or not wk:
        return False
    return bool(hk & wk)


# A serve container Up for longer than this without an endpoint (or a TP role) is not
# "loading" any more — vLLM multi-node readiness gives up after 30 min.
LOADING_MAX_S = 45 * 60
_AGE_UNITS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800, "month": 2592000}


def _container_age_s(status: str) -> float | None:
    """Seconds from a docker status like 'Up 12 minutes' / 'Up About an hour'; None if unknown."""
    m = re.search(r"\bUp\s+(Less than a|About an?|\d+)\s+(second|minute|hour|day|week|month)s?\b", status or "")
    if not m:
        return None
    n = m.group(1)
    count = 0.5 if n.startswith("Less") else 1.0 if n.startswith("About") else float(n)
    return count * _AGE_UNITS[m.group(2)]


def _up_containers(n: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in n.get("containers") or [] if "up" in str(c.get("status", "")).lower()]


def _loading_or_stray(n: dict[str, Any]) -> str:
    """A young (or age-unknown) serve container is still loading; an old one is a stray."""
    for c in _up_containers(n):
        age = _container_age_s(str(c.get("status", "")))
        if age is None or age < LOADING_MAX_S:
            return "loading"
    return "stray"


def _node_state(n: dict[str, Any]) -> str:
    if not n.get("online") and not n.get("local"):
        # Ping answered but ssh failed → the host is up; keys/sshd are the problem.
        return "unreachable" if (n.get("ping") or {}).get("ok") else "offline"
    if n.get("endpoint_healthy") and n.get("model_id"):
        return "serving"
    # Headless TP worker: container up, no endpoint by design → still serving.
    if _multinode_worker_rank(n) is not None:
        return "serving_worker"
    if _up_containers(n):
        return _loading_or_stray(n)
    return "idle"


_DOWN_STATES = ("offline", "unreachable")


def _summarize(nodes: list[dict[str, Any]], fabric: dict[str, Any]) -> dict[str, Any]:
    for n in nodes:
        n["state"] = _node_state(n)

    online = sum(1 for n in nodes if n.get("state") not in _DOWN_STATES)
    head_serving = [n for n in nodes if n.get("state") == "serving"]
    # Drop workers that do not share a serve family with a live head (or exceed TP).
    workers_serving: list[dict[str, Any]] = []
    for w in nodes:
        if w.get("state") != "serving_worker":
            continue
        rank = _multinode_worker_rank(w)
        if rank is None:
            w["state"] = _loading_or_stray(w)
            continue
        if not head_serving:
            workers_serving.append(w)
            continue
        if any(_worker_aligned_with_head(w, h, rank) for h in head_serving):
            workers_serving.append(w)
            continue
        # Leftover / unrelated Up container — do not paint as TP worker.
        w["state"] = _loading_or_stray(w) if _up_containers(w) else "idle"

    # A headless worker serves the head's model — attribute only when aligned.
    if head_serving and workers_serving:
        head_model = head_serving[0].get("model_id")
        for w in workers_serving:
            rank = _multinode_worker_rank(w)
            if rank is None:
                continue
            if not any(_worker_aligned_with_head(w, h, rank) for h in head_serving):
                continue
            if not w.get("model_id"):
                w["model_id"] = head_model
            w["headless_worker"] = True
    serving = head_serving + workers_serving
    # TP role per node (the endpoint rate lives on the serve, not on each rank).
    for n in nodes:
        n["tp_rank"] = None
    if head_serving and workers_serving:
        head_serving[0]["tp_rank"] = 0
        for w in workers_serving:
            w["tp_rank"] = _multinode_worker_rank(w)
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


def _same_subnet(a: dict[str, Any], b: dict[str, Any]) -> bool:
    try:
        return ipaddress.ip_interface(f"{a['ip']}/{a['prefix']}").network == ipaddress.ip_interface(
            f"{b['ip']}/{b['prefix']}"
        ).network
    except (KeyError, ValueError):
        return False


def _link(a: dict[str, Any], b: dict[str, Any], target: str, iface: str | None, ping: dict[str, Any]) -> dict[str, Any]:
    ra = next((r for r in a.get("rails") or [] if r.get("if") == iface), None)
    rb = next((r for r in b.get("rails") or [] if r.get("ip") == target), None)
    return {
        "from": a["id"],
        "to": b["id"],
        "via": "qsfp",
        "iface": iface,
        "target_ip": target,
        "ok": bool(ping.get("ok")),
        "rtt_ms": ping.get("rtt_ms"),
        "error": ping.get("error"),
        "from_carrier": ra["carrier"] if ra else a.get("qsfp_carrier"),
        "to_carrier": rb["carrier"] if rb else b.get("qsfp_carrier"),
        "from_speed_mbps": ra["speed_mbps"] if ra else a.get("qsfp_speed_mbps"),
        "to_speed_mbps": rb["speed_mbps"] if rb else b.get("qsfp_speed_mbps"),
    }


def _rail_targets(a: dict[str, Any], b: dict[str, Any]) -> list[tuple[str | None, str]]:
    """(a's iface, b's IP) for every RoCE rail the two share; b's configured qsfp_ip otherwise."""
    pairs = [
        (ra.get("if"), rb["ip"])
        for ra in a.get("rails") or []
        for rb in b.get("rails") or []
        if _same_subnet(ra, rb)
    ]
    if not pairs and b.get("qsfp_ip"):
        pairs = [(a.get("qsfp_if"), str(b["qsfp_ip"]))]
    return pairs


def _fabric_links(probed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One link per RoCE rail: head → each remote (pinged here), remote ↔ remote (pinged there)."""
    local_nodes = [n for n in probed if n.get("local")]
    remote_nodes = [n for n in probed if not n.get("local")]
    pairs = [(a, b) for a in local_nodes for b in remote_nodes]
    plan = [(a, b, iface, ip) for a, b in pairs for iface, ip in _rail_targets(a, b)]
    # A reachability ping that already hit this exact IP is reused, not repeated.
    reused = {
        (b["id"], b["ping"]["ip"]): b["ping"]
        for _a, b in pairs
        if (b.get("ping") or {}).get("ok")
    }
    todo = sorted({ip for _a, b, _i, ip in plan if (b["id"], ip) not in reused})
    pings: dict[str, dict[str, Any]] = {}
    if todo:
        with ThreadPoolExecutor(max_workers=len(todo)) as pool:
            pings = dict(zip(todo, pool.map(lambda ip: _ping_ok(ip, 1.0), todo)))
    links = [_link(a, b, ip, iface, reused.get((b["id"], ip)) or pings[ip]) for a, b, iface, ip in plan]
    for i, a in enumerate(remote_nodes):
        for b in remote_nodes[i + 1:]:
            for iface, ip in _rail_targets(a, b):
                if ip in (a.get("pings") or {}):
                    links.append(_link(a, b, ip, iface, a["pings"][ip]))
    return links


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


_PAYLOAD_DROP = ("net", "pings")


def collect_cluster(
    local_telemetry: dict[str, Any] | None = None,
    local_containers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Inventory every node in parallel, then fabric + states.

    `local_telemetry` / `local_containers` are the sampler's latest fast-tick sample and
    `docker ps`; without them the local node collects its own. A dead remote only delays
    its own slot (ssh timeouts), never the local node.
    """
    cfg = _load_cluster_config()
    nodes_cfg = [n for n in (cfg.get("nodes") or []) if isinstance(n, dict) and n.get("id")]
    n_cfg = len(nodes_cfg)
    remotes_cfg = [n for n in nodes_cfg if not _node_is_local(n, n_cfg)]

    def probe(node: dict[str, Any]) -> dict[str, Any]:
        if _node_is_local(node, n_cfg):
            return _probe_local(node, local_telemetry, local_containers)
        peers = [str(o["qsfp_ip"]) for o in remotes_cfg if o is not node and o.get("qsfp_ip")]
        return _probe_remote_ssh(node, peers)

    with ThreadPoolExecutor(max_workers=max(1, n_cfg)) as pool:
        probed = list(pool.map(probe, nodes_cfg))

    links = _fabric_links(probed)
    fabric = {
        "ok": all(lnk.get("ok") for lnk in links),
        "links": links,
        "note": _fabric_note(links, probed),
    }
    summary = _summarize(probed, fabric)
    for n in probed:
        for key in _PAYLOAD_DROP:
            n.pop(key, None)

    return {
        "name": cfg.get("name") or "lab-cluster",
        "updated_from": platform.node(),
        "nodes": probed,
        "fabric": fabric,
        "summary": summary,
    }
