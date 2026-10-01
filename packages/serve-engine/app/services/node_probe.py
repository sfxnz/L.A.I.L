"""One Spark's probe: the SAME code runs here (imported) and on every remote node.

Remote nodes get this file's source piped over ssh — `ssh host python3 -u - '<json>'`
— so it is stdlib-only and has no package imports. Two halves:

  telemetry  — cheap per-second readings: /proc/meminfo, /proc/stat, PSI, hwmon
               temperatures, nvidia-smi GPU + compute-apps. `stream` mode prints
               one JSON line per interval for as long as the ssh session lives.
  inventory  — slow topology facts: serve containers (docker ps + one docker
               inspect), the OpenAI endpoint on candidate ports, RoCE rails,
               LAN / Tailscale addresses, optional peer pings.

Every reading that is absent stays None — nothing here fabricates a 0.
"""
from __future__ import annotations

import functools
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.request
from typing import Any

# ─── shell ─────────────────────────────────────────────────────────────────────


def run(cmd: list[str], timeout: float = 8) -> str | None:
    """stdout of `cmd`, or None when it failed / is missing / timed out."""
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    except Exception:
        return None
    return p.stdout if p.returncode == 0 else None


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def _gib(kib: float) -> float:
    return round(kib / 1024 / 1024, 2)


# ─── telemetry ────────────────────────────────────────────────────────────────

GPU_SMI_QUERY = "name,temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total"
GPU_TELEMETRY_FIELDS = (
    "temperature_c",
    "gpu_util_pct",
    "power_w",
    "memory_used_mib",
    "memory_total_mib",
)
_GPU_NA = {"[n/a]", "n/a", "na", ""}


def _smi_num(raw: str) -> float | None:
    s = (raw or "").strip().lower()
    if s in _GPU_NA:
        return None
    for suffix in (" mib", "°c", "w", "%", "c"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
            break
    try:
        return float(s)
    except ValueError:
        return None


def parse_gpu_telemetry(csv_text: str | None) -> dict[str, Any]:
    """Parse one nvidia-smi csv line. Missing and N/A fields stay None, never 0."""
    out: dict[str, Any] = {"gpu_sku": None, **{k: None for k in GPU_TELEMETRY_FIELDS}}
    lines = (csv_text or "").strip().splitlines()
    if not lines:
        return out
    parts = [p.strip() for p in lines[0].split(",")]
    out["gpu_sku"] = parts[0] or None
    if len(parts) >= 6:
        for key, raw in zip(GPU_TELEMETRY_FIELDS, parts[1:6]):
            out[key] = _smi_num(raw)
    elif len(parts) == 2:
        out["memory_total_mib"] = _smi_num(parts[1])
    return out


def parse_compute_apps(csv_text: str | None) -> float | None:
    """GiB held by GPU compute processes (the serving engine's reservation).

    On GB10 unified memory `memory.used` is [N/A], but per-process used_memory is
    reported — it is the slice of system RAM the engine pre-allocated.
    """
    if csv_text is None:
        return None
    mib = 0.0
    for line in csv_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            v = _smi_num(parts[1])
            if v is not None:
                mib += v
    return round(mib / 1024, 2)


def parse_meminfo(text: str | None) -> dict[str, Any]:
    """MemTotal / MemAvailable / swap from /proc/meminfo (kB precision, no `free -g` floor).

    `ram_gib` keeps one decimal: it feeds placement, which was tuned on that value.
    """
    kb: dict[str, float] = {}
    for line in (text or "").splitlines():
        m = re.match(r"^(\w+):\s+(\d+)", line)
        if m:
            kb[m.group(1)] = float(m.group(2))
    total, avail = kb.get("MemTotal"), kb.get("MemAvailable")
    swap_total, swap_free = kb.get("SwapTotal"), kb.get("SwapFree")
    return {
        "ram_gib": None if total is None else round(total / 1024 / 1024, 1),
        "available_gib": None if avail is None else _gib(avail),
        "swap_total_gib": None if swap_total is None else _gib(swap_total),
        "swap_used_gib": None if swap_total is None or swap_free is None else _gib(swap_total - swap_free),
    }


def parse_psi_full_avg10(text: str | None) -> float | None:
    """`full avg10` of /proc/pressure/memory: % of the last 10 s every task stalled on memory."""
    m = re.search(r"^full avg10=([\d.]+)", text or "", re.M)
    return float(m.group(1)) if m else None


def parse_cpu_times(text: str | None) -> tuple[float, float] | None:
    """(busy, total) jiffies from the aggregate `cpu` line of /proc/stat."""
    for line in (text or "").splitlines():
        if line.startswith("cpu "):
            vals = [float(x) for x in line.split()[1:]]
            if len(vals) < 4:
                return None
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)  # idle + iowait
            total = sum(vals[:8])  # guest time is already inside user/nice
            return total - idle, total
    return None


def cpu_util_pct(prev: tuple[float, float] | None, cur: tuple[float, float] | None) -> float | None:
    if not prev or not cur:
        return None
    d_total = cur[1] - prev[1]
    if d_total <= 0:
        return None
    return round(max(0.0, min(100.0, (cur[0] - prev[0]) / d_total * 100)), 1)


def read_temps(root: str = "/sys/class/hwmon") -> dict[str, float | None]:
    """Non-GPU temperatures from hwmon: SoC (acpitz), NVMe composite, ConnectX NIC (mlx5)."""
    soc: list[float] = []
    nvme: list[float] = []
    nvme_composite: list[float] = []
    nic: list[float] = []
    try:
        dirs = sorted(os.listdir(root))
    except OSError:
        dirs = []
    for d in dirs:
        base = os.path.join(root, d)
        name = (_read(os.path.join(base, "name")) or "").strip()
        if name not in ("acpitz", "nvme", "mlx5"):
            continue
        try:
            files = os.listdir(base)
        except OSError:
            continue
        for f in files:
            m = re.match(r"^temp(\d+)_input$", f)
            if not m:
                continue
            raw = (_read(os.path.join(base, f)) or "").strip()
            try:
                c = int(raw) / 1000.0
            except ValueError:
                continue
            if c <= -40 or c > 200:
                continue
            if name == "acpitz":
                soc.append(c)
            elif name == "mlx5":
                nic.append(c)
            else:
                label = (_read(os.path.join(base, f"temp{m.group(1)}_label")) or "").strip()
                (nvme_composite if label == "Composite" else nvme).append(c)
    nvme_pick = nvme_composite or nvme
    return {
        "soc_temp_c": round(max(soc), 1) if soc else None,
        "nvme_temp_c": round(max(nvme_pick), 1) if nvme_pick else None,
        "nic_temp_c": round(max(nic), 1) if nic else None,
    }


class Telemetry:
    """Per-node live readings. Holds the previous /proc/stat sample for CPU %."""

    def __init__(self) -> None:
        self._cpu_prev: tuple[float, float] | None = None

    def sample(self) -> dict[str, Any]:
        gpu = parse_gpu_telemetry(
            run(["nvidia-smi", f"--query-gpu={GPU_SMI_QUERY}", "--format=csv,noheader,nounits"])
        )
        apps = run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        cpu = parse_cpu_times(_read("/proc/stat"))
        util = cpu_util_pct(self._cpu_prev, cpu)
        self._cpu_prev = cpu
        return {
            "sampled_at": int(time.time() * 1000),
            **gpu,
            "engine_reserved_gib": parse_compute_apps(apps) if gpu["gpu_sku"] else None,
            **parse_meminfo(_read("/proc/meminfo")),
            "mem_psi_full_avg10": parse_psi_full_avg10(_read("/proc/pressure/memory")),
            "cpu_util_pct": util,
            **read_temps(),
        }


def parse_lscpu(text: str | None) -> str | None:
    """'10× Cortex-X925 + 10× Cortex-A725' (big.LITTLE) or the one model name."""
    groups: list[list[Any]] = []  # [name, cores_per_socket, sockets]
    for line in (text or "").splitlines():
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if key == "Model name":
            groups.append([val, None, 1])
        elif groups and key == "Core(s) per socket":
            groups[-1][1] = int(val) if val.isdigit() else None
        elif groups and key in ("Socket(s)", "Cluster(s)") and val.isdigit():
            groups[-1][2] = int(val)
    if not groups:
        return None
    if len(groups) == 1:
        return groups[0][0]
    return " + ".join(f"{g[1] * g[2]}× {g[0]}" if g[1] else g[0] for g in groups)


@functools.lru_cache(maxsize=1)
def cpu_model() -> str:
    model = parse_lscpu(run(["lscpu"], timeout=4))
    if model:
        return model
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


# ─── inventory ────────────────────────────────────────────────────────────────

_SERVE_CONTAINER_RE = re.compile(
    r"vllm|spark-vllm|qwen|brain|nemotron|deepseek|llama|dspark|glm",
    re.I,
)
DEFAULT_VLLM_PORTS = (8000, 8888)


def is_serve_container(name: str, image: str = "") -> bool:
    """True for a lab vLLM/llama.cpp-style serve container, including GLM image names."""
    if _SERVE_CONTAINER_RE.search(f"{name} {image}"):
        return True
    img = image.lower()
    return "vllm" in img or "dspark" in img or "ray" in img


def parse_docker_ps(text: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or not parts[0].strip():
            continue
        name, status, image = parts[0], parts[1], parts[2]
        if not is_serve_container(name, image):
            continue
        out.append({"name": name, "status": status, "image": image, "id": parts[3] if len(parts) > 3 else ""})
    return out


def list_serve_containers() -> list[dict[str, Any]]:
    return parse_docker_ps(
        run(["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}\t{{.ID}}"])
    )


def ports_from_bindings(obj: Any) -> list[int]:
    out: list[int] = []
    if not isinstance(obj, dict):
        return out
    for bindings in obj.values():
        for b in bindings or []:
            if not isinstance(b, dict):
                continue
            try:
                port = int(b.get("HostPort"))
            except (TypeError, ValueError):
                continue
            if 1 <= port <= 65535:
                out.append(port)
    return out


def uniq_ports(*groups: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for group in groups:
        for p in group:
            try:
                port = int(p)
            except (TypeError, ValueError):
                continue
            if port in seen or not (1 <= port <= 65535):
                continue
            seen.add(port)
            out.append(port)
    return out


def cmd_blob(cfg: dict[str, Any], args: Any) -> str:
    """Cmd + Args + rank/vLLM env of one container — what rank / TP / port parsing reads."""
    parts: list[str] = []
    for val in (cfg.get("Cmd"), args):
        if isinstance(val, list):
            parts.extend(str(x) for x in val)
        elif val:
            parts.append(str(val))
    for e in cfg.get("Env") or []:
        if isinstance(e, str) and ("RANK=" in e or e.startswith("VLLM_") or e.startswith("--")):
            parts.append(e)
    return " ".join(parts)


def node_rank_from(blob: str, env: list[Any] | None) -> int | None:
    m = re.search(r"--node-rank[=\s]+(\d+)", blob or "")
    if m:
        return int(m.group(1))
    for e in env or []:
        if isinstance(e, str) and e.startswith("NODE_RANK="):
            try:
                return int(e.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


_OFFICIAL_VLLM_NAME_RE = re.compile(r"^spark-vllm-n(\d+)$")
_DSPARK_VLLM_NAME_RE = re.compile(r"(?i)^(?:.+[-_])?vllm[-_]dspark[-_](\d+)$")


def rank_from_container_name(name: str) -> int | None:
    """Rank in a known multi-node name: spark-vllm-nN, or …-vllm-dspark-N (vllm token required)."""
    n = (name or "").strip()
    m = _OFFICIAL_VLLM_NAME_RE.match(n) or _DSPARK_VLLM_NAME_RE.match(n)
    return int(m.group(1)) if m else None


def enrich_containers(containers: list[dict[str, Any]], inspect_json: str | None) -> None:
    """Attach ports / node_rank / headless / TP / started_at from ONE `docker inspect` of all."""
    try:
        data = json.loads(inspect_json or "[]")
    except ValueError:
        data = []
    by_name = {str(d.get("Name") or "").lstrip("/"): d for d in data if isinstance(d, dict)}
    for c in containers:
        d = by_name.get(c["name"])
        if d is None:
            rank = rank_from_container_name(c["name"])
            if rank is not None:
                c["node_rank"] = rank
            continue
        cfg = d.get("Config") or {}
        blob = cmd_blob(cfg, d.get("Args"))
        c["ports"] = uniq_ports(
            [int(m.group(1)) for m in re.finditer(r"--port[=\s]+(\d+)", blob)],
            ports_from_bindings((d.get("NetworkSettings") or {}).get("Ports")),
            ports_from_bindings((d.get("HostConfig") or {}).get("PortBindings")),
        )
        rank = node_rank_from(blob, cfg.get("Env"))
        c["node_rank"] = rank if rank is not None else rank_from_container_name(c["name"])
        c["headless"] = bool(re.search(r"--headless\b", blob))
        m = re.search(r"--tensor-parallel-size[=\s]+(\d+)", blob)
        if m:
            c["tensor_parallel_size"] = int(m.group(1))
        c["started_at"] = (d.get("State") or {}).get("StartedAt")


def candidate_ports(configured_url: str | None, containers: list[dict[str, Any]]) -> list[int]:
    """Ports to try for /v1/models: running containers' published/--port first, then the
    configured URL, then defaults — a discovered :8888 beats a stale configured :8000."""
    configured: list[int] = []
    m = re.search(r":(\d+)(?:/|$)", str(configured_url or ""))
    if m:
        configured.append(int(m.group(1)))
    discovered = [p for c in containers for p in (c.get("ports") or [])]
    return uniq_ports(discovered, configured, list(DEFAULT_VLLM_PORTS))


def probe_models(ports: list[int], fallback_url: str | None = None) -> dict[str, Any]:
    """First port (in order) whose /v1/models lists a model."""
    last_error = None
    for port in ports:
        url = f"http://127.0.0.1:{int(port)}"
        try:
            with urllib.request.urlopen(url + "/v1/models", timeout=2.5) as r:
                models = json.loads(r.read().decode()).get("data") or []
        except Exception as e:
            last_error = str(e)
            continue
        if models:
            first = models[0] if isinstance(models[0], dict) else {}
            return {"healthy": True, "models": models, "model_id": first.get("id"), "vllm_url": url, "error": None}
    return {
        "healthy": False,
        "models": [],
        "model_id": None,
        "vllm_url": (fallback_url or f"http://127.0.0.1:{ports[0] if ports else 8000}").rstrip("/"),
        "error": last_error,
    }


def parse_ip_cidrs(text: str | None) -> list[tuple[str, str, int]]:
    out: list[tuple[str, str, int]] = []
    for line in (text or "").splitlines():
        m = re.match(r"^\d+:\s+(\S+)\s+inet\s+([\d.]+)/(\d+)", line)
        if m:
            out.append((m.group(1), m.group(2), int(m.group(3))))
    return out


def parse_roce_up(text: str | None) -> list[str]:
    return [m.group(1) for m in re.finditer(r"==>\s+(\S+)\s+\(Up\)", text or "")]


_SKIP_IFACES = {"lo", "docker0", "tailscale0"}
_SKIP_IFACE_PREFIXES = ("br-", "veth", "virbr", "cni", "flannel", "wg")


def _iface_skipped(name: str) -> bool:
    return name in _SKIP_IFACES or name.startswith(_SKIP_IFACE_PREFIXES)


def _sys_int(iface: str, key: str) -> int | None:
    raw = (_read(f"/sys/class/net/{iface}/{key}") or "").strip()
    try:
        return int(raw)
    except ValueError:
        return None


def net_info() -> dict[str, Any]:
    """LAN / Tailscale / every Up RoCE rail with its IPv4, carrier and speed. No baked addresses."""
    cidrs = parse_ip_cidrs(run(["ip", "-4", "-o", "addr"], timeout=4))
    roce_up = parse_roce_up(run(["ibdev2netdev"], timeout=4))
    by_iface: dict[str, tuple[str, int]] = {}
    for iface, ip, prefix in cidrs:
        by_iface.setdefault(iface, (ip, prefix))
    rails = []
    for iface in roce_up:
        if iface not in by_iface:
            continue
        speed = _sys_int(iface, "speed")
        rails.append({
            "if": iface,
            "ip": by_iface[iface][0],
            "prefix": by_iface[iface][1],
            "carrier": _sys_int(iface, "carrier"),
            "speed_mbps": speed if speed and speed > 0 else None,
        })
    rail_ifs = {r["if"] for r in rails}
    lan_ip = next((ip for iface, ip, _ in cidrs if not _iface_skipped(iface) and iface not in rail_ifs), None)
    ts = by_iface.get("tailscale0", (None, 0))[0]
    if not ts:
        out = (run(["tailscale", "ip", "-4"], timeout=4) or "").strip()
        ts = out.splitlines()[0] if out else None
    return {"lan_ip": lan_ip, "tailscale_ip": ts, "rails": rails, "roce_up_ifs": roce_up}


def ping(ip: str, timeout_s: float = 1.0) -> dict[str, Any]:
    if not ip:
        return {"ok": False, "rtt_ms": None, "error": "no_ip"}
    try:
        p = subprocess.run(
            ["ping", "-c", "1", "-W", str(max(1, int(timeout_s))), ip],
            text=True, capture_output=True, timeout=timeout_s + 2,
        )
        code, out, err = p.returncode, p.stdout or "", p.stderr or ""
    except Exception as e:
        code, out, err = 1, "", str(e)
    m = re.search(r"time[=<]([\d.]+)\s*ms", out)
    return {
        "ok": code == 0,
        "rtt_ms": float(m.group(1)) if m else None,
        "error": None if code == 0 else (err or out[-200:] or f"exit {code}"),
    }


def ping_all(ips: list[str]) -> dict[str, dict[str, Any]]:
    """Ping each IP once, concurrently."""
    out: dict[str, dict[str, Any]] = {}

    def one(ip: str) -> None:
        out[ip] = ping(ip)

    threads = [threading.Thread(target=one, args=(ip,)) for ip in dict.fromkeys(ips) if ip]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def inventory(
    qsfp_if: str | None = None,
    vllm_url: str | None = None,
    ping_targets: list[str] | None = None,
    net: dict[str, Any] | None = None,
    containers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Slow topology facts for this node (seconds-scale cadence, not per tick).

    `net` is a `net_info()` result the caller already has (skips re-running ip/ibdev2netdev);
    `containers` a `list_serve_containers()` result it already has (skips `docker ps`).
    """
    containers = [dict(c) for c in containers] if containers is not None else list_serve_containers()
    if containers:
        enrich_containers(containers, run(["docker", "inspect", *[c["name"] for c in containers]], timeout=15))
    endpoint = probe_models(candidate_ports(vllm_url, containers), fallback_url=vllm_url)
    tp = next((c["tensor_parallel_size"] for c in containers if c.get("tensor_parallel_size") is not None), None)
    net = net or net_info()
    rails = net.get("rails") or []
    rail = next((r for r in rails if r["if"] == qsfp_if), None) or (rails[0] if rails and not qsfp_if else None)
    iface = qsfp_if or (rail["if"] if rail else None)
    carrier = rail["carrier"] if rail else (_sys_int(iface, "carrier") if iface else None)
    speed = rail["speed_mbps"] if rail else (_sys_int(iface, "speed") if iface else None)
    return {
        "hostname": platform.node(),
        "cpu": cpu_model(),
        "endpoint_healthy": endpoint["healthy"],
        "model_id": endpoint["model_id"],
        "models": [{"id": m.get("id")} for m in endpoint["models"][:5] if isinstance(m, dict)],
        "vllm_url": endpoint["vllm_url"],
        "containers": containers,
        "tensor_parallel_size": tp,
        "ray_hint": any("ray" in f"{c['name']} {c['image']}".lower() for c in containers),
        "lan_ip": net.get("lan_ip"),
        "tailscale_ip": net.get("tailscale_ip"),
        "qsfp_if": iface,
        "qsfp_ip": rail["ip"] if rail else None,
        "qsfp_carrier": carrier,
        "qsfp_speed_mbps": speed if speed and speed > 0 else None,
        "rails": rails,
        "roce_up_ifs": net.get("roce_up_ifs") or [],
        "pings": ping_all(list(ping_targets or [])),
    }


# ─── remote entry point ───────────────────────────────────────────────────────


def stream(interval_s: float) -> None:
    """One telemetry JSON line per interval, fixed-rate, until stdout or the session closes."""
    tel = Telemetry()
    parent = os.getppid()
    next_t = time.monotonic()
    while os.getppid() == parent:
        print(json.dumps(tel.sample()), flush=True)
        next_t += interval_s
        now = time.monotonic()
        if next_t < now:
            next_t = now
        time.sleep(next_t - now)


def main(req: dict[str, Any]) -> None:
    if req.get("mode") == "stream":
        stream(float(req.get("interval_s") or 1.0))
        return
    out = inventory(req.get("qsfp_if"), req.get("vllm_url"), req.get("ping_targets"))
    out["telemetry"] = Telemetry().sample()
    print(json.dumps(out), flush=True)


if __name__ == "__main__" and len(sys.argv) > 1:
    try:
        main(json.loads(sys.argv[1]))
    except (BrokenPipeError, KeyboardInterrupt):
        pass
