"""Background status sampler — the one writer of the cached `/api/status` snapshot.

Requests only read the cache; all blocking work runs via `asyncio.to_thread`.
Two fixed-rate asyncio loops (started from the app lifespan; no drift — the next
tick is scheduled from the previous deadline, not from when work finished):

  fast (1 s)  endpoint probe (/metrics → live rates, /health, /v1/models; the engine
              — vLLM, SGLang, llama.cpp, TensorFold — detected from what answers),
              this host's telemetry, docker ps (+ docker inspect of the serving
              container when it changes, or once per slow interval)
  slow (10 s) cluster inventory (all nodes in parallel); keeps one persistent
              telemetry stream per remote node

Every publish overlays the freshest telemetry onto the cluster nodes — the local
node from the fast tick, remote nodes from their streams — each with its own
`sampled_at` (server epoch ms). Because the sampler is the only caller of
`metadata.probe_endpoint`, the live-rate counter deltas have a single writer.

The snapshot carries only what changes: static data (serve presets/examples, the
tool-eval binary) has its own routes. `status(after_ms=…, wait_s=…)` is a long poll
that returns as soon as a newer snapshot is published, so the controller's live
stream forwards each 1 s sample the moment it lands instead of polling on a beat.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

import httpx

from ..config import DEFAULT_BASE_URL, MEM_FLOOR_GIB
from . import cluster, metadata, node_probe

log = logging.getLogger(__name__)

FAST_INTERVAL_S = 1.0
SLOW_INTERVAL_S = 10.0
FIRST_SAMPLE_TIMEOUT_S = 2.0
# Upper bound of one long poll (`status(after_ms=…, wait_s=…)`).
MAX_WAIT_S = 10.0
# /metrics is a local GET; a slow vLLM must not stall host telemetry for long.
ENDPOINT_TIMEOUT_S = 1.5

_PRESSURE_ORDER = {"ok": 0, "tight": 1, "critical": 2}
# A peer-stream line older than this many stream intervals is not shown as current.
STREAM_STALE_INTERVALS = 3


def headroom_for(tel: dict[str, Any] | None, floor_gib: float = MEM_FLOOR_GIB) -> str:
    """Memory pressure of one node, judged against the failure mode — not a fixed GiB line.

    On GB10 unified memory a healthy serve pre-allocates most of RAM, so "little
    available" is normal. What kills a serve is running out of BOTH RAM and swap
    (the lab's OOM guard docker-kills the model at MemAvailable < floor AND
    SwapFree < floor), or the kernel stalling on reclaim (PSI full).

      critical  available < 2×floor and swap free < 2×floor, or PSI full avg10 ≥ 10 %
      tight     available < 4×floor, or PSI full avg10 ≥ 1 %
      ok        otherwise (and when nothing was read)
    """
    tel = tel or {}
    avail = tel.get("available_gib")
    psi = tel.get("mem_psi_full_avg10") or 0.0
    swap_total = tel.get("swap_total_gib")
    swap_free = None if swap_total is None else swap_total - (tel.get("swap_used_gib") or 0.0)
    if avail is not None and avail < 2 * floor_gib and (swap_free is None or swap_free < 2 * floor_gib):
        return "critical"
    if psi >= 10:
        return "critical"
    if (avail is not None and avail < 4 * floor_gib) or psi >= 1:
        return "tight"
    return "ok"


def _cluster_pending() -> dict[str, Any]:
    return {"nodes": [], "summary": {"healthy": False}, "pending": True}


def _publish_cluster(
    info: dict[str, Any],
    local_tel: dict[str, Any] | None,
    streams: dict[str, cluster.PeerStream],
) -> dict[str, Any]:
    """Copy of the slow-tick cluster with the freshest telemetry overlaid per node.

    A remote node takes its stream's line only while the node is online and the
    line is fresh (≤ STREAM_STALE_INTERVALS stream intervals old): a dead peer or a
    wedged stream never keeps its last numbers on screen as if they were current.
    A fresh line always wins over the slow tick's one-shot reading, even when that
    one's `sampled_at` looks newer: the two are stamped by different clocks (the
    peer's own vs this host's receive time), and only the stream carries the 1 s
    deltas (CPU %, rail rates) — dropping them every slow tick would blank them.
    Never mutates `info` — earlier snapshots keep the numbers they were served with.
    """
    now_ms = int(time.time() * 1000)
    nodes = []
    for raw in info.get("nodes") or []:
        node = dict(raw)
        if node.get("local"):
            if local_tel and (local_tel.get("sampled_at") or 0) >= (node.get("sampled_at") or 0):
                cluster.apply_telemetry(node, local_tel)
        else:
            stream = streams.get(node.get("id"))
            reading = stream.reading() if stream else None
            at = (reading or {}).get("sampled_at") or 0
            if (
                reading
                and node.get("online")
                and now_ms - at <= STREAM_STALE_INTERVALS * stream.interval_s * 1000
            ):
                cluster.apply_telemetry(node, reading)
            node["telemetry_error"] = stream.error if stream else None
        # A down node has no current memory reading, so it never drives serve.headroom.
        live = node.get("local") or node.get("online")
        node["mem_pressure"] = headroom_for(node) if live and node.get("available_gib") is not None else None
        nodes.append(node)
    return {**info, "nodes": nodes}


class StatusSampler:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        fast_s: float = FAST_INTERVAL_S,
        slow_s: float = SLOW_INTERVAL_S,
    ) -> None:
        self.base_url = base_url
        # Set by the slow tick, applied by the fast tick: the live-rate state has one writer.
        self._pending_url: str | None = None
        self.fast_s = fast_s
        self.slow_s = slow_s
        self._probe: dict[str, Any] | None = None
        self._telemetry = node_probe.Telemetry()
        self._hardware: dict[str, Any] = {}
        self._containers: list[dict[str, Any]] = []
        # `docker inspect` of the serving container: once per slow interval (or when the
        # serving container changes), read by every fast tick for `engine.flags` / uptime.
        self._inspect: dict[str, Any] = {}
        self._inspect_name: str | None = None
        self._inspect_mono: float | None = None
        # /version only changes with the engine: fetched on the first tick and after a change.
        self._need_version = True
        # Engine detected on the endpoint (owned_by / metrics prefix) — picks the version
        # route and the metrics adapter for the next probe.
        self._engine: str | None = None
        self._cluster: dict[str, Any] = _cluster_pending()
        self._cluster_mono: float | None = None
        self._published_cluster: dict[str, Any] = self._cluster
        self._streams: dict[str, cluster.PeerStream] = {}
        self._snapshot: dict[str, Any] | None = None
        self._sampled_mono: float | None = None
        self._sampled_at: str | None = None
        self._client: httpx.AsyncClient | None = None
        self._tasks: list[asyncio.Task[None]] = []
        # Events are created in start() so they bind to the running loop.
        self._ready: asyncio.Event | None = None
        self._cluster_ready: asyncio.Event | None = None
        # Set (and dropped) on every publish: long polls wait on it for the next snapshot.
        self._next: asyncio.Event | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._ready = asyncio.Event()
        self._cluster_ready = asyncio.Event()
        self._client = httpx.AsyncClient(timeout=ENDPOINT_TIMEOUT_S)
        self._tasks = [
            loop.create_task(self._loop(self.sample_cluster, self.slow_s), name="status-sampler-slow"),
            loop.create_task(self._loop(self.sample, self.fast_s), name="status-sampler-fast"),
        ]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        self._ready = None
        self._cluster_ready = None
        streams, self._streams = self._streams, {}
        await asyncio.gather(*(asyncio.to_thread(s.stop) for s in streams.values()))
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _loop(self, tick, interval_s: float) -> None:
        """Fixed-rate: deadlines advance by `interval_s`; an overrun skips the missed slots."""
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        while True:
            try:
                await tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("status sampler: %s failed", tick.__name__)
            deadline += interval_s
            now = loop.time()
            if deadline < now:
                deadline += math.ceil((now - deadline) / interval_s) * interval_s
            await asyncio.sleep(deadline - now)

    # ── sampling ─────────────────────────────────────────────────────────────

    async def sample(self) -> dict[str, Any]:
        """Fast tick: endpoint probe (GETs concurrently), this host's telemetry, docker ps."""
        moved = self._pending_url is not None
        if moved:
            log.info("status sampler: endpoint moved %s → %s", self.base_url, self._pending_url)
            self.base_url, self._pending_url = self._pending_url, None
            self._need_version = True
            self._engine = None
            metadata.reset_live_rate_state()
        hint = self._engine or next(
            (c.get("engine") for c in self._containers if "Up" in str(c.get("status", "")) and c.get("engine")),
            None,
        )
        probe, hardware, containers = await asyncio.gather(
            metadata.probe_endpoint(
                self.base_url, ENDPOINT_TIMEOUT_S, client=self._client, version=self._need_version, engine=hint
            ),
            asyncio.to_thread(metadata.collect_hardware, self._telemetry),
            asyncio.to_thread(metadata.list_vllm_containers),
        )
        if self._engine is not None and probe.get("engine") not in (None, self._engine):
            # Another engine answers on the same URL: its counters and version are not the
            # old one's — this window's deltas mixed the two, so it shows no rate.
            metadata.reset_live_rate_state()
            self._need_version, moved = True, True
            if probe.get("metrics"):
                probe["metrics"] = {
                    **probe["metrics"], **dict.fromkeys(metadata.LIVE_RATE_KEYS), "last_burst": None, "last_prefill": None
                }
        self._engine = probe.get("engine") or self._engine
        prev = {} if moved else (self._probe or {})  # the old endpoint's scrape is not this one's
        if not self._need_version:
            probe["version"] = prev.get("version")
        elif probe.get("version") is not None:
            self._need_version = False
        if not probe.get("metrics") and probe.get("healthy") and prev.get("metrics"):
            # /metrics timed out under load: keep the last scrape's gauges and counters
            # (its sampled_at shows their age), but no window was measured — no live rate.
            probe["metrics"] = {**prev["metrics"], **dict.fromkeys(metadata.LIVE_RATE_KEYS)}
        self._probe = probe
        self._hardware = hardware
        self._containers = containers
        await self._refresh_inspect()
        self._sampled_mono = time.monotonic()
        self._sampled_at = metadata.utc_now()
        self._publish()
        if self._ready is not None:
            self._ready.set()
        return self._snapshot or {}

    async def sample_cluster(self) -> None:
        """Slow tick: cluster inventory (parallel), peer streams, endpoint URL."""
        try:
            # Reuse the fast tick's telemetry and `docker ps` (once it has run) for this host.
            info = await asyncio.to_thread(
                cluster.collect_cluster,
                self._hardware or None,
                self._containers if self._probe is not None else None,
            )
        except Exception as e:
            info = {"error": str(e), "nodes": [], "summary": {"healthy": False}}
        self._cluster = info
        self._cluster_mono = time.monotonic()
        await self._sync_streams(info.get("nodes") or [])
        self._follow_endpoint(info.get("nodes") or [])
        if self._probe is not None:
            self._publish()
        else:
            self._published_cluster = _publish_cluster(self._cluster, self._hardware, self._streams)
        if self._cluster_ready is not None:
            self._cluster_ready.set()

    async def _sync_streams(self, nodes: list[dict[str, Any]]) -> None:
        """One telemetry stream per remote node, started/stopped as the topology changes."""
        if not self._tasks:
            return  # one-shot sampling (tests, scripts): no background ssh sessions
        want = {n["id"]: n.get("ssh_host") or n["id"] for n in nodes if not n.get("local") and n.get("id")}
        for node_id in [k for k, s in self._streams.items() if want.get(k) != s.host]:
            await asyncio.to_thread(self._streams.pop(node_id).stop)
        for node_id, host in want.items():
            if node_id not in self._streams:
                stream = cluster.PeerStream(host, self.fast_s)
                stream.start()
                self._streams[node_id] = stream

    def _follow_endpoint(self, nodes: list[dict[str, Any]]) -> None:
        """Probe the port the local serve actually answers on (e.g. :8888), not a fixed :8000.

        Only records the move: the fast tick switches URL and resets the rate state
        between probes, so an in-flight scrape of the old URL can't seed the new one.
        """
        local = next((n for n in nodes if n.get("local")), None)
        url = ((local or {}).get("vllm_url") or "").rstrip("/")
        if local and local.get("endpoint_healthy") and url and url != self.base_url.rstrip("/"):
            self._pending_url = url

    async def _refresh_inspect(self) -> None:
        running = [c for c in self._containers if "Up" in str(c.get("status", ""))]
        name = running[0].get("name") if running else None
        if name != self._inspect_name:
            self._inspect = {}
            self._inspect_mono = None
            if self._inspect_name is not None:
                # Another engine: its counters and version are not the old one's.
                metadata.reset_live_rate_state()
                self._need_version = True
                self._engine = None
        self._inspect_name = name
        if not name:
            return
        mono = time.monotonic()
        if self._inspect_mono is not None and mono - self._inspect_mono < self.slow_s:
            return
        self._inspect = await asyncio.to_thread(metadata.docker_inspect_flags, name)
        self._inspect_mono = mono

    def _publish(self) -> None:
        probe = self._probe or {}
        model_id = None
        if probe.get("models"):
            model_id = probe["models"][0].get("id")
        published = _publish_cluster(self._cluster, self._hardware, self._streams)
        self._published_cluster = published
        pressures = [headroom_for(self._hardware)] + [
            n["mem_pressure"] for n in published.get("nodes") or [] if n.get("mem_pressure")
        ]
        self._snapshot = {
            "healthy": probe.get("healthy"),
            "base_url": self.base_url,
            "model_id": model_id,
            # /v1/models minus the per-call noise (vLLM stamps `created` and a fresh
            # permission id on every answer): what is served, and its context window.
            "models": [
                {"id": m.get("id"), "max_model_len": m.get("max_model_len")}
                for m in probe.get("models") or []
                if isinstance(m, dict)
            ],
            "version": probe.get("version"),
            "metrics": probe.get("metrics"),
            "engine": metadata.build_engine(probe, self._inspect),
            "hardware": self._hardware,
            "containers": self._containers,
            # Worst node: under TP, any one rank running out of memory takes the serve down.
            "headroom": max(pressures, key=_PRESSURE_ORDER.__getitem__),
            "error": probe.get("error"),
            "cluster": published,
            "sampled_at": self._sampled_at,
            "sampled_at_ms": int(time.time() * 1000),
        }
        ev, self._next = self._next, None
        if ev is not None:
            ev.set()

    # ── readers ──────────────────────────────────────────────────────────────

    def probe(self) -> dict[str, Any] | None:
        """Latest endpoint probe plus the sample it was taken with (containers, the serving
        container's inspect, this host's telemetry), for envelopes built off the loop
        (bench threads, import) — they describe the moment the Status page shows."""
        if self._probe is None:
            return None
        return {**self._probe, "containers": self._containers, "inspect": self._inspect, "hardware": self._hardware}

    def hardware(self) -> dict[str, Any]:
        """Latest fast-tick telemetry of this host ({} before the first tick)."""
        return self._hardware

    def cluster(self, max_age_s: float | None = None) -> dict[str, Any] | None:
        """Latest published cluster, or None while pending / older than `max_age_s`."""
        if self._cluster_mono is None or self._cluster.get("pending"):
            return None
        if max_age_s is not None and time.monotonic() - self._cluster_mono > max_age_s:
            return None
        return self._published_cluster

    async def status(
        self,
        timeout: float = FIRST_SAMPLE_TIMEOUT_S,
        *,
        after_ms: int | None = None,
        wait_s: float = 0.0,
    ) -> dict[str, Any]:
        """Cached snapshot plus `stale_s`. Waits (bounded) for the very first sample.

        With `after_ms` (a snapshot's `sampled_at_ms` the caller already has) it also
        waits up to `wait_s` for a newer one to be published, and returns the newest
        snapshot either way — a long poll, never a busy loop.
        """
        if self._snapshot is None and self._ready is not None and self._cluster_ready is not None:
            try:
                await asyncio.wait_for(
                    asyncio.gather(self._ready.wait(), self._cluster_ready.wait()), timeout
                )
            except asyncio.TimeoutError:
                pass
        if (
            after_ms is not None
            and wait_s > 0
            and self._snapshot is not None
            and (self._snapshot.get("sampled_at_ms") or 0) <= after_ms
        ):
            if self._next is None:
                self._next = asyncio.Event()
            try:
                await asyncio.wait_for(self._next.wait(), min(wait_s, MAX_WAIT_S))
            except asyncio.TimeoutError:
                pass
        if self._snapshot is None or self._sampled_mono is None:
            return {
                "healthy": None,
                "base_url": self.base_url,
                "model_id": None,
                "models": [],
                "version": None,
                "metrics": {},
                "engine": metadata.build_engine(None, None),
                "hardware": {},
                "containers": [],
                "headroom": "ok",
                "error": "status sampler warming up",
                "cluster": self._published_cluster,
                "sampled_at": None,
                "sampled_at_ms": None,
                "stale_s": None,
            }
        return {**self._snapshot, "stale_s": round(time.monotonic() - self._sampled_mono, 1)}


SAMPLER = StatusSampler()
