"""Background status sampler — the one writer of the cached `/api/status` snapshot.

`/api/status` used to run nvidia-smi / docker / ssh / ping synchronously on the
event loop for every request while the web polled it ~20×/min; the controller
aborted at 3 s and the UI showed "idle". Two asyncio tasks (started from the app
lifespan) refresh the snapshot instead — endpoint + hardware + containers every
~2 s, cluster (ssh/ping) + tool-eval binary every ~10 s — all blocking work via
`asyncio.to_thread`. Requests only read the cache. Because the sampler is the
only caller of `metadata.probe_endpoint`, the live tok/s counter deltas in
`metadata._LIVE_RATE` are computed over a fixed window by a single writer.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..config import DEFAULT_BASE_URL, MODEL_PRESETS, SERVE_EXAMPLES
from . import agentic, cluster, metadata

log = logging.getLogger(__name__)

FAST_INTERVAL_S = 2.0
SLOW_INTERVAL_S = 10.0
FIRST_SAMPLE_TIMEOUT_S = 2.0


def headroom_for(available_gib: float | None) -> str:
    if available_gib is None:
        return "ok"
    if available_gib < 15:
        return "critical"
    if available_gib < 60:
        return "tight"
    return "ok"


def _cluster_pending() -> dict[str, Any]:
    return {"nodes": [], "summary": {"healthy": False}, "pending": True}


class StatusSampler:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        fast_s: float = FAST_INTERVAL_S,
        slow_s: float = SLOW_INTERVAL_S,
    ) -> None:
        self.base_url = base_url
        self.fast_s = fast_s
        self.slow_s = slow_s
        self._probe: dict[str, Any] | None = None
        self._hardware: dict[str, Any] = {}
        self._containers: list[dict[str, Any]] = []
        self._cluster: dict[str, Any] = _cluster_pending()
        self._tool_eval: dict[str, Any] | None = None
        self._snapshot: dict[str, Any] | None = None
        self._sampled_mono: float | None = None
        self._sampled_at: str | None = None
        self._tasks: list[asyncio.Task[None]] = []
        # Events are created in start() so they bind to the running loop.
        self._ready: asyncio.Event | None = None
        self._cluster_ready: asyncio.Event | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._ready = asyncio.Event()
        self._cluster_ready = asyncio.Event()
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

    async def _loop(self, tick, interval_s: float) -> None:
        while True:
            try:
                await tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("status sampler: %s failed", tick.__name__)
            await asyncio.sleep(interval_s)

    # ── sampling ─────────────────────────────────────────────────────────────

    async def sample(self) -> dict[str, Any]:
        """Fast tick: endpoint probe (4 GETs concurrently), nvidia-smi, docker ps."""
        probe, hardware, containers = await asyncio.gather(
            metadata.probe_endpoint(self.base_url),
            asyncio.to_thread(metadata.collect_hardware),
            asyncio.to_thread(metadata.list_vllm_containers),
        )
        self._probe = probe
        self._hardware = hardware
        self._containers = containers
        self._sampled_mono = time.monotonic()
        self._sampled_at = metadata.utc_now()
        self._publish()
        if self._ready is not None:
            self._ready.set()
        return self._snapshot or {}

    async def sample_cluster(self) -> None:
        """Slow tick: cluster ssh/ping/fabric probes and the tool-eval binary check."""
        try:
            info = await asyncio.to_thread(cluster.collect_cluster)
        except Exception as e:
            info = {"error": str(e), "nodes": [], "summary": {"healthy": False}}
        self._cluster = info
        self._tool_eval = await asyncio.to_thread(agentic.tool_eval_available)
        if self._probe is not None:
            self._publish()
        if self._cluster_ready is not None:
            self._cluster_ready.set()

    def _publish(self) -> None:
        probe = self._probe or {}
        cluster.attach_live_rates(self._cluster, probe.get("metrics") or {})
        model_id = None
        if probe.get("models"):
            model_id = probe["models"][0].get("id")
        self._snapshot = {
            "healthy": probe.get("healthy"),
            "base_url": self.base_url,
            "model_id": model_id,
            "models": probe.get("models"),
            "version": probe.get("version"),
            "metrics": probe.get("metrics"),
            "hardware": self._hardware,
            "containers": self._containers,
            "headroom": headroom_for(self._hardware.get("available_gib")),
            "error": probe.get("error"),
            "presets": list(MODEL_PRESETS.keys()),
            "serve_examples": SERVE_EXAMPLES,
            "tool_eval": self._tool_eval,
            "cluster": self._cluster,
            "sampled_at": self._sampled_at,
        }

    # ── readers ──────────────────────────────────────────────────────────────

    def probe(self) -> dict[str, Any] | None:
        """Latest endpoint probe, for envelopes built off the loop (bench threads, import)."""
        return self._probe

    async def status(self, timeout: float = FIRST_SAMPLE_TIMEOUT_S) -> dict[str, Any]:
        """Cached snapshot plus `stale_s`. Waits (bounded) only for the very first sample."""
        if self._snapshot is None and self._ready is not None and self._cluster_ready is not None:
            try:
                await asyncio.wait_for(
                    asyncio.gather(self._ready.wait(), self._cluster_ready.wait()), timeout
                )
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
                "hardware": {},
                "containers": [],
                "headroom": "ok",
                "error": "status sampler warming up",
                "presets": list(MODEL_PRESETS.keys()),
                "serve_examples": SERVE_EXAMPLES,
                "tool_eval": self._tool_eval,
                "cluster": self._cluster,
                "sampled_at": None,
                "stale_s": None,
            }
        return {**self._snapshot, "stale_s": round(time.monotonic() - self._sampled_mono, 1)}


SAMPLER = StatusSampler()
