"use client";

import { create } from "zustand";
import { api, type ClusterNode, type LabStatus } from "./api";
import { isUnauthorizedError } from "./auth-token";

/**
 * The ONE lab-status poll. AppShell owns the cadence (2s while the tab is
 * visible, paused while hidden); every page reads from here instead of
 * running its own interval. 401 (`needToken`) and unreachable are kept apart —
 * a fresh browser with no token is not an outage.
 *
 * The store also keeps the two derived things every instrument agrees on:
 *   · `samples` — a 60 s ring buffer (30 × 2 s) of per-node telemetry, the ONE
 *     place status ticks accumulate (node sparklines, hardware strip, mini-board);
 *   · `liveRun` — the run's usage-calibrated aggregate while a Streams/Bench run
 *     is live, so the header shows the same number the run does (never two
 *     unexplained tok/s).
 */

export const LAB_STATUS_POLL_MS = 2000;
/** 30 samples × 2 s = the 60 s window every sparkline draws. */
export const NODE_SAMPLES_KEEP = 30;

export type LiveRun = {
  tok_s: number;
  peak: number;
  running: number;
  waiting: number;
  source: "streams" | "bench";
};

export type NodeSample = {
  /** epoch ms of the poll that produced it */
  t: number;
  tok_s: number | null;
  power: number | null;
  util: number | null;
  temp: number | null;
};

type LabStatusStore = {
  status: LabStatus | null;
  /** true until the first poll settles (success or failure) */
  loading: boolean;
  /** controller answered 401 — paste LAIL_TOKEN */
  needToken: boolean;
  /** controller did not answer at all */
  unreachable: boolean;
  error: string | null;
  /** epoch ms of the last successful poll */
  lastGoodAt: number | null;
  /** per-node ring buffer of the last NODE_SAMPLES_KEEP polls, oldest first */
  samples: Record<string, NodeSample[]>;
  /** the live run's aggregate, or null when no run is streaming */
  liveRun: LiveRun | null;
  refresh: () => Promise<void>;
  /** Writes only when a value changed — a fresh object per render looped hydration once. */
  setLiveRun: (run: LiveRun | null) => void;
};

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/**
 * The node whose endpoint serve.metrics describes: the serve-engine probes the endpoint
 * of the host it runs on, so only the LOCAL serving node owns those rates. A TP worker
 * serves the same tokens, and a remote node serving on its own is not what was probed.
 */
export function ownsEndpoint(n: ClusterNode): boolean {
  return n.state === "serving" && !!n.local;
}

/**
 * The live node with the least MemAvailable — under TP, any one rank running out of
 * memory takes the serve down, so this is the number that matters. Offline nodes carry
 * no current reading and are skipped.
 */
export function tightestNode(nodes: ClusterNode[] | undefined): ClusterNode | null {
  let best: ClusterNode | null = null;
  for (const n of nodes ?? []) {
    if (!(n.local || n.online) || n.available_gib == null) continue;
    if (best == null || n.available_gib < (best.available_gib as number)) best = n;
  }
  return best;
}

/**
 * Append one sample per node (pure). Nodes come and go, so series are keyed by id.
 * `tok_s` is the endpoint's decode rate (serve.metrics), recorded on the node that owns
 * the endpoint only (`ownsEndpoint`) — TP workers serve the same tokens and must not read as 2×.
 */
export function pushNodeSamples(
  prev: Record<string, NodeSample[]>,
  nodes: ClusterNode[] | undefined,
  t: number,
  keep = NODE_SAMPLES_KEEP,
  endpointTokS: number | null | undefined = null,
): Record<string, NodeSample[]> {
  if (!nodes?.length) return prev;
  const next: Record<string, NodeSample[]> = { ...prev };
  for (const n of nodes) {
    const cur = prev[n.id] ?? [];
    const last = cur[cur.length - 1];
    if (last && last.t === t) continue;
    const sample: NodeSample = {
      t,
      tok_s: ownsEndpoint(n) ? num(endpointTokS) : null,
      power: num(n.power_w),
      util: num(n.gpu_util_pct),
      temp: num(n.temperature_c),
    };
    next[n.id] = cur.length >= keep ? [...cur.slice(cur.length - keep + 1), sample] : [...cur, sample];
  }
  return next;
}

export function sameLiveRun(a: LiveRun | null, b: LiveRun | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    a.tok_s === b.tok_s &&
    a.peak === b.peak &&
    a.running === b.running &&
    a.waiting === b.waiting &&
    a.source === b.source
  );
}

export const useLabStatusStore = create<LabStatusStore>((set, get) => ({
  status: null,
  loading: true,
  needToken: false,
  unreachable: false,
  error: null,
  lastGoodAt: null,
  samples: {},
  liveRun: null,
  refresh: async () => {
    try {
      const status = await api.labStatus();
      const now = Date.now();
      const nodes = status.cluster?.nodes ?? status.serve?.cluster?.nodes;
      set((s) => ({
        status,
        loading: false,
        needToken: false,
        unreachable: false,
        error: null,
        lastGoodAt: now,
        samples: pushNodeSamples(s.samples, nodes, now, NODE_SAMPLES_KEEP, status.serve?.metrics?.decode_tok_per_s),
      }));
    } catch (e) {
      const unauthorized = isUnauthorizedError(e);
      set({
        loading: false,
        needToken: unauthorized,
        unreachable: !unauthorized,
        error: unauthorized ? null : String((e as Error).message || e),
      });
    }
  },
  setLiveRun: (liveRun) => {
    if (sameLiveRun(get().liveRun, liveRun)) return;
    set({ liveRun });
  },
}));

/** Read-only view for pages. Re-renders on every poll, like a page-local poll did. */
export function useLabStatus() {
  return useLabStatusStore();
}

/**
 * Start the poll. Returns a stop function. Only AppShell calls this; ticks
 * pause while `document.hidden` and resume (with an immediate tick) on return.
 */
export function startLabStatusPolling(): () => void {
  const { refresh } = useLabStatusStore.getState();
  let timer: ReturnType<typeof setInterval> | null = null;

  const stop = () => {
    if (timer) clearInterval(timer);
    timer = null;
  };
  const start = () => {
    if (timer || document.hidden) return;
    void refresh();
    timer = setInterval(() => void refresh(), LAB_STATUS_POLL_MS);
  };
  const onVisibility = () => (document.hidden ? stop() : start());

  start();
  document.addEventListener("visibilitychange", onVisibility);
  return () => {
    stop();
    document.removeEventListener("visibilitychange", onVisibility);
  };
}

/** Serve is healthy when the engine answered, was reachable, and reports a live endpoint. */
export function serveHealthy(status: LabStatus | null): boolean {
  const s = status?.serve;
  return !!(s && !s.unreachable && s.healthy);
}
