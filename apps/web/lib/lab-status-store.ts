"use client";

import { useEffect, useState } from "react";
import { create } from "zustand";
import { ApiError, api, type ClusterNode, type LabStatus } from "./api";
import { isUnauthorizedError } from "./auth-token";

/**
 * The ONE live lab status. AppShell starts the transport (lib/live-connection.ts):
 * the controller's /api/live stream pushes every 1 s serve-engine sample, with a
 * polling fallback through `refresh()`. Every page reads this store through
 * selectors, so a tick re-renders only what it changed. 401 (`needToken`) and
 * unreachable are kept apart — a fresh browser with no token is not an outage.
 *
 * Time: every timestamp in a snapshot (serve.sampled_at_ms, metrics.sampled_at,
 * node.sampled_at) is the serve-engine host's clock. The browser may be another
 * machine with another clock, so ages are measured against the server clock, and
 * the browser clock only measures how long ago a snapshot arrived (`receivedAt`).
 *
 * The store also keeps:
 *   · `samples` — 60 s of per-node hardware and endpoint rates, keyed by the
 *     server's sampled_at (a repeated snapshot never adds a point; a missed one
 *     leaves a gap) — node sparklines and the bench hardware strip draw these;
 *   · `liveRun` — strand counts while a Streams/Bench run streams in this tab.
 */

/** Polling fallback cadence (the stream pushes every sample on its own). */
export const LAB_STATUS_POLL_MS = 2000;
/** History kept per series: the 60 s every sparkline draws, plus a little slack. */
export const SAMPLES_KEEP_MS = 65_000;
/** The sampler publishes every second: older than 3 cadences is stale, not live. */
export const STALE_AFTER_S = 3;

export type LiveRun = {
  running: number;
  waiting: number;
  source: "streams" | "bench";
};

export type RailRate = { tx_bps: number; rx_bps: number };

export type NodeSample = {
  /** server epoch ms of the node's telemetry */
  t: number;
  power: number | null;
  util: number | null;
  temp: number | null;
  /** RAM in use: MemTotal − MemAvailable, GiB */
  mem: number | null;
  rails: Record<string, RailRate> | null;
};

export type EndpointSample = {
  /** server epoch ms of the /metrics scrape */
  t: number;
  decode: number | null;
  throughput: number | null;
};

export type Samples = {
  endpoint: EndpointSample[];
  nodes: Record<string, NodeSample[]>;
};

/** Static/slow fields the stream sends apart from the per-second tick. */
export type LiveMeta = {
  controller: string;
  defaultBackend: string;
  defaultModel: string;
  openAiBase: string;
  backends: LabStatus["backends"];
  models: Array<{ id: string; max_model_len?: number | null }> | null;
  version: { version?: string | null } | null;
  flags: string[] | null;
};

type Serve = NonNullable<LabStatus["serve"]>;

type LabStatusStore = {
  status: LabStatus | null;
  /** true until the first snapshot or failure */
  loading: boolean;
  /** controller answered 401 — paste LAIL_TOKEN */
  needToken: boolean;
  /** controller did not answer at all */
  unreachable: boolean;
  error: string | null;
  /** browser epoch ms when `status` arrived */
  receivedAt: number | null;
  /** how status is arriving: the live stream, or the polling fallback */
  transport: "stream" | "poll" | null;
  /**
   * The controller answers but serve-engine does not (or is warming up). `status`
   * then keeps the last real snapshot, which ages into "stale" — never replaced by
   * an empty one that would read as "no model serving".
   */
  engineError: string | null;
  samples: Samples;
  liveRun: LiveRun | null;
  /** last `meta` event of the stream (merged into every tick) */
  meta: LiveMeta | null;
  /** One poll of /api/lab-status. Never overlaps itself; an older answer never wins. */
  refresh: () => Promise<boolean>;
  /** Writes only when a value changed — a fresh object per render looped hydration once. */
  setLiveRun: (run: LiveRun | null) => void;
};

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export const EMPTY_SAMPLES: Samples = { endpoint: [], nodes: {} };

/**
 * The node whose endpoint serve.metrics describes: the serve-engine probes the endpoint
 * of the host it runs on, so only the LOCAL serving node owns those rates. A TP worker
 * serves the same tokens, and a remote node serving on its own is not what was probed.
 */
export function ownsEndpoint(n: ClusterNode): boolean {
  return n.state === "serving" && !!n.local;
}

/** A node that has a current reading (the local host, or a remote that answers). */
export function nodeLive(n: ClusterNode): boolean {
  return !!(n.local || n.online);
}

/**
 * The live node with the least MemAvailable — under TP, any one rank running out of
 * memory takes the serve down, so this is the number that matters. Offline nodes carry
 * no current reading and are skipped.
 */
export function tightestNode(nodes: ClusterNode[] | undefined): ClusterNode | null {
  let best: ClusterNode | null = null;
  for (const n of nodes ?? []) {
    if (!nodeLive(n) || n.available_gib == null) continue;
    if (best == null || n.available_gib < (best.available_gib as number)) best = n;
  }
  return best;
}

function pushSeries<T extends { t: number }>(cur: T[] | undefined, sample: T | null, floor: number): T[] | null {
  const list = cur ?? [];
  const last = list[list.length - 1];
  const add = sample && (!last || sample.t > last.t);
  const drop = list.length > 0 && list[0].t < floor;
  if (!add && !drop) return null;
  const kept = drop ? list.filter((s) => s.t >= floor) : list.slice();
  if (add) kept.push(sample);
  return kept;
}

/**
 * Fold one snapshot (a full status or a compact history entry) into the series (pure).
 * A point is added only when its server timestamp advanced — the same sample seen
 * twice adds nothing, and a sample never seen leaves a gap on the time axis. Series
 * older than SAMPLES_KEEP_MS before the snapshot are pruned. Returns `prev` itself
 * when nothing changed, so subscribers do not re-render.
 */
export function pushSamples(prev: Samples, serve: Partial<Serve> | null | undefined, keepMs = SAMPLES_KEEP_MS): Samples {
  const at = num(serve?.sampled_at_ms);
  if (!serve || at == null) return prev;
  const floor = at - keepMs;
  let changed = false;

  const m = serve.metrics;
  const mt = num(m?.sampled_at);
  const endpoint = pushSeries(
    prev.endpoint,
    mt != null ? { t: mt, decode: num(m?.decode_tok_per_s), throughput: num(m?.throughput_tok_per_s) } : null,
    floor,
  );
  if (endpoint) changed = true;

  const nodes: Record<string, NodeSample[]> = { ...prev.nodes };
  for (const n of serve.cluster?.nodes ?? []) {
    const t = num(n.sampled_at);
    const ram = num(n.ram_gib);
    const avail = num(n.available_gib);
    const next = pushSeries(
      prev.nodes[n.id],
      t != null
        ? {
            t,
            power: num(n.power_w),
            util: num(n.gpu_util_pct),
            temp: num(n.temperature_c),
            mem: ram != null && avail != null ? Math.max(0, ram - avail) : null,
            rails: n.rail_rates ?? null,
          }
        : null,
      floor,
    );
    if (next) {
      nodes[n.id] = next;
      changed = true;
    }
  }
  return changed ? { endpoint: endpoint ?? prev.endpoint, nodes } : prev;
}

/** A tick of the stream plus the last meta → the same LabStatus /api/lab-status returns. */
export function mergeTick(meta: LiveMeta | null, tick: { serve: Serve | null }): LabStatus {
  const serve = tick.serve;
  return {
    controller: meta?.controller ?? "ok",
    defaultBackend: meta?.defaultBackend ?? "",
    defaultModel: meta?.defaultModel ?? "",
    openAiBase: meta?.openAiBase ?? "",
    backends: meta?.backends ?? {},
    serve: serve
      ? {
          ...serve,
          models: meta?.models ?? undefined,
          version: meta?.version ?? undefined,
          engine: serve.engine ? { ...serve.engine, flags: meta?.flags ?? null } : serve.engine,
        }
      : serve,
  };
}

/** True when `next` is older than what the store already shows (an out-of-order answer). */
export function isOlder(cur: LabStatus | null, next: LabStatus): boolean {
  const a = num(cur?.serve?.sampled_at_ms);
  const b = num(next.serve?.sampled_at_ms);
  return a != null && b != null && b < a;
}

export function sameLiveRun(a: LiveRun | null, b: LiveRun | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return a.running === b.running && a.waiting === b.waiting && a.source === b.source;
}

let inflight: Promise<boolean> | null = null;

export const useLabStatusStore = create<LabStatusStore>((set, get) => ({
  status: null,
  loading: true,
  needToken: false,
  unreachable: false,
  error: null,
  receivedAt: null,
  transport: null,
  engineError: null,
  samples: EMPTY_SAMPLES,
  liveRun: null,
  meta: null,
  refresh: () => {
    if (inflight) return inflight;
    inflight = api
      .labStatus(AbortSignal.timeout(4000))
      .then((status) => {
        ingestStatus(status, "poll");
        return true;
      })
      .catch((e) => {
        ingestFailure(e);
        return false;
      })
      .finally(() => {
        inflight = null;
      });
    return inflight;
  },
  setLiveRun: (liveRun) => {
    if (sameLiveRun(get().liveRun, liveRun)) return;
    set({ liveRun });
  },
}));

/** A status whose serve block is serve-engine's failure (or warm-up), not a snapshot. */
export function engineFailure(status: LabStatus): string | null {
  const serve = status.serve;
  if (serve && num(serve.sampled_at_ms) != null) return null;
  return serve?.error || (serve ? null : "serve-engine did not answer");
}

/**
 * Show one snapshot (from either transport). An older one than shown is dropped.
 * A serve-engine failure keeps the last real snapshot on screen, aging (and so
 * dimmed as stale) with the failure noted, rather than blanking the instruments.
 */
export function ingestStatus(status: LabStatus, transport: "stream" | "poll"): void {
  const s = useLabStatusStore.getState();
  if (isOlder(s.status, status)) return;
  const engineError = engineFailure(status);
  const base = { loading: false, needToken: false, unreachable: false, error: null, transport, engineError };
  if (engineError && num(s.status?.serve?.sampled_at_ms) != null) {
    useLabStatusStore.setState(base);
    return;
  }
  useLabStatusStore.setState({
    ...base,
    status,
    receivedAt: Date.now(),
    samples: pushSamples(s.samples, status.serve),
  });
}

/** Fold the stream's `history` event (compact snapshots, oldest first) into the series. */
export function ingestHistory(entries: Array<{ serve?: Partial<Serve> }>): void {
  let samples = useLabStatusStore.getState().samples;
  for (const e of entries) samples = pushSamples(samples, e.serve);
  useLabStatusStore.setState({ samples });
}

export function ingestFailure(e: unknown): void {
  const unauthorized = isUnauthorizedError(e);
  // The web server's proxy answers a bare 5xx page when the controller is down.
  const proxied = e instanceof ApiError && !e.json && e.status >= 500;
  useLabStatusStore.setState({
    loading: false,
    needToken: unauthorized,
    unreachable: !unauthorized,
    error: unauthorized ? null : proxied ? `no answer from the controller (HTTP ${(e as ApiError).status})` : String((e as Error)?.message || e),
  });
}

/** Serve is healthy when the engine answered, was reachable, and reports a live endpoint. */
export function serveHealthy(status: LabStatus | null): boolean {
  const s = status?.serve;
  return !!(s && !s.unreachable && s.healthy);
}

// ── Freshness ────────────────────────────────────────────────────────────────

type Clock = Pick<LabStatusStore, "status" | "receivedAt">;

/** The serve-engine host's clock now, estimated from the last snapshot. */
export function serverNow(s: Clock, now = Date.now()): number | null {
  const pub = num(s.status?.serve?.sampled_at_ms);
  if (pub == null || s.receivedAt == null) return null;
  return pub + Math.max(0, now - s.receivedAt);
}

/** Seconds since the newest snapshot was sampled (its own age plus time since it arrived). */
export function snapshotAge(s: Clock, now = Date.now()): number | null {
  if (s.receivedAt == null || !s.status?.serve) return null;
  return Math.max(0, num(s.status.serve.stale_s) ?? 0) + Math.max(0, now - s.receivedAt) / 1000;
}

/** Seconds since a server-clock timestamp from the current snapshot (e.g. node.sampled_at). */
export function ageOf(s: Clock, serverMs: number | null | undefined, now = Date.now()): number | null {
  const sn = serverNow(s, now);
  if (sn == null || serverMs == null) return null;
  return Math.max(0, (sn - serverMs) / 1000);
}

/** True once the newest snapshot is older than `thresholdS` (no stream, no snapshot yet: false). */
export function isStale(s: Clock & { unreachable?: boolean }, thresholdS = STALE_AFTER_S, now = Date.now()): boolean {
  const age = snapshotAge(s, now);
  return !!s.unreachable || (age != null && age > thresholdS);
}

/** Whether the live data stopped updating. Checked every second; re-renders only when it flips. */
export function useStale(thresholdS = STALE_AFTER_S): boolean {
  const [stale, setStale] = useState(false);
  useEffect(() => {
    const check = () => setStale(isStale(useLabStatusStore.getState(), thresholdS));
    check();
    const t = setInterval(check, 1000);
    return () => clearInterval(t);
  }, [thresholdS]);
  return stale;
}

/** Re-renders the caller every `ms` (for age readouts that must move while nothing arrives). */
export function useNow(ms = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(t);
  }, [ms]);
  return now;
}
