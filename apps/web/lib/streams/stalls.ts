/**
 * Stall + desync rules for a strand. Evidence, not decoration: a stall is a
 * measured inter-token gap; desync is an error state or a live gap past
 * DESYNC_MS. Both thresholds come from the design brief (§5.4).
 */

/** Inter-token gap that draws a stall marker. */
export const STALL_MS = 2000;
/** Live gap (no delta) after which a decoding strand is frozen as desync. */
export const DESYNC_MS = 5000;

/** Indices into `itl_ms` whose gap is a stall. */
export function stallIndices(itlMs: readonly number[] | undefined, thresholdMs = STALL_MS): number[] {
  if (!itlMs) return [];
  const out: number[] = [];
  for (let k = 0; k < itlMs.length; k++) if (itlMs[k] >= thresholdMs) out.push(k);
  return out;
}

/** Nearest-rank percentile over an ascending array (mirrors the engine's metrics.ts). */
export function percentile(sorted: readonly number[], p: number): number | null {
  const n = sorted.length;
  if (!n) return null;
  const rank = Math.ceil((p / 100) * n);
  return sorted[Math.min(n - 1, Math.max(0, rank - 1))];
}

export type ItlStats = { p50: number | null; p95: number | null; max: number | null; stalls: number };

export function itlStats(itlMs: readonly number[] | undefined): ItlStats {
  if (!itlMs || !itlMs.length) return { p50: null, p95: null, max: null, stalls: 0 };
  const sorted = [...itlMs].sort((a, b) => a - b);
  return {
    p50: percentile(sorted, 50),
    p95: percentile(sorted, 95),
    max: sorted[sorted.length - 1],
    stalls: stallIndices(itlMs).length,
  };
}

export type DesyncInput = {
  state: string;
  error?: string;
  /** client stamp of the last delta, if any */
  at_last_delta?: number;
  /** client stamp when decode was first observed */
  at_decode?: number;
};

export type Desync = { desync: boolean; reason: string | null; stalledMs: number };

/**
 * `error` desyncs immediately. A decoding strand with no delta for DESYNC_MS is
 * a desync too. Prefill is exempt: a long prompt legitimately waits > 5 s for
 * its first token.
 */
export function strandDesync(s: DesyncInput, nowAt: number): Desync {
  if (s.state === "error") return { desync: true, reason: s.error || "error", stalledMs: 0 };
  if (s.state !== "decode") return { desync: false, reason: null, stalledMs: 0 };
  const last = s.at_last_delta ?? s.at_decode;
  if (last === undefined) return { desync: false, reason: null, stalledMs: 0 };
  const stalledMs = Math.max(0, nowAt - last);
  if (stalledMs >= DESYNC_MS) {
    return { desync: true, reason: `no token for ${(stalledMs / 1000).toFixed(1)} s`, stalledMs };
  }
  return { desync: false, reason: null, stalledMs };
}
