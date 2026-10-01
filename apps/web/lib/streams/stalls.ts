import { percentile, perTokenLatencies } from "@lail/shared";

/**
 * Stall + desync rules for a strand. Evidence, not decoration: a stall is a
 * measured gap between decode steps; desync is an error state or a live gap past
 * DESYNC_MS. Both thresholds come from the design brief (§5.4).
 */

/** Step gap (no output for this long) that draws a stall marker. */
export const STALL_MS = 2000;
/** Live gap (no delta) after which a decoding strand is frozen as desync. */
export const DESYNC_MS = 5000;

/** Indices into a strand's `step_ms` whose gap is a stall. */
export function stallIndices(itlMs: readonly number[] | undefined, thresholdMs = STALL_MS): number[] {
  if (!itlMs) return [];
  const out: number[] = [];
  for (let k = 0; k < itlMs.length; k++) if (itlMs[k] >= thresholdMs) out.push(k);
  return out;
}

export type ItlStats = { p50: number | null; p95: number | null; max: number | null; stalls: number };

/**
 * Per-token ITL p50/p95 from decode steps (gap ÷ tokens in the step, weighted by
 * tokens — `perTokenLatencies`), the longest step gap, and stalls (step gaps ≥
 * STALL_MS: no output for that long, however many tokens followed).
 */
export function itlStats(stepMs: readonly number[] | undefined, stepTokens?: readonly number[]): ItlStats {
  if (!stepMs || !stepMs.length) return { p50: null, p95: null, max: null, stalls: 0 };
  const lat = perTokenLatencies(stepMs, stepTokens ?? stepMs.map(() => 1));
  return {
    p50: percentile(lat, 50),
    p95: percentile(lat, 95),
    max: Math.max(...stepMs),
    stalls: stallIndices(stepMs).length,
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
