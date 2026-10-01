/**
 * Per-strand live tok/s, computed client-side from `delta.tokens` growth.
 *
 * The engine's `agg.tok_s` is the ONE aggregate number (exact usage tokens in its
 * sliding window). The stacked sparkline needs that number split per strand, so
 * each strand keeps `(at, cumulative tokens)` samples — one per coalesced delta —
 * and its rate over a sliding window is token growth ÷ the time it grew over.
 * `stackShares` then scales the split so the stack's top edge IS the engine
 * aggregate: bands are the client-observed share of the one reconciled number,
 * never a second unexplained total.
 */

export type RateSample = { at: number; tokens: number };

/** Sliding window for per-strand rates (1.5 s smooths 80 ms coalescing). */
export const RATE_WINDOW_MS = 1500;
/** Samples older than this fall off the ring. */
export const RATE_KEEP_MS = 3000;

export function pushSample(samples: readonly RateSample[], at: number, tokens: number, keepMs = RATE_KEEP_MS): RateSample[] {
  const floor = at - keepMs;
  const kept = samples.length && samples[0].at < floor ? samples.filter((s) => s.at >= floor) : samples.slice();
  kept.push({ at, tokens });
  return kept;
}

/**
 * Tokens per second over `(at − windowMs, at]`: token growth since the baseline —
 * the last sample at or before the window start — divided by the time since that
 * baseline (a coalesced sample carries tokens since the previous one, so the growth
 * spans back to the baseline, not to the window start). With no sample that old,
 * the earliest sample is the baseline over its own (shorter) span, so a strand's
 * first second reads a rate instead of zero. A strand that stopped emitting decays
 * to 0 as `at` advances past its last sample.
 */
export function windowRate(samples: readonly RateSample[], at: number, windowMs = RATE_WINDOW_MS): number {
  if (samples.length < 2) return 0;
  const latest = samples[samples.length - 1];
  const start = at - windowMs;
  let base: RateSample | null = null;
  for (let k = samples.length - 1; k >= 0; k--) {
    if (samples[k].at <= start) {
      base = samples[k];
      break;
    }
  }
  const baseline = base ?? samples[0];
  const spanMs = Math.max(250, at - baseline.at);
  const grown = Math.max(0, latest.tokens - baseline.tokens);
  const rate = grown / (spanMs / 1000);
  return Math.round(rate * 10) / 10;
}

/**
 * Scale per-strand rates so they sum to the engine aggregate. If the client saw
 * no growth but the engine reports throughput (window phase), the aggregate is
 * split evenly across `active` strands so the stack never shows a false zero.
 */
export function stackShares(rates: readonly number[], aggregateTokS: number, active: readonly boolean[]): number[] {
  const sum = rates.reduce((a, r) => a + r, 0);
  if (aggregateTokS <= 0) return rates.map(() => 0);
  if (sum > 0) {
    const k = aggregateTokS / sum;
    return rates.map((r) => Math.round(r * k * 10) / 10);
  }
  const n = active.filter(Boolean).length;
  if (!n) return rates.map(() => 0);
  const each = Math.round((aggregateTokS / n) * 10) / 10;
  return rates.map((_, i) => (active[i] ? each : 0));
}
