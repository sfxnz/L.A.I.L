/**
 * The one set of statistic definitions the bench engine (controller) and the web
 * share. Pure; hand-computable.
 */

/** Nearest-rank percentile over an ascending array. */
export function percentile(sorted: readonly number[], p: number): number | null {
  const n = sorted.length;
  if (!n) return null;
  const rank = Math.ceil((p / 100) * n);
  return sorted[Math.min(n - 1, Math.max(0, rank - 1))];
}

/** Middle value (mean of the two middle values for an even count). */
export function median(vals: readonly number[]): number | null {
  if (!vals.length) return null;
  const s = [...vals].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/**
 * Live rates (the engine's `agg.tok_s` and per-strand `tok_s`): tokens that arrived in
 * this window ÷ the window. Several seconds, so the 1–4-token chunks of speculative
 * decoding (±1 chunk at a window edge) move the number by ~1 tok/s, not ±4, and a
 * "peak" is not the luckiest single second.
 */
export const LIVE_RATE_WINDOW_MS = 3000;

/** Tail percentiles (p95/p99) are reported only from this many samples up; below it they read as noise. */
export const TAIL_MIN_SAMPLES = 5;

/**
 * Per-token inter-token latencies from decode steps. One upstream chunk is one
 * scheduler step and, with speculative (MTP) decoding, carries several tokens:
 * a step of `gap` ms that delivered `k` tokens contributes `k` samples of
 * `gap / k` ms. Steps with no token count contribute nothing. Ascending.
 */
export function perTokenLatencies(stepMs: readonly number[], stepTokens: readonly number[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < stepMs.length; i++) {
    const k = stepTokens[i] ?? 1;
    if (!(k > 0)) continue;
    const v = stepMs[i] / k;
    for (let j = 0; j < k; j++) out.push(v);
  }
  return out.sort((a, b) => a - b);
}
