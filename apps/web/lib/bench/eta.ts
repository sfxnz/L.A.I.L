/**
 * ETA from measured rates. Decode: remaining ≈ Σ over remaining levels of
 * (max_tokens ÷ predicted per-stream tok/s + predicted TTFT), plus what is left
 * of the current wave; padded 12 %, rounded to 5 s. Prefill: TTFT roughly
 * doubles per context doubling, so the next size ≈ ratio × the last measured
 * TTFT. `reviseEta` makes the shown number fall freely and rise only when the
 * new estimate is clearly larger.
 */

export const ETA_PAD = 1.12;
export const ETA_RISE_HYSTERESIS = 1.25;

export type MeasuredLevel = { concurrency: number; perStream: number | null; ttftMs: number | null };

/**
 * Per-stream tok/s at concurrency `c`: a power law fitted through the last two
 * measured levels (per-stream ∝ c^−β), or the last measurement when there is
 * only one. Never above the ×1 rate.
 */
export function predictPerStream(measured: MeasuredLevel[], c: number): number | null {
  const ms = measured.filter((m) => m.perStream !== null && m.perStream > 0).sort((a, b) => a.concurrency - b.concurrency);
  if (!ms.length) return null;
  const last = ms[ms.length - 1];
  if (ms.length === 1) return last.perStream;
  const prev = ms[ms.length - 2];
  if (last.concurrency === prev.concurrency) return last.perStream;
  const beta = -Math.log((last.perStream as number) / (prev.perStream as number)) / Math.log(last.concurrency / prev.concurrency);
  const predicted = (last.perStream as number) * Math.pow(c / last.concurrency, -Math.max(0, beta));
  return Math.min(predicted, ms[0].perStream as number);
}

export function predictTtftMs(measured: MeasuredLevel[], c: number): number {
  const ms = measured.filter((m) => m.ttftMs !== null).sort((a, b) => a.concurrency - b.concurrency);
  if (!ms.length) return 0;
  const last = ms[ms.length - 1];
  // TTFT grows roughly with the number of prompts prefilled together.
  return (last.ttftMs as number) * Math.max(1, c / last.concurrency) * 0.8;
}

export type CurrentWave = {
  concurrency: number;
  /** tokens emitted so far per strand (live estimate) */
  tokens: number[];
  /** live per-stream rate estimate (median of running strands), tok/s */
  rateTokS: number | null;
  /** strands that have not produced a first token yet */
  waitingFirstToken: number;
};

export function etaDecodeMs(opts: {
  levels: number[];
  maxTokens: number;
  measured: MeasuredLevel[];
  current: CurrentWave | null;
}): number | null {
  const { levels, maxTokens, measured, current } = opts;
  // Before the first level settles, the live rate of the current wave is the only basis.
  const basis: MeasuredLevel[] =
    measured.length || !current?.rateTokS
      ? measured
      : [{ concurrency: current.concurrency, perStream: current.rateTokS, ttftMs: null }];
  let ms = 0;

  if (current) {
    const rate = current.rateTokS ?? predictPerStream(basis, current.concurrency);
    if (!rate || rate <= 0) return null;
    const slowest = current.tokens.length ? Math.min(...current.tokens) : 0;
    ms += (Math.max(0, maxTokens - slowest) / rate) * 1000;
    if (current.waitingFirstToken > 0) ms += predictTtftMs(basis, current.concurrency);
  }
  const remaining = levels.slice(measured.length + (current ? 1 : 0));
  for (const c of remaining) {
    const rate = predictPerStream(basis, c);
    if (rate === null || rate <= 0) return null;
    ms += (maxTokens / rate) * 1000 + predictTtftMs(basis, c);
  }
  if (!current && !remaining.length) return 0;
  return Math.round(ms * ETA_PAD);
}

export type MeasuredSize = { size: number; ttftMs: number | null };

/** Growth of TTFT per context doubling, from the last two measured sizes; 2 until there are two. */
export function doublingRatio(measured: MeasuredSize[]): number {
  const ms = measured.filter((m) => m.ttftMs !== null && m.ttftMs > 0).sort((a, b) => a.size - b.size);
  if (ms.length < 2) return 2;
  const a = ms[ms.length - 2];
  const b = ms[ms.length - 1];
  if (b.size <= a.size) return 2;
  const r = Math.pow((b.ttftMs as number) / (a.ttftMs as number), 1 / Math.log2(b.size / a.size));
  return Math.min(4, Math.max(1, r));
}

export function predictTtftForSize(measured: MeasuredSize[], size: number): number | null {
  const ms = measured.filter((m) => m.ttftMs !== null && m.ttftMs > 0).sort((a, b) => a.size - b.size);
  if (!ms.length) return null;
  const last = ms[ms.length - 1];
  const octaves = Math.log2(size / last.size);
  return (last.ttftMs as number) * Math.pow(doublingRatio(measured), octaves);
}

export function etaPrefillMs(opts: {
  sizes: number[];
  measured: MeasuredSize[];
  skipped: ReadonlySet<number>;
  current: { size: number; elapsedMs: number } | null;
}): number | null {
  const { sizes, measured, skipped, current } = opts;
  let ms = 0;
  let known = false;
  const pending = sizes.filter((s) => !skipped.has(s) && !measured.some((m) => m.size === s));
  for (const size of pending) {
    const predicted = predictTtftForSize(measured, size);
    if (predicted === null) return null;
    const left = current && current.size === size ? Math.max(0, predicted - current.elapsedMs) : predicted;
    ms += left;
    known = true;
  }
  return known ? Math.round(ms * ETA_PAD) : 0;
}

/** Falls freely; rises only past the hysteresis so the number never creeps upward. */
export function reviseEta(shown: number | null, next: number | null): number | null {
  if (next === null) return shown;
  if (shown === null) return next;
  if (next <= shown) return next;
  return next > shown * ETA_RISE_HYSTERESIS ? next : shown;
}

/** "~45 s", "~1 m 10 s" — rounded to 5 s, never under 5 s while positive. */
export function fmtEta(ms: number | null): string {
  if (ms === null) return "";
  const s = Math.max(5, Math.round(ms / 5000) * 5);
  if (s < 60) return `~${s} s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  return r ? `~${m} m ${r} s` : `~${m} m`;
}
