import {
  TAIL_MIN_SAMPLES,
  median,
  percentile,
  type BenchArm,
  type BenchEnvelope,
  type BenchHardware,
  type BenchHeadline,
  type BenchPerRequest,
  type BenchRange,
  type ServerLevelMetrics,
  type StreamLevelEvent,
} from "@lail/shared";

/**
 * The bench's metric definitions — the only ones (the serve-engine decode bench is
 * retired). Per strand, with t_first/t_last the first/last token-bearing chunk:
 *   TTFT            t_first − t_start
 *   decode tok/s    (n − n_first) / (t_last − t_first)      n_first = tokens in the first chunk
 *   TPOT            (t_last − t_first) / (n − n_first)       = 1 / decode tok/s
 * Per wave (one burst of c strands): wall-clock aggregate and decode-span aggregate below.
 * Per level (⌈samples ÷ c⌉ waves): medians over waves / pooled strands, with min–max.
 * Percentiles: nearest rank (`@lail/shared` `percentile`); p95/p99 only from TAIL_MIN_SAMPLES.
 */

/** Timing record of one finished (or failed) strand. Times are `performance.now()` ms. */
export type StrandResult = {
  i: number;
  ok: boolean;
  t_start: number;
  /** First token-bearing output chunk (content or reasoning). */
  t_first: number | null;
  /** Last token-bearing output chunk. */
  t_last: number | null;
  /** Stream closed or failed. */
  t_end: number;
  /** `usage.completion_tokens`, or the chunk count when the usage frame is missing. */
  completion_tokens: number | null;
  /** Tokens delivered with the first chunk (at t_first, so outside the decode span). */
  first_tokens: number;
  prompt_tokens: number | null;
  estimated: boolean;
  finish_reason: string | null;
  error: string | null;
  token_times_ms: number[];
  token_counts: number[];
  wave?: number;
};

export type WaveKey = { concurrency?: number; size?: number };

export type WaveSummary = {
  ok: number;
  requests: number;
  errors: string[];
  tokens: number;
  aggregate_tok_s: number | null;
  aggregate_steady_tok_s: number | null;
  per_stream_median_tok_s: number | null;
  ttft_s: { p50: number | null; p95: number | null; p99: number | null };
  tpot_s: number | null;
  prompt_tokens: number | null;
  prefill_tok_s: number | null;
  samples: number;
  aggregate_range: BenchRange | null;
  per_stream_range: BenchRange | null;
};

/** Per-level context measured around the waves (not from the strands). */
export type LevelContext = { foreign_max: number | null; server: ServerLevelMetrics | null };

export { median, percentile };

/** t_last − t_first in s; null when the strand never produced a second token-bearing chunk. */
export function decodeTimeS(r: StrandResult): number | null {
  if (r.t_first === null || r.t_last === null || r.t_last <= r.t_first) return null;
  return (r.t_last - r.t_first) / 1000;
}

/** Tokens decoded inside the decode span: n − n_first. */
function decodedTokens(r: StrandResult): number | null {
  const n = r.completion_tokens;
  if (!n || n <= 0) return null;
  const d = n - Math.max(0, r.first_tokens);
  return d > 0 ? d : null;
}

/** (n − n_first) / (t_last − t_first). */
export function decodeTokPerS(r: StrandResult): number | null {
  const ds = decodeTimeS(r);
  const d = decodedTokens(r);
  return ds === null || d === null ? null : d / ds;
}

/** t_first − t_start. */
export function ttftS(r: StrandResult): number | null {
  return r.t_first === null ? null : (r.t_first - r.t_start) / 1000;
}

/** (t_last − t_first) / (n − n_first) = 1 / decodeTokPerS. */
export function tpotS(r: StrandResult): number | null {
  const ds = decodeTimeS(r);
  const d = decodedTokens(r);
  return ds === null || d === null ? null : ds / d;
}

/** prompt_tokens / TTFT. Meaningful for the long prompts of the prefill bench only. */
export function prefillTokPerS(r: StrandResult): number | null {
  const ttft = ttftS(r);
  const n = r.prompt_tokens;
  if (!n || n <= 0 || !ttft || ttft <= 0) return null;
  return n / ttft;
}

/**
 * Wave aggregate, wall-clock: Σ completion_tokens (ok strands) / (max t_last − min t_start).
 * Spans every strand's prefill and the slowest strand's tail, so at ×N it can read below ×1
 * when one straggler decodes alone at the end.
 */
export function aggregateTokPerS(results: StrandResult[]): number | null {
  const ok = results.filter((r) => r.ok && r.t_last !== null);
  if (!ok.length) return null;
  const tokens = ok.reduce((a, r) => a + (r.completion_tokens || 0), 0);
  const span = (Math.max(...ok.map((r) => r.t_last as number)) - Math.min(...ok.map((r) => r.t_start))) / 1000;
  return span > 0 && tokens > 0 ? tokens / span : null;
}

/**
 * Wave aggregate, decode span: Σ (n − n_first) / (max t_last − min t_first) over ok strands —
 * what the server delivered while it was emitting tokens; the time-average of the live
 * window rate. At ×1 it equals the strand's decode tok/s.
 */
export function aggregateSteadyTokPerS(results: StrandResult[]): number | null {
  const ok = results.filter((r) => r.ok && r.t_first !== null && r.t_last !== null);
  if (!ok.length) return null;
  const tokens = ok.reduce((a, r) => a + (decodedTokens(r) ?? 0), 0);
  const span = (Math.max(...ok.map((r) => r.t_last as number)) - Math.min(...ok.map((r) => r.t_first as number))) / 1000;
  return span > 0 && tokens > 0 ? tokens / span : null;
}

const r2 = (v: number | null) => (v === null ? null : Math.round(v * 100) / 100);
const r3 = (v: number | null) => (v === null ? null : Math.round(v * 1000) / 1000);
const ms1 = (s: number | null) => (s === null ? null : Math.round(s * 10000) / 10);
const nn = (v: number | null): v is number => v !== null;
const range = (vals: number[]): BenchRange | null =>
  vals.length >= 2 ? [r2(Math.min(...vals))!, r2(Math.max(...vals))!] : null;

/** One wave. */
export function summarizeWave(results: StrandResult[]): WaveSummary {
  return summarizeLevel([results]);
}

/**
 * A level of one or more waves. Aggregates are per wave (a wave is one burst; pooling waves
 * would count the gaps between them), reported as the median with min–max. Per-strand
 * numbers pool every ok strand of the level.
 */
export function summarizeLevel(waves: StrandResult[][]): WaveSummary {
  const all = waves.flat();
  const ok = all.filter((r) => r.ok);
  const ttfts = ok.map(ttftS).filter(nn).sort((a, b) => a - b);
  const rates = ok.map(decodeTokPerS).filter(nn);
  const tpots = ok.map(tpotS).filter(nn);
  const prefill = ok.map(prefillTokPerS).filter(nn);
  const prompts = ok.map((r) => r.prompt_tokens).filter(nn);
  const aggs = waves.map(aggregateTokPerS).filter(nn);
  const steadies = waves.map(aggregateSteadyTokPerS).filter(nn);
  const tail = ttfts.length >= TAIL_MIN_SAMPLES;
  const tpot = median(tpots);
  return {
    ok: ok.length,
    requests: all.length,
    errors: all.filter((r) => !r.ok).map((r) => `#${r.i}: ${r.error || "failed"}`),
    tokens: ok.reduce((a, r) => a + (r.completion_tokens || 0), 0),
    aggregate_tok_s: r2(median(aggs)),
    aggregate_steady_tok_s: r2(median(steadies)),
    per_stream_median_tok_s: r2(median(rates)),
    ttft_s: {
      p50: r3(percentile(ttfts, 50)),
      p95: tail ? r3(percentile(ttfts, 95)) : null,
      p99: tail ? r3(percentile(ttfts, 99)) : null,
    },
    tpot_s: tpot === null ? null : Math.round(tpot * 1e6) / 1e6,
    prompt_tokens: prompts.length ? Math.round(median(prompts)!) : null,
    prefill_tok_s: r2(median(prefill)),
    samples: waves.length,
    aggregate_range: range(aggs),
    per_stream_range: range(rates.length ? rates : prefill),
  };
}

export function toLevelEvent(index: number, key: WaveKey, ws: WaveSummary, ctx?: LevelContext): StreamLevelEvent {
  const ev: StreamLevelEvent = {
    type: "level",
    index,
    ...key,
    aggregate_tok_s: ws.aggregate_tok_s,
    aggregate_steady_tok_s: ws.aggregate_steady_tok_s,
    per_stream_median_tok_s: ws.per_stream_median_tok_s,
    samples: ws.samples,
    aggregate_range: ws.aggregate_range,
    per_stream_range: ws.per_stream_range,
    ttft_p50_ms: ms1(ws.ttft_s.p50),
    ttft_p95_ms: ms1(ws.ttft_s.p95),
    ttft_p99_ms: ms1(ws.ttft_s.p99),
    tpot_ms: ws.tpot_s === null ? null : Math.round(ws.tpot_s * 100000) / 100,
    ok: ws.ok,
    requests: ws.requests,
    errors: ws.errors,
  };
  if (key.size !== undefined) {
    ev.prompt_tokens = ws.prompt_tokens;
    ev.prefill_tok_s = ws.prefill_tok_s;
  }
  if (ctx) {
    ev.foreign_max = ctx.foreign_max;
    ev.server = ctx.server;
  }
  return ev;
}

export function skippedLevelEvent(index: number, key: WaveKey, reason: string): StreamLevelEvent {
  return {
    type: "level",
    index,
    ...key,
    aggregate_tok_s: null,
    aggregate_steady_tok_s: null,
    per_stream_median_tok_s: null,
    ttft_p50_ms: null,
    ttft_p95_ms: null,
    ttft_p99_ms: null,
    tpot_ms: null,
    ok: 0,
    requests: 0,
    errors: [],
    skipped: reason,
  };
}

export function toArm(key: WaveKey, ws: WaveSummary | null, skipped?: string, ctx?: LevelContext): BenchArm {
  const arm: BenchArm = {
    ...key,
    ok: ws?.ok ?? 0,
    requests: ws?.requests ?? 0,
    ttft_s: ws?.ttft_s ?? { p50: null, p95: null, p99: null },
    aggregate_tok_per_s: ws?.aggregate_tok_s ?? null,
    aggregate_steady_tok_per_s: ws?.aggregate_steady_tok_s ?? null,
    decode_tok_per_s_median: ws?.per_stream_median_tok_s ?? null,
    tpot_s: ws?.tpot_s ?? null,
    errors: ws?.errors ?? [],
    samples: ws?.samples ?? 0,
    aggregate_range: ws?.aggregate_range ?? null,
    per_stream_range: ws?.per_stream_range ?? null,
    foreign_max: ctx?.foreign_max ?? null,
    server: ctx?.server ?? null,
  };
  if (key.size !== undefined) {
    arm.prompt_tokens = ws?.prompt_tokens ?? null;
    arm.prefill_tok_per_s = ws?.prefill_tok_s ?? null;
  }
  if (skipped) arm.skipped = skipped;
  return arm;
}

export function toPerRequest(r: StrandResult): BenchPerRequest {
  return {
    i: r.i,
    ttft_s: r3(ttftS(r)),
    decode_s: r3(decodeTimeS(r)),
    completion_tokens: r.completion_tokens,
    prompt_tokens: r.prompt_tokens,
    tok_per_s: r2(decodeTokPerS(r)),
    finish_reason: r.finish_reason,
    error: r.error,
    token_times_ms: r.token_times_ms,
    token_counts: r.token_counts,
    ...(r.wave !== undefined ? { wave: r.wave } : {}),
    ...(r.estimated ? { estimated: true } : {}),
  };
}

export type BenchArmInput = { key: WaveKey; ws: WaveSummary | null; results: StrandResult[]; skipped?: string; ctx?: LevelContext };

/** The decode level with the highest wall-clock aggregate. */
export function peakArm(arms: BenchArm[]): BenchArm | null {
  let peak: BenchArm | null = null;
  for (const a of arms) {
    if (a.aggregate_tok_per_s != null && (peak === null || a.aggregate_tok_per_s > (peak.aggregate_tok_per_s ?? 0))) peak = a;
  }
  return peak;
}

export function headline(kind: "decode" | "prefill", arms: BenchArm[]): BenchHeadline {
  if (kind === "prefill") {
    const done = arms.filter((a) => a.ok > 0 && a.prefill_tok_per_s != null);
    const largest = done.length ? done.reduce((a, b) => ((b.size ?? 0) > (a.size ?? 0) ? b : a)) : null;
    return {
      decode_tok_per_s_median_c1: null,
      aggregate_peak_tok_per_s: null,
      aggregate_peak_concurrency: null,
      prefill_tok_per_s_sustained: largest?.prefill_tok_per_s ?? null,
      ttft_p50_s_c1: null,
    };
  }
  const c1 = arms.find((a) => a.concurrency === 1) ?? arms[0];
  const peak = peakArm(arms);
  return {
    decode_tok_per_s_median_c1: c1?.decode_tok_per_s_median ?? null,
    aggregate_peak_tok_per_s: peak?.aggregate_tok_per_s ?? null,
    aggregate_peak_concurrency: peak?.concurrency ?? null,
    ttft_p50_s_c1: c1?.ttft_s.p50 ?? null,
  };
}

export function buildEnvelope(opts: {
  kind: "decode" | "prefill";
  model: string;
  workload: BenchEnvelope["workload"];
  arms: BenchArmInput[];
  hardware?: BenchHardware | null;
}): BenchEnvelope {
  const arms = opts.arms.map((a) => toArm(a.key, a.ws, a.skipped, a.ctx));
  const full_arms = opts.arms.map((a, idx) => ({ ...arms[idx], per_request: a.results.map(toPerRequest) }));
  const head = headline(opts.kind, arms);
  return {
    kind: opts.kind,
    model: opts.model,
    workload: opts.workload,
    metrics: { arms, full_arms, headline: head, hardware: opts.hardware ?? null },
    summary: {
      ...head,
      pack: opts.workload.pack,
      max_tokens: opts.workload.max_tokens,
      serve_fingerprint: opts.workload.serve_fingerprint,
      energy_j_per_token: opts.hardware?.energy_j_per_token ?? null,
    },
    source: "controller-streams",
  };
}
