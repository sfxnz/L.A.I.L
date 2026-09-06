import type {
  BenchArm,
  BenchEnvelope,
  BenchHeadline,
  BenchPerRequest,
  StreamLevelEvent,
  StreamThinking,
} from "@lail/shared";

/** Timing record of one finished (or failed) strand. Times are `performance.now()` ms. */
export type StrandResult = {
  i: number;
  ok: boolean;
  t_start: number;
  /** First output chunk (content or reasoning). */
  t_first: number | null;
  /** Last output chunk. */
  t_last: number | null;
  /** Stream closed or failed. */
  t_end: number;
  /** `usage.completion_tokens`, or the chunk count when the usage frame is missing. */
  completion_tokens: number | null;
  prompt_tokens: number | null;
  estimated: boolean;
  finish_reason: string | null;
  error: string | null;
  token_times_ms: number[];
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
};

// ── Definitions mirror serve-engine perf.py ─────────────────────────

/** Nearest-rank percentile over an ascending array. */
export function percentile(sorted: number[], p: number): number | null {
  const n = sorted.length;
  if (!n) return null;
  const rank = Math.ceil((p / 100) * n);
  return sorted[Math.min(n - 1, Math.max(0, rank - 1))];
}

export function median(vals: number[]): number | null {
  if (!vals.length) return null;
  const s = [...vals].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/** First emitted token → last. Falls back to stream end when only one chunk arrived. */
export function decodeTimeS(r: StrandResult): number | null {
  if (r.t_first === null) return null;
  let end = r.t_last;
  if (end === null || end <= r.t_first) end = r.t_end;
  const ds = (end - r.t_first) / 1000;
  return ds > 0 ? ds : null;
}

/** completion_tokens / (t_last − t_first). */
export function decodeTokPerS(r: StrandResult): number | null {
  const ds = decodeTimeS(r);
  const n = r.completion_tokens;
  if (ds === null || !n || n <= 0) return null;
  return n / ds;
}

/** t_first − t_start. */
export function ttftS(r: StrandResult): number | null {
  return r.t_first === null ? null : (r.t_first - r.t_start) / 1000;
}

/** (t_last − t_first) / (completion_tokens − 1). */
export function tpotS(r: StrandResult): number | null {
  const n = r.completion_tokens;
  if (r.t_first === null || r.t_last === null || !n || n < 2) return null;
  const span = (r.t_last - r.t_first) / 1000;
  return span > 0 ? span / (n - 1) : null;
}

/** prompt_tokens / TTFT. */
export function prefillTokPerS(r: StrandResult): number | null {
  const ttft = ttftS(r);
  const n = r.prompt_tokens;
  if (!n || n <= 0 || !ttft || ttft <= 0) return null;
  return n / ttft;
}

/**
 * Wave aggregate, wall-clock: Σ completion_tokens (ok strands) / (max t_last − min t_start).
 * Same definition as perf.py. The window spans every strand's prefill and the slowest
 * strand's tail, so at ×N it can read below ×1 when one straggler decodes alone at the end.
 */
export function aggregateTokPerS(results: StrandResult[]): number | null {
  const ok = results.filter((r) => r.ok && r.t_last !== null);
  if (!ok.length) return null;
  const tokens = ok.reduce((a, r) => a + (r.completion_tokens || 0), 0);
  const span = (Math.max(...ok.map((r) => r.t_last as number)) - Math.min(...ok.map((r) => r.t_start))) / 1000;
  return span > 0 && tokens > 0 ? tokens / span : null;
}

/**
 * Wave aggregate, steady-state: Σ of per-strand decode rates (completion_tokens / (t_last − t_first)).
 * Each strand is measured only over its own decode window, so this is what the server
 * sustained while strands were decoding — an upper bound that excludes TTFT and straggler tails.
 */
export function aggregateSteadyTokPerS(results: StrandResult[]): number | null {
  const rates = results.filter((r) => r.ok).map(decodeTokPerS).filter((v): v is number => v !== null);
  return rates.length ? rates.reduce((a, b) => a + b, 0) : null;
}

const r2 = (v: number | null) => (v === null ? null : Math.round(v * 100) / 100);
const r3 = (v: number | null) => (v === null ? null : Math.round(v * 1000) / 1000);
const ms1 = (s: number | null) => (s === null ? null : Math.round(s * 10000) / 10);

export function summarizeWave(results: StrandResult[]): WaveSummary {
  const ok = results.filter((r) => r.ok);
  const ttfts = ok.map(ttftS).filter((v): v is number => v !== null).sort((a, b) => a - b);
  const rates = ok.map(decodeTokPerS).filter((v): v is number => v !== null);
  const tpots = ok.map(tpotS).filter((v): v is number => v !== null);
  const prefill = ok.map(prefillTokPerS).filter((v): v is number => v !== null);
  const prompts = ok.map((r) => r.prompt_tokens).filter((v): v is number => v !== null);
  return {
    ok: ok.length,
    requests: results.length,
    errors: results.filter((r) => !r.ok).map((r) => `#${r.i}: ${r.error || "failed"}`),
    tokens: ok.reduce((a, r) => a + (r.completion_tokens || 0), 0),
    aggregate_tok_s: r2(aggregateTokPerS(results)),
    aggregate_steady_tok_s: r2(aggregateSteadyTokPerS(results)),
    per_stream_median_tok_s: r2(median(rates)),
    ttft_s: { p50: r3(percentile(ttfts, 50)), p95: r3(percentile(ttfts, 95)), p99: r3(percentile(ttfts, 99)) },
    tpot_s: tpots.length ? Math.round(median(tpots)! * 1e6) / 1e6 : null,
    prompt_tokens: prompts.length ? Math.round(median(prompts)!) : null,
    prefill_tok_s: r2(median(prefill)),
  };
}

export function toLevelEvent(index: number, key: WaveKey, ws: WaveSummary): StreamLevelEvent {
  return {
    type: "level",
    index,
    ...key,
    aggregate_tok_s: ws.aggregate_tok_s,
    aggregate_steady_tok_s: ws.aggregate_steady_tok_s,
    per_stream_median_tok_s: ws.per_stream_median_tok_s,
    ttft_p50_ms: ms1(ws.ttft_s.p50),
    ttft_p95_ms: ms1(ws.ttft_s.p95),
    ttft_p99_ms: ms1(ws.ttft_s.p99),
    tpot_ms: ws.tpot_s === null ? null : Math.round(ws.tpot_s * 100000) / 100,
    ok: ws.ok,
    requests: ws.requests,
    errors: ws.errors,
    prompt_tokens: ws.prompt_tokens,
    prefill_tok_s: ws.prefill_tok_s,
  };
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

export function toArm(key: WaveKey, ws: WaveSummary | null, skipped?: string): BenchArm {
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
    ...(r.estimated ? { estimated: true } : {}),
  };
}

export type BenchArmInput = { key: WaveKey; ws: WaveSummary | null; results: StrandResult[]; skipped?: string };

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
  let peak: BenchArm | null = null;
  for (const a of arms) {
    if (a.aggregate_tok_per_s !== null && (peak === null || a.aggregate_tok_per_s > (peak.aggregate_tok_per_s ?? 0))) peak = a;
  }
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
  workload: {
    pack: string;
    levels?: number[];
    sizes?: number[];
    max_tokens: number;
    thinking: StreamThinking;
    temperature: number;
    fill_to_max: boolean;
    base_url: string;
  };
  arms: BenchArmInput[];
}): BenchEnvelope {
  const arms = opts.arms.map((a) => toArm(a.key, a.ws, a.skipped));
  const full_arms = opts.arms.map((a, idx) => ({ ...arms[idx], per_request: a.results.map(toPerRequest) }));
  const head = headline(opts.kind, arms);
  return {
    kind: opts.kind,
    model: opts.model,
    workload: opts.workload,
    metrics: { arms, full_arms, headline: head },
    summary: head,
    source: "controller-streams",
  };
}
