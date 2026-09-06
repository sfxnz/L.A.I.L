/**
 * One result shape for the bench, whatever fed it: the live `level` events of a
 * running sync, a serve-engine envelope from history (`GET /api/runs/{id}`,
 * both the controller's `metrics.arms` and the older perf.py arms), or a
 * controller snapshot when the import into the run index failed.
 * Times are ms here; the envelope keeps seconds.
 */
import type { BenchArm, BenchHeadline, StreamRunSnapshot, StreamRunSummary } from "@lail/shared";
import type { RunRow } from "../api";
import type { LevelRow } from "../use-stream-run";

export type DecodeArm = {
  concurrency: number;
  aggregate: number | null;
  perStream: number | null;
  ttftP50: number | null;
  ttftP95: number | null;
  ttftP99: number | null;
  tpotMs: number | null;
  ok: number;
  requests: number;
  errors: string[];
};

export type PrefillArm = {
  size: number;
  promptTokens: number | null;
  prefillTokS: number | null;
  ttftMs: number | null;
  ok: number;
  requests: number;
  errors: string[];
  skipped: string | null;
};

export type ResultMeta = {
  /** Serve-engine run id when saved; controller run id otherwise. */
  id: string | null;
  savedRunId: string | null;
  model: string;
  pack: string;
  maxTokens: number | null;
  createdAt: string | null;
  durationMs: number | null;
  engine: string | null;
  source: "live" | "history" | "controller";
};

export type DecodeResult = ResultMeta & { kind: "decode"; levels: number[]; arms: DecodeArm[] };
export type PrefillResult = ResultMeta & { kind: "prefill"; sizes: number[]; arms: PrefillArm[] };
export type BenchResult = DecodeResult | PrefillResult;

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const sToMs = (v: unknown): number | null => {
  const n = num(v);
  return n === null ? null : Math.round(n * 10000) / 10;
};
const strs = (v: unknown): string[] => (Array.isArray(v) ? v.map(String) : []);

export function peakArm(arms: DecodeArm[]): DecodeArm | null {
  let best: DecodeArm | null = null;
  for (const a of arms) {
    if (a.aggregate !== null && (best === null || a.aggregate > (best.aggregate ?? 0))) best = a;
  }
  return best;
}

export function c1Arm(arms: DecodeArm[]): DecodeArm | null {
  return arms.find((a) => a.concurrency === 1) ?? arms[0] ?? null;
}

/** Largest completed size with a prefill rate — the "sustained" number. */
export function sustainedArm(arms: PrefillArm[]): PrefillArm | null {
  let best: PrefillArm | null = null;
  for (const a of arms) {
    if (a.prefillTokS !== null && !a.skipped && (best === null || a.size > best.size)) best = a;
  }
  return best;
}

// ── Live events ─────────────────────────────────────────────────────

export function decodeArmsFromLevels(levels: LevelRow[]): DecodeArm[] {
  return levels
    .filter((l) => typeof l.concurrency === "number")
    .map((l) => ({
      concurrency: l.concurrency as number,
      aggregate: num(l.aggregate_tok_s),
      perStream: num(l.per_stream_median_tok_s),
      ttftP50: num(l.ttft_p50_ms),
      ttftP95: num(l.ttft_p95_ms),
      ttftP99: num(l.ttft_p99_ms),
      tpotMs: num(l.tpot_ms),
      ok: l.ok,
      requests: l.requests,
      errors: l.errors ?? [],
    }));
}

export function prefillArmsFromLevels(levels: LevelRow[]): PrefillArm[] {
  return levels
    .filter((l) => typeof l.size === "number")
    .map((l) => ({
      size: l.size as number,
      promptTokens: num(l.prompt_tokens),
      prefillTokS: num(l.prefill_tok_s),
      ttftMs: num(l.ttft_p50_ms),
      ok: l.ok,
      requests: l.requests,
      errors: l.errors ?? [],
      skipped: l.skipped ?? null,
    }));
}

export function decodeResultFromLive(opts: {
  runId: string;
  model: string;
  pack: string;
  levels: number[];
  maxTokens: number;
  startedAt: string;
  rows: LevelRow[];
  summary: StreamRunSummary | null;
  savedRunId: string | null;
}): DecodeResult {
  return {
    kind: "decode",
    id: opts.savedRunId ?? opts.runId,
    savedRunId: opts.savedRunId,
    model: opts.model,
    pack: opts.pack,
    maxTokens: opts.maxTokens,
    createdAt: opts.startedAt,
    durationMs: opts.summary?.duration_ms ?? null,
    engine: null,
    source: "live",
    levels: opts.levels,
    arms: decodeArmsFromLevels(opts.rows),
  };
}

export function prefillResultFromLive(opts: {
  runId: string;
  model: string;
  pack: string;
  sizes: number[];
  startedAt: string;
  rows: LevelRow[];
  summary: StreamRunSummary | null;
  savedRunId: string | null;
}): PrefillResult {
  return {
    kind: "prefill",
    id: opts.savedRunId ?? opts.runId,
    savedRunId: opts.savedRunId,
    model: opts.model,
    pack: opts.pack,
    maxTokens: 1,
    createdAt: opts.startedAt,
    durationMs: opts.summary?.duration_ms ?? null,
    engine: null,
    source: "live",
    sizes: opts.sizes,
    arms: prefillArmsFromLevels(opts.rows),
  };
}

// ── Envelopes (serve-engine run index) ──────────────────────────────

type Env = Record<string, unknown>;

function envArms(envelope: Env): Record<string, unknown>[] {
  const metrics = envelope.metrics as Env | undefined;
  const arms = metrics?.arms;
  return Array.isArray(arms) ? (arms as Record<string, unknown>[]) : [];
}

function envWorkload(envelope: Env): Env {
  const w = envelope.workload;
  return w && typeof w === "object" ? (w as Env) : {};
}

function envModel(envelope: Env, index?: RunRow): string {
  const m = envelope.model;
  if (m && typeof m === "object" && typeof (m as Env).id === "string") return (m as Env).id as string;
  if (typeof m === "string") return m;
  return index?.model_id ?? "";
}

function envEngine(envelope: Env): string | null {
  const e = envelope.engine;
  if (!e || typeof e !== "object") return null;
  const name = (e as Env).name;
  const version = (e as Env).version;
  if (typeof name !== "string") return null;
  return typeof version === "string" && version ? `${name} ${version}` : name;
}

/** The perf.py bench wrote `workload` as the family id; the controller writes `workload.pack`. */
function envPack(envelope: Env, index?: RunRow): string {
  const w = envWorkload(envelope);
  if (typeof w.pack === "string") return w.pack;
  if (typeof w.kind === "string") return w.kind;
  const s = index?.summary;
  if (s && typeof s.workload === "string") return s.workload;
  return "";
}

function decodeArmFromEnvelope(a: Record<string, unknown>): DecodeArm | null {
  const c = num(a.concurrency);
  if (c === null) return null;
  const ttft = (a.ttft_s ?? {}) as Env;
  return {
    concurrency: c,
    aggregate: num(a.aggregate_tok_per_s),
    perStream: num(a.decode_tok_per_s_median),
    ttftP50: sToMs(ttft.p50),
    ttftP95: sToMs(ttft.p95),
    ttftP99: sToMs(ttft.p99),
    tpotMs: sToMs(a.tpot_s ?? a.tpot_s_median),
    ok: num(a.ok) ?? 0,
    requests: num(a.requests) ?? 0,
    errors: strs(a.errors),
  };
}

function prefillArmFromEnvelope(a: Record<string, unknown>): PrefillArm | null {
  const size = num(a.size);
  if (size === null) return null;
  const ttft = (a.ttft_s ?? {}) as Env;
  return {
    size,
    promptTokens: num(a.prompt_tokens),
    prefillTokS: num(a.prefill_tok_per_s),
    ttftMs: sToMs(ttft.p50),
    ok: num(a.ok) ?? 0,
    requests: num(a.requests) ?? 0,
    errors: strs(a.errors),
    skipped: typeof a.skipped === "string" ? a.skipped : null,
  };
}

export function resultFromEnvelope(index: RunRow, envelope: Env | null): BenchResult | null {
  if (!envelope) return null;
  const kind = envelope.kind ?? index.kind;
  const w = envWorkload(envelope);
  const meta = {
    id: index.run_id,
    savedRunId: index.run_id,
    model: envModel(envelope, index),
    pack: envPack(envelope, index),
    maxTokens: num(w.max_tokens),
    createdAt: (typeof envelope.created_at === "string" ? envelope.created_at : null) ?? index.created_at ?? null,
    durationMs: null,
    engine: envEngine(envelope),
    source: "history" as const,
  };
  const arms = envArms(envelope);
  if (kind === "prefill") {
    const pa = arms.map(prefillArmFromEnvelope).filter((a): a is PrefillArm => a !== null);
    const sizes = Array.isArray(w.sizes) ? (w.sizes as number[]) : pa.map((a) => a.size);
    return { ...meta, kind: "prefill", sizes, arms: pa };
  }
  const da = arms.map(decodeArmFromEnvelope).filter((a): a is DecodeArm => a !== null);
  if (!da.length && kind !== "decode") return null;
  const levels = Array.isArray(w.levels)
    ? (w.levels as number[])
    : Array.isArray(w.concurrencies)
      ? (w.concurrencies as number[])
      : da.map((a) => a.concurrency);
  return { ...meta, kind: "decode", levels, arms: da };
}

/** A controller run whose import failed: rebuild the view from its snapshot. */
export function resultFromSnapshot(runId: string, snap: StreamRunSnapshot): BenchResult | null {
  const h = snap.hello;
  const base = {
    id: runId,
    savedRunId: snap.done?.saved_run_id ?? null,
    model: h.model,
    pack: h.prompts[0]?.pack ?? "",
    maxTokens: h.max_tokens,
    createdAt: h.started_at,
    durationMs: snap.done?.summary.duration_ms ?? null,
    engine: null,
    source: "controller" as const,
  };
  const rows = snap.levels.map((l) => ({ ...l }));
  if (h.mode === "bench-decode") {
    return { ...base, kind: "decode", levels: h.levels ?? [], arms: decodeArmsFromLevels(rows) };
  }
  if (h.mode === "bench-prefill") {
    return { ...base, kind: "prefill", sizes: h.sizes ?? [], arms: prefillArmsFromLevels(rows) };
  }
  return null;
}

// ── Index rows (no envelope yet) ────────────────────────────────────

export type HeadlineSummary = Partial<BenchHeadline> & {
  workload?: string;
  concurrencies?: number[];
  [k: string]: unknown;
};

/** Peak aggregate @ ×N straight from the index row, before its envelope is loaded. */
export function headlineFromIndex(row: RunRow): {
  c1: number | null;
  peak: number | null;
  peakAt: number | null;
  sustained: number | null;
} {
  const s = (row.summary ?? {}) as HeadlineSummary;
  return {
    c1: num(s.decode_tok_per_s_median_c1),
    peak: num(s.aggregate_peak_tok_per_s),
    peakAt: num(s.aggregate_peak_concurrency),
    sustained: num(s.prefill_tok_per_s_sustained),
  };
}

/** The same numbers from a full result, so history rows and hero agree. */
export function headlineFromResult(r: BenchResult): {
  c1: number | null;
  peak: number | null;
  peakAt: number | null;
  sustained: number | null;
} {
  if (r.kind === "prefill") {
    const s = sustainedArm(r.arms);
    return { c1: null, peak: null, peakAt: null, sustained: s?.prefillTokS ?? null };
  }
  const p = peakArm(r.arms);
  return {
    c1: c1Arm(r.arms)?.perStream ?? null,
    peak: p?.aggregate ?? null,
    peakAt: p?.concurrency ?? null,
    sustained: null,
  };
}

/** Compatibility with the retired decode-bench index rows (perf.py wrote `summary.workload`). */
export function decodeRunLabel(summary: Record<string, unknown> | undefined): string | null {
  const w = summary?.workload;
  if (typeof w !== "string") return null;
  const label: Record<string, string> = { structured: "Structured", prose: "Prose", code: "Code", json: "JSON" };
  return label[w] ?? null;
}

export type { BenchArm };
