import type { RunRow } from "../api";
import { decodeConfigToQuery, defaultDecodeConfig } from "./levels";
import { headlineFromIndex, headlineFromResult, type BenchResult, type DecodeArm } from "./result";

/**
 * The Status "Last synchronization" card, as data: newest decode run, its
 * hero, the delta against the run before it, and where its two actions go.
 * Envelopes are optional — the index row's summary carries the headline.
 */

export type LastSync = {
  id: string;
  createdAt: string;
  model: string;
  pack: string | null;
  levels: number[];
  peak: number | null;
  peakAt: number | null;
  c1: number | null;
  /** (peak − previous peak) ÷ previous peak, or null without a comparable previous run */
  delta: number | null;
  arms: DecodeArm[];
  previousArms: DecodeArm[] | null;
  openHref: string;
  runAgainHref: string;
};

export function latestDecodeRuns(rows: RunRow[], n = 2): RunRow[] {
  return rows
    .filter((r) => r.kind === "decode")
    .sort((a, b) => (a.created_at < b.created_at ? 1 : a.created_at > b.created_at ? -1 : 0))
    .slice(0, n);
}

export function lastSync(rows: RunRow[], envelopes: { current: BenchResult | null; previous: BenchResult | null }): LastSync | null {
  const [cur, prev] = latestDecodeRuns(rows);
  if (!cur) return null;
  const c = envelopes.current?.kind === "decode" ? envelopes.current : null;
  const p = envelopes.previous?.kind === "decode" ? envelopes.previous : null;
  const h = c ? headlineFromResult(c) : headlineFromIndex(cur);
  const ph = prev ? (p ? headlineFromResult(p) : headlineFromIndex(prev)) : null;
  const delta = h.peak !== null && ph?.peak && ph.peak > 0 ? (h.peak - ph.peak) / ph.peak : null;
  const levels = c?.levels ?? (Array.isArray(cur.summary?.concurrencies) ? (cur.summary.concurrencies as number[]) : []);
  const cfg = {
    ...defaultDecodeConfig(),
    ...(c?.pack ? { pack: c.pack } : {}),
    ...(levels.length ? { levels } : {}),
    ...(c?.maxTokens ? { maxTokens: c.maxTokens } : {}),
  };
  return {
    id: cur.run_id,
    createdAt: cur.created_at,
    model: c?.model || cur.model_id || "",
    pack: c?.pack ?? null,
    levels,
    peak: h.peak,
    peakAt: h.peakAt,
    c1: h.c1,
    delta,
    arms: c?.arms ?? [],
    previousArms: p?.arms ?? null,
    openHref: `/bench?run=${encodeURIComponent(cur.run_id)}`,
    runAgainHref: `/bench?${decodeConfigToQuery(cfg)}`,
  };
}
