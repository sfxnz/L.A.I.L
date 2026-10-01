import type { RunRow } from "../api";
import { decodeConfigToQuery, defaultDecodeConfig } from "./levels";
import { comparable, headlineFromIndex, headlineFromResult, type BenchResult, type DecodeArm } from "./result";

/**
 * The Status "Last synchronization" card, as data: the newest decode run of the
 * model being served, its hero, the delta against the previous comparable run
 * (same model, pack, tokens/stream and serve fingerprint — never just "the run
 * before it"), and where its two actions go. Envelopes are optional — the index
 * row's summary carries the headline and the comparability keys.
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
  /** Created-at of the run the delta compares against. */
  comparedTo: string | null;
  arms: DecodeArm[];
  previousArms: DecodeArm[] | null;
  openHref: string;
  runAgainHref: string;
};

/** Comparability keys straight from an index row (the controller writes them into `summary`). */
function indexKey(r: RunRow) {
  const s = (r.summary ?? {}) as Record<string, unknown>;
  return {
    model: r.model_id ?? "",
    pack: typeof s.pack === "string" ? s.pack : "",
    maxTokens: typeof s.max_tokens === "number" ? s.max_tokens : null,
    fingerprint: typeof s.serve_fingerprint === "string" && s.serve_fingerprint ? s.serve_fingerprint : null,
  };
}

/**
 * The newest index row created before `current` (and not `current` itself) that is like
 * for like with it — the Bench ghost's baseline, found the way Status finds its delta.
 */
export function previousComparable(
  rows: RunRow[],
  current: { id: string | null; savedRunId: string | null; model: string; pack: string; maxTokens: number | null; fingerprint: string | null; createdAt: string | null },
): RunRow | null {
  return (
    rows
      .filter((r) => r.run_id !== current.id && r.run_id !== current.savedRunId && (!current.createdAt || r.created_at <= current.createdAt))
      .sort((a, b) => (a.created_at < b.created_at ? 1 : a.created_at > b.created_at ? -1 : 0))
      .find((r) => comparable(indexKey(r), current)) ?? null
  );
}

/**
 * The newest decode run for `servingModel` (any model when nothing is served) and the
 * newest earlier run comparable to it.
 */
export function latestDecodeRuns(rows: RunRow[], servingModel: string | null): { cur: RunRow | null; prev: RunRow | null } {
  const decode = rows
    .filter((r) => r.kind === "decode" && (!servingModel || r.model_id === servingModel))
    .sort((a, b) => (a.created_at < b.created_at ? 1 : a.created_at > b.created_at ? -1 : 0));
  const cur = decode[0] ?? null;
  if (!cur) return { cur: null, prev: null };
  const key = indexKey(cur);
  return { cur, prev: decode.slice(1).find((r) => comparable(indexKey(r), key)) ?? null };
}

export function lastSync(
  rows: RunRow[],
  servingModel: string | null,
  envelopes: { current: BenchResult | null; previous: BenchResult | null },
): LastSync | null {
  const { cur, prev } = latestDecodeRuns(rows, servingModel);
  if (!cur) return null;
  const c = envelopes.current?.kind === "decode" ? envelopes.current : null;
  const p = envelopes.previous?.kind === "decode" ? envelopes.previous : null;
  const h = c ? headlineFromResult(c) : headlineFromIndex(cur);
  const ph = prev ? (p ? headlineFromResult(p) : headlineFromIndex(prev)) : null;
  const delta = h.peak !== null && ph?.peak && ph.peak > 0 ? (h.peak - ph.peak) / ph.peak : null;
  const levels = c?.levels ?? [];
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
    comparedTo: prev?.created_at ?? null,
    arms: c?.arms ?? [],
    previousArms: p?.arms ?? null,
    openHref: `/bench?run=${encodeURIComponent(cur.run_id)}`,
    runAgainHref: `/bench?${decodeConfigToQuery(cfg)}`,
  };
}
