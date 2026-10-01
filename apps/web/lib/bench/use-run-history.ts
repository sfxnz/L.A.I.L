"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, type RunRow } from "../api";
import type { StreamRunRow } from "../stream-run-types";
import { comparable, headlineFromIndex, headlineFromResult, resultFromEnvelope, resultFromSnapshot, type BenchResult } from "./result";

/**
 * The history strand: bench runs from the serve-engine index (`/api/runs`),
 * plus controller runs whose import into that index failed (`/api/streams/runs`
 * with `saved_run_id: null`), newest first. Envelopes load lazily and are cached
 * for the session so the ghost, the Status card and the strand share one fetch.
 */

export type Headline = { c1: number | null; peak: number | null; peakAt: number | null; sustained: number | null };

export type HistoryEntry = {
  id: string;
  source: "engine" | "controller";
  createdAt: string;
  model: string;
  headline: Headline;
  result: BenchResult | null;
  envelope: Record<string, unknown> | null;
  failed: boolean;
};

type Loaded = { result: BenchResult | null; envelope: Record<string, unknown> | null };
const cache = new Map<string, Promise<Loaded>>();

export function loadEntry(id: string, source: "engine" | "controller"): Promise<Loaded> {
  const key = `${source}:${id}`;
  let p = cache.get(key);
  if (!p) {
    p =
      source === "engine"
        ? api.run(id).then((r) => ({ result: resultFromEnvelope(r.index, r.envelope), envelope: r.envelope }))
        : api.streamRunSnapshot(id).then((snap) => ({ result: resultFromSnapshot(id, snap), envelope: null }));
    p.catch(() => cache.delete(key));
    cache.set(key, p);
  }
  return p;
}

function fromIndex(row: RunRow): HistoryEntry {
  return {
    id: row.run_id,
    source: "engine",
    createdAt: row.created_at,
    model: row.model_id ?? "",
    headline: headlineFromIndex(row),
    result: null,
    envelope: null,
    failed: false,
  };
}

function fromController(row: StreamRunRow): HistoryEntry {
  const h = row.summary?.headline;
  return {
    id: row.run_id,
    source: "controller",
    createdAt: row.started_at,
    model: row.model,
    headline: {
      c1: h?.decode_tok_per_s_median_c1 ?? null,
      peak: h?.aggregate_peak_tok_per_s ?? null,
      peakAt: h?.aggregate_peak_concurrency ?? null,
      sustained: h?.prefill_tok_per_s_sustained ?? null,
    },
    result: null,
    envelope: null,
    failed: false,
  };
}

export async function fetchHistory(kind: "decode" | "prefill", limit: number): Promise<HistoryEntry[]> {
  const [index, controller] = await Promise.all([
    api.runs({ kind, limit }).catch(() => [] as RunRow[]),
    api.listStreamRuns().catch(() => [] as StreamRunRow[]),
  ]);
  const saved = new Set(index.map((r) => r.run_id));
  const orphans = controller.filter(
    (r) => r.mode === `bench-${kind}` && r.status === "done" && (!r.saved_run_id || !saved.has(r.saved_run_id)),
  );
  return [...index.map(fromIndex), ...orphans.map(fromController)]
    .sort((a, b) => (a.createdAt < b.createdAt ? 1 : a.createdAt > b.createdAt ? -1 : 0))
    .slice(0, limit);
}

export function useRunHistory(kind: "decode" | "prefill", limit = 12) {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    const list = await fetchHistory(kind, limit);
    if (!alive.current) return;
    setEntries(list);
    setLoading(false);
    // Envelopes: newest first, so the ghost and the strand's pack/levels fill in top-down.
    for (const e of list) {
      void loadEntry(e.id, e.source)
        .then((l) => {
          if (!alive.current) return;
          setEntries((prev) =>
            prev.map((x) =>
              x.id === e.id
                ? { ...x, result: l.result, envelope: l.envelope, headline: l.result ? headlineFromResult(l.result) : x.headline }
                : x,
            ),
          );
        })
        .catch(() => {
          if (alive.current) setEntries((prev) => prev.map((x) => (x.id === e.id ? { ...x, failed: true } : x)));
        });
    }
  }, [kind, limit]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { entries, loading, refresh };
}

/** Most recent earlier run that is like for like (`comparable`) — the ghost behind a result. */
export function previousOf(
  entries: HistoryEntry[],
  current: Pick<BenchResult, "id" | "savedRunId" | "pack" | "model" | "createdAt" | "maxTokens" | "fingerprint"> | null,
): HistoryEntry | null {
  if (!current) return null;
  for (const e of entries) {
    if (!e.result) continue;
    if (e.id === current.id || e.id === current.savedRunId) continue;
    if (current.createdAt && e.createdAt > current.createdAt) continue;
    if (!comparable(e.result, current)) continue;
    return e;
  }
  return null;
}
