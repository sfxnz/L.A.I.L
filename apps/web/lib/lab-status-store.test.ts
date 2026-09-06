/**
 * The shared lab-status store: the per-node 60 s ring buffer (the ONE place
 * status ticks accumulate) and `setLiveRun`'s change detection (a fresh object
 * per render must not count as a store change — that looped hydration once).
 */
import { describe, expect, test } from "bun:test";
import type { ClusterNode } from "./api";
import { NODE_SAMPLES_KEEP, pushNodeSamples, sameLiveRun, useLabStatusStore, type LiveRun } from "./lab-status-store";

const node = (id: string, tok: number | null, power: number | null = 30): ClusterNode => ({
  id,
  label: id,
  gen_tok_per_s: tok,
  power_w: power,
  gpu_util_pct: 50,
  temperature_c: 45,
});

describe("node sample ring buffer", () => {
  test("appends one sample per node per tick, keyed by id", () => {
    let s = pushNodeSamples({}, [node("spark1", 40), node("spark2", 40)], 1000);
    s = pushNodeSamples(s, [node("spark1", 42), node("spark2", 41)], 3000);
    expect(Object.keys(s).sort()).toEqual(["spark1", "spark2"]);
    expect(s.spark1.map((x) => x.tok_s)).toEqual([40, 42]);
    expect(s.spark2[1]).toEqual({ t: 3000, tok_s: 41, power: 30, util: 50, temp: 45 });
  });

  test("keeps the last NODE_SAMPLES_KEEP points (30 × 2 s = 60 s), oldest first", () => {
    let s: ReturnType<typeof pushNodeSamples> = {};
    for (let i = 0; i < NODE_SAMPLES_KEEP + 7; i++) s = pushNodeSamples(s, [node("spark1", i)], i * 2000);
    expect(s.spark1).toHaveLength(NODE_SAMPLES_KEEP);
    expect(s.spark1[0].tok_s).toBe(7);
    expect(s.spark1[NODE_SAMPLES_KEEP - 1].tok_s).toBe(NODE_SAMPLES_KEEP + 6);
  });

  test("a repeated tick (same t) is not duplicated; missing readings are null, never 0", () => {
    let s = pushNodeSamples({}, [node("spark1", 40)], 1000);
    s = pushNodeSamples(s, [node("spark1", 99)], 1000);
    expect(s.spark1).toHaveLength(1);
    s = pushNodeSamples(s, [node("spark1", null, null)], 3000);
    expect(s.spark1[1].tok_s).toBeNull();
    expect(s.spark1[1].power).toBeNull();
  });

  test("no nodes → the previous buffer is returned untouched", () => {
    const prev = pushNodeSamples({}, [node("spark1", 1)], 1);
    expect(pushNodeSamples(prev, undefined, 2)).toBe(prev);
    expect(pushNodeSamples(prev, [], 2)).toBe(prev);
  });
});

describe("setLiveRun change detection", () => {
  const run: LiveRun = { tok_s: 48.2, peak: 61, running: 4, waiting: 0, source: "streams" };

  test("sameLiveRun compares by value, null-safe", () => {
    expect(sameLiveRun(null, null)).toBe(true);
    expect(sameLiveRun(run, { ...run })).toBe(true);
    expect(sameLiveRun(run, { ...run, tok_s: 48.3 })).toBe(false);
    expect(sameLiveRun(run, { ...run, source: "bench" })).toBe(false);
    expect(sameLiveRun(run, null)).toBe(false);
  });

  test("an equal fresh object does not write; a changed one does; null clears", () => {
    const store = useLabStatusStore;
    let writes = 0;
    const unsub = store.subscribe(() => writes++);
    store.getState().setLiveRun(run);
    expect(writes).toBe(1);
    store.getState().setLiveRun({ ...run });
    expect(writes).toBe(1);
    expect(store.getState().liveRun).toBe(run); // same reference kept
    store.getState().setLiveRun({ ...run, running: 3 });
    expect(writes).toBe(2);
    store.getState().setLiveRun(null);
    store.getState().setLiveRun(null);
    expect(writes).toBe(3);
    expect(store.getState().liveRun).toBeNull();
    unsub();
  });
});
