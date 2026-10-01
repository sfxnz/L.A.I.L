/**
 * The shared lab-status store: the per-node 60 s ring buffer (the ONE place
 * status ticks accumulate) and `setLiveRun`'s change detection (a fresh object
 * per render must not count as a store change — that looped hydration once).
 */
import { describe, expect, test } from "bun:test";
import type { ClusterNode } from "./api";
import {
  NODE_SAMPLES_KEEP,
  ownsEndpoint,
  pushNodeSamples,
  sameLiveRun,
  tightestNode,
  useLabStatusStore,
  type LiveRun,
} from "./lab-status-store";

const node = (id: string, power: number | null = 30, state = "serving", local = id === "spark1"): ClusterNode => ({
  id,
  label: id,
  state,
  local,
  power_w: power,
  gpu_util_pct: 50,
  temperature_c: 45,
});
const K = NODE_SAMPLES_KEEP;

describe("node sample ring buffer", () => {
  test("appends one sample per node per tick; the endpoint rate lands on the serving head only", () => {
    const tp = () => [node("spark1"), node("spark2", 30, "serving_worker")];
    let s = pushNodeSamples({}, tp(), 1000, K, 40);
    s = pushNodeSamples(s, tp(), 3000, K, 42);
    expect(Object.keys(s).sort()).toEqual(["spark1", "spark2"]);
    expect(s.spark1.map((x) => x.tok_s)).toEqual([40, 42]);
    // a TP worker serves the same tokens: it never carries a second copy of the rate
    expect(s.spark2[1]).toEqual({ t: 3000, tok_s: null, power: 30, util: 50, temp: 45 });
  });

  test("a remote node serving its own endpoint never carries the LOCAL endpoint's rate", () => {
    // e.g. multi_mismatch: spark2 runs a single-node serve the serve-engine does not probe
    const s = pushNodeSamples({}, [node("spark1", 30, "idle"), node("spark2", 30, "serving")], 1000, K, 40);
    expect(s.spark1[0].tok_s).toBeNull();
    expect(s.spark2[0].tok_s).toBeNull();
    expect(ownsEndpoint(node("spark2", 30, "serving"))).toBe(false);
    expect(ownsEndpoint(node("spark1", 30, "serving"))).toBe(true);
    expect(ownsEndpoint(node("spark1", 30, "serving_worker"))).toBe(false);
  });

  test("keeps the last NODE_SAMPLES_KEEP points (30 × 2 s = 60 s), oldest first", () => {
    let s: ReturnType<typeof pushNodeSamples> = {};
    for (let i = 0; i < K + 7; i++) s = pushNodeSamples(s, [node("spark1")], i * 2000, K, i);
    expect(s.spark1).toHaveLength(K);
    expect(s.spark1[0].tok_s).toBe(7);
    expect(s.spark1[K - 1].tok_s).toBe(K + 6);
  });

  test("a repeated tick (same t) is not duplicated; missing readings are null, never 0", () => {
    let s = pushNodeSamples({}, [node("spark1")], 1000, K, 40);
    s = pushNodeSamples(s, [node("spark1")], 1000, K, 99);
    expect(s.spark1).toHaveLength(1);
    s = pushNodeSamples(s, [node("spark1", null)], 3000, K, null);
    expect(s.spark1[1].tok_s).toBeNull();
    expect(s.spark1[1].power).toBeNull();
  });

  test("no nodes → the previous buffer is returned untouched", () => {
    const prev = pushNodeSamples({}, [node("spark1")], 1, K, 1);
    expect(pushNodeSamples(prev, undefined, 2)).toBe(prev);
    expect(pushNodeSamples(prev, [], 2)).toBe(prev);
  });
});

describe("tightestNode", () => {
  const mem = (id: string, available_gib: number | null, extra: Partial<ClusterNode> = {}): ClusterNode => ({
    id,
    label: id,
    available_gib,
    ...extra,
  });

  test("the live node with the least MemAvailable — the rank that runs out first", () => {
    const nodes = [mem("spark1", 14.2, { local: true }), mem("spark2", 12.9, { online: true })];
    expect(tightestNode(nodes)?.id).toBe("spark2");
  });

  test("offline nodes and missing readings are skipped; nothing live → null", () => {
    const nodes = [mem("spark1", 14.2, { local: true }), mem("spark2", 3.0, { online: false }), mem("spark3", null, { online: true })];
    expect(tightestNode(nodes)?.id).toBe("spark1");
    expect(tightestNode([mem("spark2", 3.0, { online: false })])).toBeNull();
    expect(tightestNode(undefined)).toBeNull();
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
