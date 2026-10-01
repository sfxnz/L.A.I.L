/**
 * The shared lab-status store: samples keyed by the server's clock (a repeated
 * snapshot adds nothing, a missed one leaves a gap), the stream's meta + tick merge,
 * out-of-order protection, freshness against the server clock, and `setLiveRun`'s
 * change detection (a fresh object per render must not count as a store change).
 */
import { afterEach, describe, expect, test } from "bun:test";
import type { ClusterNode, LabStatus } from "./api";
import {
  EMPTY_SAMPLES,
  LAB_STATUS_POLL_MS,
  SAMPLES_KEEP_MS,
  STALE_AFTER_S,
  ingestStatus,
  isOlder,
  isStale,
  mergeTick,
  pushSamples,
  sameLiveRun,
  serverNow,
  snapshotAge,
  staleAfterS,
  tightestNode,
  useLabStatusStore,
  type LiveMeta,
  type LiveRun,
} from "./lab-status-store";

type Serve = NonNullable<LabStatus["serve"]>;

const node = (id: string, t: number | null, extra: Partial<ClusterNode> = {}): ClusterNode => ({
  id,
  label: id,
  state: "serving",
  local: id === "spark1",
  sampled_at: t,
  power_w: 30,
  gpu_util_pct: 50,
  temperature_c: 45,
  ram_gib: 121.7,
  available_gib: 15.2,
  ...extra,
});

const snap = (at: number, nodeT: Record<string, number | null>, metricsT: number | null = at - 10): Serve => ({
  sampled_at_ms: at,
  metrics: { sampled_at: metricsT, decode_tok_per_s: 80, throughput_tok_per_s: 0 },
  cluster: { nodes: Object.entries(nodeT).map(([id, t]) => node(id, t)) },
});

afterEach(() => {
  useLabStatusStore.setState({ status: null, receivedAt: null, samples: EMPTY_SAMPLES, liveRun: null, loading: true, transport: null, engineError: null });
});

describe("samples keyed by the server's sampled_at", () => {
  test("one point per new server timestamp: a repeated snapshot adds nothing and returns the same object", () => {
    let s = pushSamples(EMPTY_SAMPLES, snap(1000, { spark1: 990, spark2: 700 }));
    s = pushSamples(s, snap(2000, { spark1: 1990, spark2: 1700 }));
    const again = pushSamples(s, snap(2000, { spark1: 1990, spark2: 1700 }));
    expect(again).toBe(s); // no re-render for an unchanged sample
    expect(s.nodes.spark1.map((p) => p.t)).toEqual([990, 1990]);
    expect(s.nodes.spark2.map((p) => p.t)).toEqual([700, 1700]);
    expect(s.endpoint.map((p) => p.t)).toEqual([990, 1990]);
    // memory in use = MemTotal − MemAvailable
    expect(s.nodes.spark1[0].mem).toBeCloseTo(106.5, 5);
  });

  test("a node whose telemetry did not advance keeps its series; a down node (no reading) adds no point", () => {
    let s = pushSamples(EMPTY_SAMPLES, snap(1000, { spark1: 990, spark2: 700 }));
    s = pushSamples(s, snap(2000, { spark1: 1990, spark2: 700 })); // spark2 stream stalled
    s = pushSamples(s, snap(3000, { spark1: 2990, spark2: null })); // spark2 offline
    expect(s.nodes.spark2.map((p) => p.t)).toEqual([700]);
    expect(s.nodes.spark1).toHaveLength(3);
  });

  test("missing readings are null, never 0; the endpoint rate is not copied onto nodes", () => {
    const s = pushSamples(EMPTY_SAMPLES, {
      sampled_at_ms: 1000,
      metrics: { sampled_at: 999, decode_tok_per_s: null, throughput_tok_per_s: 0 },
      cluster: { nodes: [node("spark1", 990, { power_w: null, available_gib: null })] },
    });
    expect(s.nodes.spark1[0]).toEqual({ t: 990, power: null, util: 50, temp: 45, mem: null, rails: null });
    expect(s.endpoint[0]).toEqual({ t: 999, decode: null, throughput: 0 });
  });

  test("series are pruned by time, not by count", () => {
    let s = EMPTY_SAMPLES;
    for (let i = 0; i <= 80; i++) s = pushSamples(s, snap(i * 1000, { spark1: i * 1000 }));
    const ts = s.nodes.spark1.map((p) => p.t);
    expect(ts[0]).toBe(80_000 - SAMPLES_KEEP_MS);
    expect(ts[ts.length - 1]).toBe(80_000);
  });

  test("no timestamp on the snapshot → nothing to place on the axis", () => {
    expect(pushSamples(EMPTY_SAMPLES, { metrics: { sampled_at: 1 } })).toBe(EMPTY_SAMPLES);
    expect(pushSamples(EMPTY_SAMPLES, null)).toBe(EMPTY_SAMPLES);
  });
});

describe("stream merge and ordering", () => {
  const meta: LiveMeta = {
    controller: "ok",
    defaultBackend: "vllm",
    defaultModel: "auto",
    openAiBase: "http://127.0.0.1:8000/v1",
    backends: { vllm: { ok: true, url: "http://127.0.0.1:8000" } },
    models: [{ id: "org/m", max_model_len: 8192 }],
    version: { version: "0.30.0" },
    flags: ["--port", "8000"],
  };

  test("a tick plus the last meta is the same status /api/lab-status returns", () => {
    const st = mergeTick(meta, { serve: { healthy: true, sampled_at_ms: 5, engine: { kv_usage_pct: 1 } } });
    expect(st.openAiBase).toBe("http://127.0.0.1:8000/v1");
    expect(st.backends.vllm.ok).toBe(true);
    expect(st.serve?.models?.[0].id).toBe("org/m");
    expect(st.serve?.engine).toEqual({ kv_usage_pct: 1, flags: ["--port", "8000"] });
    // an engine that is down has no engine block: none is invented
    expect(mergeTick(meta, { serve: { error: "down", unreachable: true } }).serve?.engine).toBeUndefined();
  });

  test("an older snapshot never replaces a newer one (a slow poll landing after a stream tick)", () => {
    const st = (at: number): LabStatus => mergeTick(meta, { serve: snap(at, { spark1: at - 5 }) });
    expect(isOlder(st(2000), st(1000))).toBe(true);
    expect(isOlder(st(1000), st(2000))).toBe(false);
    ingestStatus(st(2000), "stream");
    ingestStatus(st(1000), "poll");
    expect(useLabStatusStore.getState().status?.serve?.sampled_at_ms).toBe(2000);
    expect(useLabStatusStore.getState().transport).toBe("stream");
  });
});

describe("serve-engine failures", () => {
  test("a failure keeps the last real snapshot (aging into stale) instead of blanking it", () => {
    const ok = mergeTick(null, { serve: snap(2000, { spark1: 1995 }) });
    ingestStatus(ok, "stream");
    const t0 = useLabStatusStore.getState().receivedAt;
    ingestStatus(mergeTick(null, { serve: { error: "The operation timed out.", unreachable: true } }), "poll");
    const s = useLabStatusStore.getState();
    expect(s.status?.serve?.sampled_at_ms).toBe(2000); // last real reading kept
    expect(s.receivedAt).toBe(t0); // so its age keeps growing
    expect(s.engineError).toBe("The operation timed out.");
    expect(s.unreachable).toBe(false); // the controller answered
    ingestStatus(mergeTick(null, { serve: snap(3000, { spark1: 2995 }) }), "stream");
    expect(useLabStatusStore.getState().engineError).toBeNull();
  });

  test("with nothing to keep, the failure itself is shown", () => {
    ingestStatus(mergeTick(null, { serve: { error: "status sampler warming up", sampled_at_ms: null } }), "stream");
    const s = useLabStatusStore.getState();
    expect(s.engineError).toBe("status sampler warming up");
    expect(s.status?.serve?.error).toBe("status sampler warming up");
  });
});

describe("freshness against the server clock", () => {
  test("ages use the server's own timestamps; the browser clock only measures time since arrival", () => {
    // the browser clock is far off the server's: ages must not care
    const status = { serve: { sampled_at_ms: 1_000_000, stale_s: 0.2 } } as LabStatus;
    const receivedAt = 5_000_000;
    const clock = { status, receivedAt };
    // the server's "now" counts the snapshot's age when served (stale_s) and the time since
    expect(serverNow(clock, receivedAt + 1500)).toBe(1_001_700);
    expect(snapshotAge(clock, receivedAt + 1500)).toBeCloseTo(1.7, 5);
    expect(isStale(clock, 3, receivedAt + 1000)).toBe(false);
    expect(isStale(clock, 3, receivedAt + 4000)).toBe(true);
    expect(isStale({ ...clock, unreachable: true }, 3, receivedAt)).toBe(true);
    expect(snapshotAge({ status: null, receivedAt: null })).toBeNull();
  });

  test("a healthy polling cycle is not stale: the poll interval is part of the threshold", () => {
    const status = { serve: { sampled_at_ms: 1_000_000, stale_s: 0.9 } } as LabStatus;
    const receivedAt = 5_000_000;
    // answer 0.9 s old, next poll lands 2 s + request time later: age ~3.2 s
    const at = receivedAt + 2300;
    expect(isStale({ status, receivedAt, transport: "stream" }, undefined, at)).toBe(true);
    expect(isStale({ status, receivedAt, transport: "poll" }, undefined, at)).toBe(false);
    expect(staleAfterS("poll")).toBe(STALE_AFTER_S + LAB_STATUS_POLL_MS / 1000);
    // polling that stopped answering still goes stale
    expect(isStale({ status, receivedAt, transport: "poll" }, undefined, receivedAt + 4500)).toBe(true);
  });
});

describe("nodes", () => {
  test("tightestNode: the live node with the least MemAvailable; down nodes skipped", () => {
    const nodes = [
      node("spark1", 1, { available_gib: 14.2 }),
      node("spark2", 1, { local: false, online: true, available_gib: 12.9 }),
      node("spark3", 1, { local: false, online: false, available_gib: 3 }),
    ];
    expect(tightestNode(nodes)?.id).toBe("spark2");
    expect(tightestNode([nodes[2]])).toBeNull();
    expect(tightestNode(undefined)).toBeNull();
  });
});

describe("setLiveRun change detection", () => {
  const run: LiveRun = { running: 4, waiting: 0, source: "streams" };

  test("sameLiveRun compares by value, null-safe", () => {
    expect(sameLiveRun(null, null)).toBe(true);
    expect(sameLiveRun(run, { ...run })).toBe(true);
    expect(sameLiveRun(run, { ...run, running: 3 })).toBe(false);
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
    expect(store.getState().liveRun).toBe(run);
    store.getState().setLiveRun({ ...run, running: 3 });
    expect(writes).toBe(2);
    store.getState().setLiveRun(null);
    store.getState().setLiveRun(null);
    expect(writes).toBe(3);
    unsub();
  });
});
