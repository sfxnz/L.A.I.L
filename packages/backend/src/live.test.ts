import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { createApp } from "./app";
import { config } from "./config";
import {
  LiveHub,
  compactStatus,
  liveResponse,
  probeBackends,
  resetBackendsCache,
  splitStatus,
  labStatusOf,
  type Backends,
} from "./live";

const origFetch = globalThis.fetch;
const prevToken = config.token;
beforeEach(() => resetBackendsCache());
afterEach(() => {
  globalThis.fetch = origFetch;
  config.token = prevToken;
  resetBackendsCache();
});

function snapshot(at: number, extra: Record<string, unknown> = {}) {
  return {
    healthy: true,
    model_id: "org/m",
    models: [{ id: "org/m" }],
    version: { version: "0.30.0" },
    metrics: { sampled_at: at - 5, decode_tok_per_s: 80, throughput_tok_per_s: 160, requests_running: 2 },
    engine: { kv_usage_pct: 1.5, flags: ["--port", "8000"], flags_fingerprint: "abc" },
    cluster: { nodes: [{ id: "spark1", local: true, sampled_at: at - 3, power_w: 30, containers: [{ name: "c" }] }] },
    sampled_at_ms: at,
    ...extra,
  };
}

const NO_BACKENDS = async (): Promise<Backends> => ({ vllm: { ok: true, url: "http://127.0.0.1:8000" } });

/** A fake serve-engine long poll: hands out queued snapshots, then hangs until aborted. */
function fakeEngine(queue: Array<Record<string, unknown>>) {
  const calls: Array<{ after: number | null | undefined; wait: number }> = [];
  const fetchStatus = (after?: number | null, wait = 0, signal?: AbortSignal) => {
    calls.push({ after, wait });
    const next = queue.shift();
    if (next) return Promise.resolve(next);
    return new Promise<Record<string, unknown>>((resolve) =>
      signal?.addEventListener("abort", () => resolve({ error: "aborted", unreachable: true })),
    );
  };
  return { calls, fetchStatus };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

describe("splitStatus", () => {
  test("static data rides meta; the tick carries the rest, flags dropped from the engine block", () => {
    const { meta, tick } = splitStatus(labStatusOf(snapshot(1000), {}));
    expect(meta.models).toEqual([{ id: "org/m" }]);
    expect(meta.version).toEqual({ version: "0.30.0" });
    expect(meta.flags).toEqual(["--port", "8000"]);
    expect(meta.backends).toEqual({});
    const serve = tick.serve as Record<string, unknown>;
    expect(serve.models).toBeUndefined();
    expect(serve.version).toBeUndefined();
    expect(serve.engine).toEqual({ kv_usage_pct: 1.5, flags_fingerprint: "abc" });
    expect(serve.sampled_at_ms).toBe(1000);
  });

  test("an unreachable engine (no engine block) splits without inventing one", () => {
    const { tick } = splitStatus(labStatusOf({ error: "down", unreachable: true }, {}));
    expect(tick.serve).toEqual({ error: "down", unreachable: true });
  });
});

describe("compactStatus", () => {
  test("keeps only what sparklines draw, in the status shape", () => {
    const c = compactStatus(snapshot(1000)) as { serve: { cluster: { nodes: Array<Record<string, unknown>> } } };
    expect(c.serve.cluster.nodes[0].power_w).toBe(30);
    expect(c.serve.cluster.nodes[0].containers).toBeUndefined();
    expect(JSON.stringify(c)).not.toContain("flags");
  });
});

describe("LiveHub", () => {
  test("one upstream long poll fans out; meta only when it changes; repeats are not re-sent", async () => {
    const eng = fakeEngine([snapshot(1000), snapshot(1000), snapshot(2000)]);
    const hub = new LiveHub(eng.fetchStatus, NO_BACKENDS, 0);
    const a: string[] = [];
    const b: string[] = [];
    const offA = hub.subscribe((e) => a.push(e));
    const offB = hub.subscribe((e) => b.push(e));
    // (an engine that answers a repeat at once is re-polled after 250 ms, not spun on)
    await new Promise((r) => setTimeout(r, 350));
    // the same snapshot (long poll timed out) is not pushed twice
    expect(a).toEqual(["meta", "tick", "tick"]);
    expect(b).toEqual(a);
    // first call is a plain read, then each poll asks for what comes after the last sample
    expect(eng.calls.map((c) => c.after)).toEqual([null, 1000, 1000, 2000]);
    expect(eng.calls.every((c) => c.wait > 0)).toBe(true);
    // a late subscriber gets meta, the history and the latest tick at once
    const c: Array<[string, string]> = [];
    const offC = hub.subscribe((e, d) => c.push([e, d]));
    expect(c.map(([e]) => e)).toEqual(["meta", "history", "tick"]);
    expect(JSON.parse(c[1][1]).map((h: { serve: { sampled_at_ms: number } }) => h.serve.sampled_at_ms)).toEqual([1000, 2000]);
    // after an idle spell the cached sample is not replayed as if it were current
    const realNow = Date.now;
    Date.now = () => realNow() + 120_000;
    const d: string[] = [];
    const offD = hub.subscribe((e) => d.push(e));
    Date.now = realNow;
    expect(d).toEqual(["meta"]);
    offA();
    offB();
    offC();
    offD();
    for (let i = 0; i < 5; i++) await tick();
    expect(hub.size).toBe(0);
  });

  test("an unreachable engine is reported as a tick, then retried with backoff", async () => {
    const eng = fakeEngine([{ error: "connect refused", unreachable: true }]);
    const hub = new LiveHub(eng.fetchStatus, NO_BACKENDS, 0);
    const got: Array<[string, string]> = [];
    const off = hub.subscribe((e, d) => got.push([e, d]));
    for (let i = 0; i < 10; i++) await tick();
    const ticks = got.filter(([e]) => e === "tick").map(([, d]) => JSON.parse(d));
    expect(ticks).toEqual([{ serve: { error: "connect refused", unreachable: true } }]);
    expect(eng.calls.length).toBe(1); // backing off, not spinning
    off();
  });
});

describe("lingering", () => {
  test("after the last tab leaves the loop keeps sampling, so the next one gets history at once", async () => {
    const eng = fakeEngine([snapshot(1000), snapshot(2000)]);
    const hub = new LiveHub(eng.fetchStatus, NO_BACKENDS, 60_000);
    const off = hub.subscribe(() => {});
    for (let i = 0; i < 5; i++) await tick();
    off();
    expect(hub.size).toBe(0);
    for (let i = 0; i < 5; i++) await tick();
    const got: string[] = [];
    const off2 = hub.subscribe((e) => got.push(e));
    expect(got).toEqual(["meta", "history", "tick"]);
    expect(eng.calls.length).toBe(3); // still polling with nobody subscribed
    off2();
  });
});

describe("liveResponse", () => {
  test("streams SSE frames and stops when the browser goes away", async () => {
    const eng = fakeEngine([snapshot(1000)]);
    const hub = new LiveHub(eng.fetchStatus, NO_BACKENDS, 0);
    const ac = new AbortController();
    const res = liveResponse(hub, ac.signal);
    expect(res.headers.get("content-type")).toBe("text/event-stream");
    expect(res.headers.get("content-encoding")).toBe("identity");
    const reader = res.body!.getReader();
    let text = "";
    while (!text.includes("event: tick")) {
      const { value } = await reader.read();
      text += new TextDecoder().decode(value);
    }
    expect(text).toContain("retry: 2000");
    expect(text).toMatch(/event: meta\ndata: \{.*"models"/);
    expect(hub.size).toBe(1);
    ac.abort();
    expect(hub.size).toBe(0);
    const { done } = await reader.read();
    expect(done).toBe(true);
  });
});

describe("routes", () => {
  test("/api/lab-status: one cluster copy, no share block, backends probed once per TTL", async () => {
    config.token = "";
    const urls: string[] = [];
    globalThis.fetch = (async (url: string) => {
      urls.push(String(url));
      if (String(url).includes("/api/status")) return Response.json(snapshot(1000));
      return Response.json({ data: [] });
    }) as unknown as typeof fetch;
    const app = createApp();
    const body = await (await app.request("/api/lab-status")).json();
    await app.request("/api/lab-status");
    expect(body.cluster).toBeUndefined();
    expect(body.share).toBeUndefined();
    expect(body.serve.cluster.nodes[0].id).toBe("spark1");
    expect(Object.keys(body.backends).length).toBeGreaterThan(0);
    const probes = urls.filter((u) => u.endsWith("/v1/models"));
    expect(probes.length).toBe(Object.keys(body.backends).length); // second call reused the probe
    expect(urls.filter((u) => u.includes("/api/status")).length).toBe(2);
  });

  test("/api/live sits behind the token gate", async () => {
    config.token = "secret";
    const app = createApp();
    const r = await app.request("/api/live");
    expect(r.status).toBe(401);
    expect((await r.json()).error).toBe("unauthorized");
  });

  test("backend probes are shared while one is in flight", async () => {
    let n = 0;
    globalThis.fetch = (async () => {
      n += 1;
      await new Promise((r) => setTimeout(r, 5));
      return Response.json({});
    }) as unknown as typeof fetch;
    const [a, b] = await Promise.all([probeBackends(), probeBackends()]);
    expect(a).toBe(b);
    const enabled = Object.keys(a).length;
    expect(n).toBe(enabled);
  });
});
