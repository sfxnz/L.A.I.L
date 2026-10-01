/**
 * The live transport: an SSE reader over fetch (token in a header, 401 by status),
 * the polling fallback when the stream keeps failing but the controller answers,
 * and "unreachable" only when nothing answers.
 */
import { afterEach, describe, expect, test } from "bun:test";
import { EMPTY_SAMPLES, useLabStatusStore } from "./lab-status-store";
import { readSse, startLive } from "./live-connection";

const origFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = origFetch;
  useLabStatusStore.setState({
    status: null,
    meta: null,
    loading: true,
    needToken: false,
    unreachable: false,
    error: null,
    receivedAt: null,
    transport: null,
    samples: EMPTY_SAMPLES,
  });
});

const enc = new TextEncoder();
function sseBody(chunks: string[], keepOpen = false): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(ctrl) {
      for (const c of chunks) ctrl.enqueue(enc.encode(c));
      if (!keepOpen) ctrl.close();
    },
  });
}
const until = async (cond: () => boolean, ms = 2000) => {
  const t0 = Date.now();
  while (!cond()) {
    if (Date.now() - t0 > ms) throw new Error("timed out");
    await new Promise((r) => setTimeout(r, 5));
  }
};

const META = JSON.stringify({ controller: "ok", defaultBackend: "vllm", defaultModel: "auto", openAiBase: "http://x/v1", backends: {}, models: [{ id: "org/m" }], version: null, flags: ["--a"] });
const tick = (at: number) => JSON.stringify({ serve: { healthy: true, sampled_at_ms: at, engine: { kv_usage_pct: 2 }, cluster: { nodes: [{ id: "spark1", sampled_at: at - 3, power_w: 9 }] }, metrics: { sampled_at: at - 5 } } });

describe("readSse", () => {
  test("events split across chunks, comments ignored, multi-line data joined", async () => {
    const got: Array<[string, string]> = [];
    let chunks = 0;
    await readSse(
      sseBody(["retry: 2000\n\n: ping\n\nevent: me", "ta\ndata: {\"a\":1}\n", "\nevent: tick\ndata: x\ndata: y\r\n\r\n"]),
      (e, d) => got.push([e, d]),
      () => chunks++,
    );
    expect(got).toEqual([
      ["meta", '{"a":1}'],
      ["tick", "x\ny"],
    ]);
    expect(chunks).toBe(3);
  });
});

describe("startLive", () => {
  test("streams meta + history + ticks into the store, with the token in a header", async () => {
    const calls: Array<{ url: string; init?: RequestInit }> = [];
    const stop = startLive(async (url, init) => {
      calls.push({ url, init });
      return new Response(
        sseBody(
          [`event: meta\ndata: ${META}\n\n`, `event: history\ndata: [${tick(1000)}]\n\n`, `event: tick\ndata: ${tick(2000)}\n\n`],
          true,
        ),
        { headers: { "content-type": "text/event-stream" } },
      );
    });
    await until(() => useLabStatusStore.getState().status?.serve?.sampled_at_ms === 2000);
    stop();
    const s = useLabStatusStore.getState();
    expect(calls[0].url).toBe("/api/live");
    expect(s.transport).toBe("stream");
    expect(s.loading).toBe(false);
    expect(s.status?.serve?.models?.[0].id).toBe("org/m");
    expect(s.status?.serve?.engine?.flags).toEqual(["--a"]);
    // history and the live tick both landed on the time axis
    expect(s.samples.nodes.spark1.map((p) => p.t)).toEqual([997, 1997]);
  });

  test("401 from the stream is the token banner, decided by status — and nothing retries", async () => {
    let n = 0;
    globalThis.fetch = (async () => Response.json({ error: "unauthorized" }, { status: 401 })) as unknown as typeof fetch;
    const stop = startLive(async () => {
      n += 1;
      return Response.json({ error: "unauthorized", message: "LAIL_TOKEN required" }, { status: 401 });
    });
    await until(() => useLabStatusStore.getState().needToken);
    await new Promise((r) => setTimeout(r, 50));
    stop();
    expect(n).toBe(1);
    expect(useLabStatusStore.getState().unreachable).toBe(false);
  });

  test("controller answers but the stream keeps failing → polling takes over", async () => {
    let polls = 0;
    globalThis.fetch = (async (url: string) => {
      if (String(url).startsWith("/api/lab-status")) {
        polls += 1;
        return Response.json(JSON.parse(tick(1000 + polls)));
      }
      throw new Error("unexpected " + url);
    }) as unknown as typeof fetch;
    const realSetTimeout = globalThis.setTimeout;
    // collapse the reconnect backoff so the test does not wait seconds
    globalThis.setTimeout = ((fn: () => void, ms?: number) => realSetTimeout(fn, Math.min(ms ?? 0, 5))) as typeof setTimeout;
    try {
      const stop = startLive(async () => new Response("bad gateway", { status: 502 }));
      await until(() => useLabStatusStore.getState().transport === "poll" && polls >= 3);
      stop();
    } finally {
      globalThis.setTimeout = realSetTimeout;
    }
    expect(useLabStatusStore.getState().unreachable).toBe(false);
    expect(useLabStatusStore.getState().status?.serve?.sampled_at_ms).toBeGreaterThan(1000);
  });

  test("nothing answers → unreachable, with the error", async () => {
    globalThis.fetch = (async () => {
      throw new TypeError("fetch failed");
    }) as unknown as typeof fetch;
    const stop = startLive(async () => {
      throw new TypeError("fetch failed");
    });
    await until(() => useLabStatusStore.getState().unreachable);
    stop();
    expect(useLabStatusStore.getState().needToken).toBe(false);
    expect(useLabStatusStore.getState().error).toContain("fetch failed");
  });
});
