/**
 * The live transport: an SSE reader over fetch (token in a header, 401 by status),
 * the polling fallback when the stream keeps failing but the controller answers,
 * and "unreachable" only when nothing answers.
 */
import { afterEach, describe, expect, test } from "bun:test";
import { EMPTY_SAMPLES, LAB_STATUS_POLL_MS, useLabStatusStore } from "./lab-status-store";
import { RETRY_STREAM_MS, STALL_MS, readSse, startLive } from "./live-connection";

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
    engineError: null,
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

  test("behind a proxy that buffers the stream, polling never pauses while the stream is retried", async () => {
    const polls: number[] = [];
    const streams: number[] = [];
    globalThis.fetch = (async (url: string) => {
      if (String(url).startsWith("/api/lab-status")) {
        polls.push(Date.now());
        return Response.json(JSON.parse(tick(1000 + polls.length)));
      }
      throw new Error("unexpected " + url);
    }) as unknown as typeof fetch;
    // every timer runs 100× faster: stall 80 ms, poll 20 ms, stream retry 300 ms
    const SCALE = 100;
    const realSetTimeout = globalThis.setTimeout;
    globalThis.setTimeout = ((fn: () => void, ms?: number) => realSetTimeout(fn, (ms ?? 0) / SCALE)) as typeof setTimeout;
    let stop = () => {};
    try {
      // headers arrive, then not one byte: the stall timer is what ends each attempt
      stop = startLive(async (_url, init) => {
        streams.push(Date.now());
        const signal = init?.signal;
        return new Response(
          new ReadableStream({
            start(ctrl) {
              signal?.addEventListener("abort", () => ctrl.error(new DOMException("aborted", "AbortError")));
            },
          }),
          { headers: { "content-type": "text/event-stream" } },
        );
      });
      // two failed attempts → poll mode → retries every RETRY_STREAM_MS; watch two retries
      await until(() => streams.length >= 4, 5000);
      await new Promise((r) => realSetTimeout(r, (STALL_MS + LAB_STATUS_POLL_MS) / SCALE));
    } finally {
      stop();
      globalThis.setTimeout = realSetTimeout;
    }
    expect(useLabStatusStore.getState().transport).toBe("poll");
    // a failed retry goes back to waiting RETRY_STREAM_MS, not through the reconnect backoff
    expect(streams[3] - streams[2]).toBeGreaterThanOrEqual(RETRY_STREAM_MS / SCALE - 5);
    // from the first retry on, polls kept landing every LAB_STATUS_POLL_MS (no stall-long holes)
    const during = polls.filter((t) => t >= streams[2]);
    expect(during.length).toBeGreaterThan(10);
    const gaps = during.slice(1).map((t, i) => t - during[i]);
    expect(Math.max(...gaps)).toBeLessThan(STALL_MS / SCALE);
  });

  test("the web proxy's bare 500 (controller down) reads as no answer, not as a server bug", async () => {
    globalThis.fetch = (async () => new Response("Internal Server Error", { status: 500 })) as unknown as typeof fetch;
    const stop = startLive(async () => new Response("Internal Server Error", { status: 500 }));
    await until(() => useLabStatusStore.getState().unreachable);
    stop();
    expect(useLabStatusStore.getState().error).toBe("no answer from the controller (HTTP 500)");
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
