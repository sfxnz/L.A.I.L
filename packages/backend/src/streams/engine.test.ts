import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { Hono } from "hono";
import type { StreamRunEvent, StreamRunSnapshot } from "@lail/shared";
import { config } from "../config";
import { StreamsEngine, arrivalDelays } from "./engine";
import { createStreamsRoutes } from "./routes";
import { assignPrompts, getPack, listPacks, strandSystemPrompt } from "./packs";
import { aggregateSteadyTokPerS, aggregateTokPerS, summarizeWave, type StrandResult } from "./metrics";
import { SseParser } from "./sse-parser";

// ── Mock OpenAI server: reasoning + content deltas, finish_reason, trailing usage frame ──

type MockState = {
  requests: Array<Record<string, unknown>>;
  aborted: number;
  importMode: "ok" | "404";
  imported: unknown[];
  /** Honour `stream_options.continuous_usage_stats` (vLLM): `usage` on every chunk. */
  continuousUsage: boolean;
};
const state: MockState = { requests: [], aborted: 0, importMode: "ok", imported: [], continuousUsage: false };

function frame(delta: Record<string, unknown>, finish: string | null = null, usage?: Record<string, number>): string {
  const chunk = { id: "cmpl-1", object: "chat.completion.chunk", choices: [{ index: 0, delta, finish_reason: finish }], ...(usage ? { usage } : {}) };
  return `data: ${JSON.stringify(chunk)}\n\n`;
}

function completions(body: Record<string, unknown>, signal: AbortSignal): Response {
  const maxTokens = Number(body.max_tokens) || 1;
  const n = Math.min(maxTokens, 120);
  const slow = maxTokens >= 100;
  // max_tokens divisible by 5 emulates speculative/MTP decoding: every chunk carries 3 tokens.
  const tokensPerChunk = maxTokens % 5 === 0 ? 3 : 1;
  const messages = body.messages as Array<{ role: string; content: string }>;
  const promptTokens = Math.ceil(messages[messages.length - 1].content.length / 4);
  const opts = body.stream_options as { continuous_usage_stats?: boolean } | undefined;
  const continuous = state.continuousUsage && opts?.continuous_usage_stats === true;
  let outChunks = 0;
  const usage = () => (continuous ? { prompt_tokens: promptTokens, completion_tokens: outChunks * tokensPerChunk } : undefined);
  const out = (delta: Record<string, unknown>) => {
    outChunks++;
    return frame(delta, null, usage());
  };
  const frames: string[] = [": keep-alive\n\n", frame({ role: "assistant", content: "" }, null, usage())];
  if (n > 1) frames.push(out({ reasoning_content: "thinking " }), out({ reasoning_content: "hard. " }));
  for (let k = 0; k < n; k++) frames.push(out({ content: `tok${k} ` }));
  frames.push(frame({}, "length", usage()));
  const completionTokens = (n + (n > 1 ? 2 : 0)) * tokensPerChunk;
  frames.push(`data: ${JSON.stringify({ id: "cmpl-1", choices: [], usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens } })}\r\n\r\n`);
  frames.push("data: [DONE]\n\n");
  // Split the first content frame across two writes to exercise the parser over HTTP.
  const split = frames.findIndex((f) => f.includes("tok0"));
  const [a, b] = [frames[split].slice(0, 12), frames[split].slice(12)];
  frames.splice(split, 1, a, b);

  const enc = new TextEncoder();
  signal.addEventListener("abort", () => state.aborted++, { once: true });
  return new Response(
    new ReadableStream({
      async start(controller) {
        for (const f of frames) {
          if (signal.aborted) break;
          await Bun.sleep(slow ? 15 : 3);
          try {
            controller.enqueue(enc.encode(f));
          } catch {
            break;
          }
        }
        try {
          controller.close();
        } catch {
          /* closed */
        }
      },
    }),
    { headers: { "content-type": "text/event-stream" } },
  );
}

const server = Bun.serve({
  port: 0,
  hostname: "127.0.0.1",
  async fetch(req) {
    const url = new URL(req.url);
    if (url.pathname === "/v1/models") {
      return Response.json({ data: [{ id: "mock/served", max_model_len: 4096 }] });
    }
    if (url.pathname === "/tokenize") {
      const body = (await req.json()) as { messages: Array<{ content: string }> };
      return Response.json({ count: Math.ceil(body.messages[0].content.length / 4), max_model_len: 4096 });
    }
    if (url.pathname === "/v1/chat/completions") {
      const body = (await req.json()) as Record<string, unknown>;
      state.requests.push(body);
      return completions(body, req.signal);
    }
    if (url.pathname === "/api/runs/import") {
      if (state.importMode === "404") return Response.json({ detail: "Not Found" }, { status: 404 });
      state.imported.push(await req.json());
      return Response.json({ run_id: "imported-123" });
    }
    return new Response("nope", { status: 404 });
  },
});
const BASE = `http://127.0.0.1:${server.port}`;

const engine = new StreamsEngine({ idleAbortMs: 300 });
const app = new Hono();
app.route("/api/streams", createStreamsRoutes(engine));

async function post(path: string, body: unknown) {
  return app.request(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
}

async function collect(runId: string, until: (ev: StreamRunEvent) => boolean = (ev) => ev.type === "done"): Promise<StreamRunEvent[]> {
  const res = await app.request(`/api/streams/runs/${runId}/events`);
  expect(res.status).toBe(200);
  expect(res.headers.get("content-type")).toContain("text/event-stream");
  expect(res.headers.get("content-encoding")).toBe("identity");
  expect(res.headers.get("x-accel-buffering")).toBe("no");
  const reader = res.body!.getReader();
  const dec = new TextDecoder();
  const parser = new SseParser();
  const events: StreamRunEvent[] = [];
  for (;;) {
    const { done, value } = await reader.read();
    const payloads = done ? parser.end() : parser.feed(dec.decode(value, { stream: true }));
    for (const p of payloads) {
      const ev = JSON.parse(p) as StreamRunEvent;
      events.push(ev);
      if (until(ev)) {
        await reader.cancel();
        return events;
      }
    }
    if (done) return events;
  }
}

const byType = <T extends StreamRunEvent["type"]>(events: StreamRunEvent[], type: T) =>
  events.filter((e) => e.type === type) as Array<Extract<StreamRunEvent, { type: T }>>;

let prevServeEngine: string;
beforeAll(() => {
  prevServeEngine = config.serveEngineUrl;
  config.serveEngineUrl = BASE;
});
afterAll(() => {
  config.serveEngineUrl = prevServeEngine;
  server.stop(true);
});

describe("packs", () => {
  test("six packs; the four perf.py families are verbatim; mixed rotates", async () => {
    const res = await app.request("/api/streams/packs");
    const packs = (await res.json()) as ReturnType<typeof listPacks>;
    expect(packs.map((p) => p.id)).toEqual(["prose", "structured", "code", "json", "chat-short", "mixed"]);
    expect(getPack("prose")!.prompts[0].text.startsWith("Continue this essay in the same voice. Do not stop.")).toBe(true);
    expect(getPack("code")!.prompts[0].text).toContain("class TokenBucket:");
    expect(getPack("chat-short")!.prompts.length).toBeGreaterThanOrEqual(6);
    expect(getPack("mixed")!.prompts.map((p) => p.pack).slice(0, 5)).toEqual(["prose", "structured", "code", "json", "chat-short"]);
  });

  test("round-robin assignment: user text is exactly the pack text; uniqueness lives in the system prompt", () => {
    const refs = assignPrompts(getPack("prose")!, 4);
    expect(refs.every((r) => r.text === getPack("prose")!.prompts[0].text)).toBe(true);
    expect(refs.map((r) => r.i)).toEqual([0, 1, 2, 3]);
    const mixed = assignPrompts(getPack("mixed")!, 10, 5, 2);
    expect(mixed[0]).toMatchObject({ i: 5, pack: "prose", level: 2 });
    expect(mixed[8].pack).toBe("prose");
    expect(strandSystemPrompt("run1", 0)).not.toBe(strandSystemPrompt("run1", 1));
    expect(strandSystemPrompt("run1", 0)).not.toBe(strandSystemPrompt("run2", 0));
  });

  test("wave aggregates: wall-clock spans the straggler's tail, steady sums per-strand decode rates", () => {
    const strand = (i: number, t_start: number, t_first: number, t_last: number, tokens: number): StrandResult => ({
      i,
      ok: true,
      t_start,
      t_first,
      t_last,
      t_end: t_last,
      completion_tokens: tokens,
      prompt_tokens: null,
      estimated: false,
      finish_reason: "length",
      error: null,
      token_times_ms: [],
    });
    // Two strands at 10 tok/s each; strand 1 starts late and drags the wall-clock window to 3 s.
    const results = [strand(0, 0, 500, 1500, 10), strand(1, 0, 2000, 3000, 10)];
    expect(aggregateTokPerS(results)).toBeCloseTo(20 / 3);
    expect(aggregateSteadyTokPerS(results)).toBeCloseTo(20);
    const ws = summarizeWave(results);
    expect(ws.aggregate_tok_s).toBe(6.67);
    expect(ws.aggregate_steady_tok_s).toBe(20);
    expect(aggregateSteadyTokPerS([{ ...results[0], ok: false }])).toBeNull();
  });

  test("arrival schedules", () => {
    expect(arrivalDelays(3, "burst", 250)).toEqual([0, 0, 0]);
    expect(arrivalDelays(3, "staggered", 250)).toEqual([0, 250, 500]);
    const poisson = arrivalDelays(4, "poisson", 100);
    expect(poisson[0]).toBe(0);
    for (let k = 1; k < poisson.length; k++) expect(poisson[k]).toBeGreaterThanOrEqual(poisson[k - 1]);
  });
});

describe("request validation", () => {
  test("rejects foreign base_url, bad mode, unknown pack, malformed body", async () => {
    expect((await post("/api/streams/runs", { mode: "load", pack: "prose", base_url: "http://evil.example:8000" })).status).toBe(400);
    expect((await post("/api/streams/runs", { mode: "load", pack: "prose", base_url: "https://127.0.0.1:8000" })).status).toBe(400);
    expect((await post("/api/streams/runs", { mode: "load", pack: "prose", base_url: "not a url" })).status).toBe(400);
    expect((await post("/api/streams/runs", { mode: "warp", pack: "prose", base_url: BASE })).status).toBe(400);
    expect((await post("/api/streams/runs", { mode: "load", pack: "nope", base_url: BASE })).status).toBe(400);
    expect((await post("/api/streams/runs", { mode: "load", pack: "prose", base_url: BASE, n: 33 })).status).toBe(400);
    const res = await app.request("/api/streams/runs", { method: "POST", body: "{" });
    expect(res.status).toBe(400);
  });

  test("502 when /v1/models is unreachable", async () => {
    const res = await post("/api/streams/runs", { mode: "load", pack: "prose", base_url: "http://127.0.0.1:1" });
    expect(res.status).toBe(502);
    expect(((await res.json()) as { error: string }).error).toBe("models_unreachable");
  });

  test("unknown run → 404 on snapshot, events and stop", async () => {
    expect((await app.request("/api/streams/runs/nope")).status).toBe(404);
    expect((await app.request("/api/streams/runs/nope/events")).status).toBe(404);
    expect((await app.request("/api/streams/runs/nope/stop", { method: "POST" })).status).toBe(404);
  });
});

describe("load run", () => {
  test("2 strands: hello, coalesced deltas (reasoning tagged), strand lifecycle, agg, done", async () => {
    state.requests = [];
    const res = await post("/api/streams/runs", { mode: "load", pack: "mixed", base_url: BASE, n: 2, max_tokens: 6, thinking: "off" });
    expect(res.status).toBe(201);
    const { run_id } = (await res.json()) as { run_id: string };
    const events = await collect(run_id);

    const hello = byType(events, "hello")[0];
    expect(events[0]).toBe(hello);
    expect(hello).toMatchObject({ run_id, mode: "load", model: "mock/served", base_url: BASE, n: 2, max_tokens: 6, max_model_len: 4096 });
    expect(hello.prompts.map((p) => p.pack)).toEqual(["prose", "structured"]);
    expect(hello.prompts[0].text).toBe(getPack("prose")!.prompts[0].text);

    // Upstream request shape: unique system message per strand, user turn = pack text verbatim
    expect(state.requests).toHaveLength(2);
    expect(state.requests[0]).toMatchObject({
      model: "mock/served",
      max_tokens: 6,
      temperature: 0.2,
      stream: true,
      stream_options: { include_usage: true, continuous_usage_stats: true },
      chat_template_kwargs: { enable_thinking: false },
    });
    expect(state.requests[0]).not.toHaveProperty("min_tokens");
    // Burst launches both strands at once, so upstream arrival order is not strand order:
    // key each request by its strand's system prompt instead of its position in state.requests.
    const msgs = [0, 1].map((i) => {
      const r = state.requests.find((q) => (q.messages as Array<{ content: string }>)[0].content === strandSystemPrompt(run_id, i));
      expect(r).toBeDefined();
      return r!.messages as Array<{ role: string; content: string }>;
    });
    expect(msgs[0].map((m) => m.role)).toEqual(["system", "user"]);
    expect(msgs[0][1].content).toBe(getPack("prose")!.prompts[0].text);
    expect(msgs[0][0].content).toContain(run_id);
    expect(msgs[0][0].content).not.toBe(msgs[1][0].content);

    for (const i of [0, 1]) {
      const deltas = byType(events, "delta").filter((d) => d.i === i);
      const reasoning = deltas.filter((d) => d.reasoning).map((d) => d.text).join("");
      const text = deltas.filter((d) => !d.reasoning).map((d) => d.text).join("");
      expect(reasoning).toBe("thinking hard. ");
      expect(text).toBe("tok0 tok1 tok2 tok3 tok4 tok5 ");
      expect(deltas.reduce((a, d) => a + d.chunks, 0)).toBe(8);

      const strands = byType(events, "strand").filter((s) => s.i === i);
      const states = strands.map((s) => s.state);
      expect(states[0]).toBe("prefill");
      expect(states).toContain("decode");
      const last = strands[strands.length - 1];
      expect(last.state).toBe("done");
      expect(last.tokens).toBe(8); // from the usage frame, not the chunk count
      expect(last.finish_reason).toBe("length");
      expect(last.ttft_ms).toBeGreaterThan(0);
      expect(last.tok_s).toBeGreaterThan(0);
      expect(last.itl_ms).toHaveLength(7);
      // live decode emits carry the gaps so far (the card draws ITL while decoding)
      expect(strands.some((s) => s.state === "decode" && Array.isArray(s.itl_ms))).toBe(true);
    }

    const aggs = byType(events, "agg");
    expect(aggs.length).toBeGreaterThan(0);
    const lastAgg = aggs[aggs.length - 1];
    expect(lastAgg).toMatchObject({ running: 0, waiting: 0, done: 2, tokens: 16, tokens_per_chunk: 1, calibrated: true });
    expect(lastAgg.peak_tok_s).toBeGreaterThan(0);
    expect(byType(events, "level")).toHaveLength(0);

    const done = byType(events, "done")[0];
    expect(events[events.length - 1]).toBe(done);
    expect(done.saved_run_id).toBeNull();
    expect(done.summary).toMatchObject({ status: "done", mode: "load", ok: 2, requests: 2, tokens: 16, errors: [] });
    expect(done.summary.aggregate_tok_s).toBeGreaterThan(0);
    expect(done.summary.headline).toBeUndefined();

    // Snapshot after completion + recent list
    const snap = (await (await app.request(`/api/streams/runs/${run_id}`)).json()) as StreamRunSnapshot;
    expect(snap.status).toBe("done");
    expect(snap.hello).toMatchObject({ run_id, mode: "load", n: 2 });
    expect(snap.strands[1]).toMatchObject({ i: 1, state: "done", text: "tok0 tok1 tok2 tok3 tok4 tok5 ", reasoning_text: "thinking hard. ", chunks: 8 });
    expect(snap.agg.length).toBe(aggs.length);
    expect(snap.levels).toEqual([]);
    expect(snap.done?.summary.status).toBe("done");
    expect(snap.error).toBeNull();
    const list = (await (await app.request("/api/streams/runs")).json()) as Array<{ run_id: string; status: string }>;
    expect(list.find((r) => r.run_id === run_id)?.status).toBe("done");

    // A late subscriber gets hello + snapshot (incl. the whole retained agg history) + done and closes
    const late = await collect(run_id);
    expect(late.map((e) => e.type).slice(0, 7)).toEqual(["hello", "strand", "delta", "delta", "strand", "delta", "delta"]);
    expect(byType(late, "agg")).toEqual(aggs);
    expect(late[late.length - 1].type).toBe("done");
  });

  test("fill_to_max sends min_tokens + ignore_eos; thinking auto omits chat_template_kwargs", async () => {
    state.requests = [];
    const res = await post("/api/streams/runs", { mode: "load", pack: "prose", base_url: BASE, n: 1, max_tokens: 3, fill_to_max: true, thinking: "auto" });
    const { run_id } = (await res.json()) as { run_id: string };
    await collect(run_id);
    expect(state.requests[0]).toMatchObject({ min_tokens: 3, ignore_eos: true });
    expect(state.requests[0]).not.toHaveProperty("chat_template_kwargs");
  });

  test("stop aborts every stream: strands cancelled, upstream sees the abort, run is cancelled", async () => {
    state.aborted = 0;
    const res = await post("/api/streams/runs", { mode: "load", pack: "prose", base_url: BASE, n: 2, max_tokens: 120 });
    const { run_id } = (await res.json()) as { run_id: string };
    // Second run on the same base while this one is active → 409 with the active id
    const dup = await post("/api/streams/runs", { mode: "bench-decode", pack: "prose", base_url: BASE, levels: [1] });
    expect(dup.status).toBe(409);
    expect((await dup.json()) as object).toMatchObject({ error: "run_active", run_id });

    const events: StreamRunEvent[] = [];
    const seen = collect(run_id, (ev) => {
      events.push(ev);
      return ev.type === "done";
    });
    // wait for the first delta, then stop
    while (!events.some((e) => e.type === "delta")) await Bun.sleep(10);
    const stop = await app.request(`/api/streams/runs/${run_id}/stop`, { method: "POST" });
    expect(stop.status).toBe(200);
    const all = await seen;
    const done = byType(all, "done")[0];
    expect(done.summary.status).toBe("cancelled");
    expect(done.summary.tokens).toBeGreaterThan(0); // partial output of cancelled strands is counted
    // Partial summary from what was observed: both strands produced output and reached first token.
    expect(done.summary.ok).toBe(2);
    expect(done.summary.requests).toBe(2);
    expect(done.summary.errors).toEqual([]);
    expect(done.summary.aggregate_tok_s).toBeGreaterThan(0);
    expect(done.summary.per_stream_median_tok_s).toBeGreaterThan(0);
    expect(done.summary.ttft_p50_ms).toBeGreaterThan(0);
    const finals = [0, 1].map((i) => byType(all, "strand").filter((s) => s.i === i).pop()!);
    expect(finals.map((s) => s.state)).toEqual(["cancelled", "cancelled"]);
    await Bun.sleep(50);
    expect(state.aborted).toBe(2);
    // exclusivity released
    const again = await post("/api/streams/runs", { mode: "load", pack: "prose", base_url: BASE, n: 1, max_tokens: 1 });
    expect(again.status).toBe(201);
    await collect(((await again.json()) as { run_id: string }).run_id);
  });

  test("subscribing mid-stream replays retained text once: no duplicate at the join point", async () => {
    const run_id = await engine.createRun({ mode: "load", pack: "prose", base_url: BASE, n: 1, max_tokens: 120 });
    const keep = engine.subscribe(run_id)!; // keeps the run alive; we never read from it
    // Wait until text has been retained AND some of it is still pending (frames every 15 ms, flush every 80 ms).
    while (!engine.snapshot(run_id)!.strands[0]?.text) await Bun.sleep(5);
    await Bun.sleep(30);
    const events = await collect(run_id);
    engine.unsubscribe(run_id, keep);
    const text = byType(events, "delta")
      .filter((d) => d.i === 0 && !d.reasoning)
      .map((d) => d.text)
      .join("");
    expect(text).toBe(engine.snapshot(run_id)!.strands[0].text);
    const toks = text.trim().split(/\s+/);
    expect(new Set(toks).size).toBe(toks.length); // tok0 … tok119, each exactly once
  });

  test("load run is aborted when its last subscriber leaves and nobody reattaches", async () => {
    const run_id = await engine.createRun({ mode: "load", pack: "prose", base_url: BASE, n: 1, max_tokens: 120 });
    const sub = engine.subscribe(run_id)!;
    engine.unsubscribe(run_id, sub);
    await Bun.sleep(150);
    expect(engine.snapshot(run_id)!.status).toBe("running");
    const back = engine.subscribe(run_id)!; // reattach within the grace period cancels the abort
    await Bun.sleep(250);
    expect(engine.snapshot(run_id)!.status).toBe("running");
    engine.unsubscribe(run_id, back);
    await Bun.sleep(450);
    expect(engine.snapshot(run_id)!.status).toBe("cancelled");
  });
});

describe("bench runs", () => {
  test("bench-decode: sequential waves, level events, envelope imported into the serve-engine", async () => {
    state.requests = [];
    state.imported = [];
    state.importMode = "ok";
    const res = await post("/api/streams/runs", { mode: "bench-decode", pack: "prose", base_url: BASE, levels: [2, 1], max_tokens: 4 });
    expect(res.status).toBe(201);
    const { run_id } = (await res.json()) as { run_id: string };
    const events = await collect(run_id);

    const hello = byType(events, "hello")[0];
    expect(hello).toMatchObject({ mode: "bench-decode", levels: [1, 2], n: 3 });
    expect(hello.prompts.map((p) => p.level)).toEqual([0, 1, 1]);
    expect(state.requests.every((r) => r.min_tokens === 4 && r.ignore_eos === true)).toBe(true);

    const levels = byType(events, "level");
    expect(levels).toHaveLength(2);
    expect(levels[0]).toMatchObject({ index: 0, concurrency: 1, ok: 1, requests: 1, errors: [] });
    expect(levels[1]).toMatchObject({ index: 1, concurrency: 2, ok: 2, requests: 2 });
    expect(levels[1].aggregate_tok_s).toBeGreaterThan(0);
    expect(levels[1].aggregate_steady_tok_s).toBeGreaterThanOrEqual(levels[1].aggregate_tok_s!);
    expect(levels[1].per_stream_median_tok_s).toBeGreaterThan(0);
    expect(levels[1].ttft_p50_ms).toBeGreaterThan(0);
    // wave 2 starts only after wave 1 finished
    const strand0Done = events.findIndex((e) => e.type === "strand" && e.i === 0 && e.state === "done");
    const strand1Start = events.findIndex((e) => e.type === "strand" && e.i === 1);
    expect(strand1Start).toBeGreaterThan(strand0Done);

    const done = byType(events, "done")[0];
    expect(done.saved_run_id).toBe("imported-123");
    expect(done.summary.headline).toMatchObject({ aggregate_peak_concurrency: 2 });
    expect(done.summary.headline!.decode_tok_per_s_median_c1).toBeGreaterThan(0);

    expect(state.imported).toHaveLength(1);
    const env = state.imported[0] as Record<string, any>;
    expect(env.kind).toBe("decode");
    expect(env.source).toBe("controller-streams");
    expect(env.model).toBe("mock/served");
    expect(env.workload).toEqual({ pack: "prose", levels: [1, 2], max_tokens: 4, thinking: "off", temperature: 0.2, fill_to_max: true, base_url: BASE });
    expect(env.metrics.arms.map((a: { concurrency: number }) => a.concurrency)).toEqual([1, 2]);
    expect(env.metrics.arms[1].aggregate_steady_tok_per_s).toBe(levels[1].aggregate_steady_tok_s);
    expect(env.metrics.full_arms[1].per_request).toHaveLength(2);
    expect(env.metrics.full_arms[1].per_request[0].token_times_ms).toHaveLength(6);
    expect(env.metrics.full_arms[1].per_request[0].completion_tokens).toBe(6);
    expect(env.summary).toEqual(env.metrics.headline);
  });

  test("live token estimate is calibrated from usage (multi-token chunks) and remembered per base_url+model", async () => {
    state.importMode = "404";
    // max_tokens 5 → the mock reports 3 tokens per chunk; wave 1 (c=1) calibrates wave 2 (c=2).
    const res = await post("/api/streams/runs", { mode: "bench-decode", pack: "prose", base_url: BASE, levels: [1, 2], max_tokens: 5 });
    const { run_id } = (await res.json()) as { run_id: string };
    const events = await collect(run_id);
    const aggs = byType(events, "agg");
    expect(aggs[aggs.length - 1].tokens_per_chunk).toBe(3);
    // The first decode event of a strand carries exactly one chunk: ×1 before calibration (strand 0), ×3 after (strand 2).
    const firstDecode = (i: number) => byType(events, "strand").find((s) => s.i === i && s.state === "decode")!;
    expect(firstDecode(0).tokens).toBe(1);
    expect(firstDecode(2).tokens).toBe(3);
    // Terminal counts always come from usage.
    expect(byType(events, "strand").filter((s) => s.i === 2).pop()!.tokens).toBe(21);
    expect(byType(events, "done")[0].summary.tokens).toBe(63);

    // A new run against the same base_url+model starts calibrated.
    const again = await post("/api/streams/runs", { mode: "load", pack: "prose", base_url: BASE, n: 1, max_tokens: 5 });
    const second = await collect(((await again.json()) as { run_id: string }).run_id);
    expect(byType(second, "strand").find((s) => s.state === "decode")!.tokens).toBe(3);
    state.importMode = "ok";
  });

  test("without continuous usage, `calibrated` stays false until a strand's trailing usage frame", async () => {
    // Seeded from memory (previous test) is not "calibrated": the run has not measured anything yet.
    const res = await post("/api/streams/runs", { mode: "load", pack: "prose", base_url: BASE, n: 1, max_tokens: 120 });
    const { run_id } = (await res.json()) as { run_id: string };
    const events = await collect(run_id);
    const aggs = byType(events, "agg");
    expect(aggs.length).toBeGreaterThan(2);
    expect(aggs[0].calibrated).toBe(false);
    // The trailing usage frame follows every output chunk but precedes [DONE] by one frame, so an
    // agg tick may land between it and the strand's `done`. What must hold: the first calibrated
    // agg already counts every output chunk, i.e. calibration came from the trailing frame.
    const doneTokens = byType(events, "strand").find((s) => s.state === "done")!.tokens!;
    expect(aggs.find((a) => a.calibrated)!.tokens).toBe(doneTokens);
    expect(aggs[aggs.length - 1]).toMatchObject({ calibrated: true, tokens_per_chunk: 3 });
  });

  test("continuous usage stats calibrate from the first output chunk, before any strand finishes", async () => {
    state.continuousUsage = true;
    state.importMode = "404";
    try {
      const res = await post("/api/streams/runs", { mode: "bench-decode", pack: "prose", base_url: BASE, levels: [2], max_tokens: 120 });
      const { run_id } = (await res.json()) as { run_id: string };
      const events = await collect(run_id);
      const firstDecode = byType(events, "strand").find((s) => s.state === "decode")!;
      expect(firstDecode.tokens).toBe(3);
      const firstDone = events.findIndex((e) => e.type === "strand" && e.state === "done");
      const firstCalibratedAgg = events.findIndex((e) => e.type === "agg" && e.calibrated);
      expect(firstCalibratedAgg).toBeGreaterThan(-1);
      expect(firstCalibratedAgg).toBeLessThan(firstDone);
      expect(byType(events, "level")[0]).toMatchObject({ concurrency: 2, ok: 2 });
      expect(byType(events, "done")[0].summary.tokens).toBe(732);
    } finally {
      state.continuousUsage = false;
      state.importMode = "ok";
    }
  });

  test("a cancelled bench keeps the level rows it completed and a partial summary", async () => {
    state.importMode = "404";
    const res = await post("/api/streams/runs", { mode: "bench-decode", pack: "prose", base_url: BASE, levels: [1, 2], max_tokens: 120 });
    const { run_id } = (await res.json()) as { run_id: string };
    const events: StreamRunEvent[] = [];
    const seen = collect(run_id, (ev) => {
      events.push(ev);
      return ev.type === "done";
    });
    while (!events.some((e) => e.type === "level")) await Bun.sleep(10);
    while (!events.some((e) => e.type === "strand" && e.i === 1 && e.state === "decode")) await Bun.sleep(10);
    expect((await app.request(`/api/streams/runs/${run_id}/stop`, { method: "POST" })).status).toBe(200);
    const all = await seen;
    expect(byType(all, "level")).toHaveLength(1);
    expect(byType(all, "level")[0]).toMatchObject({ index: 0, concurrency: 1, ok: 1 });
    const done = byType(all, "done")[0];
    expect(done.summary.status).toBe("cancelled");
    expect(done.saved_run_id).toBeNull();
    expect(done.summary.ok).toBe(3); // strand 0 finished; strands 1–2 streamed output before the stop
    expect(done.summary.aggregate_tok_s).toBeGreaterThan(0);
    const snap = (await (await app.request(`/api/streams/runs/${run_id}`)).json()) as StreamRunSnapshot;
    expect(snap.status).toBe("cancelled");
    expect(snap.levels).toHaveLength(1);
    expect(snap.agg.length).toBe(byType(all, "agg").length);
    state.importMode = "ok";
  });

  test("bench-prefill: sizes via /tokenize, max_tokens 1, sizes ≥ max_model_len skipped with reason", async () => {
    state.requests = [];
    state.imported = [];
    state.importMode = "404"; // import route not deployed yet → saved_run_id null, run still completes
    const res = await post("/api/streams/runs", { mode: "bench-prefill", pack: "prose", base_url: BASE, sizes: [8192, 64] });
    const { run_id } = (await res.json()) as { run_id: string };
    const events = await collect(run_id);

    expect(byType(events, "hello")[0]).toMatchObject({ mode: "bench-prefill", sizes: [64, 8192], max_tokens: 1, n: 2 });
    expect(state.requests).toHaveLength(1);
    expect(state.requests[0]).toMatchObject({ max_tokens: 1, min_tokens: 1, ignore_eos: true });

    const levels = byType(events, "level");
    expect(levels).toHaveLength(2);
    expect(levels[0]).toMatchObject({ index: 0, size: 64, ok: 1, requests: 1 });
    expect(Math.abs(levels[0].prompt_tokens! - 64) / 64).toBeLessThanOrEqual(0.02);
    expect(levels[0].prefill_tok_s).toBeGreaterThan(0);
    expect(levels[1]).toMatchObject({ index: 1, size: 8192, ok: 0, requests: 0, skipped: "size 8192 ≥ max_model_len 4096" });
    const skipped = byType(events, "strand").filter((s) => s.i === 1).pop()!;
    expect(skipped.state).toBe("cancelled");

    const done = byType(events, "done")[0];
    expect(done.summary.status).toBe("done");
    expect(done.saved_run_id).toBeNull();
    expect(done.summary.headline!.prefill_tok_per_s_sustained).toBeGreaterThan(0);
    expect(done.summary.headline!.decode_tok_per_s_median_c1).toBeNull();
    state.importMode = "ok";
  });
});
