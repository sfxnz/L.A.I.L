import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { Hono } from "hono";
import type { StreamRunEvent, StreamRunSnapshot } from "@lail/shared";
import { config } from "../config";
import { RATE_WINDOW_MS, StreamsEngine, Subscriber, arrivalDelays, strandWindowRate } from "./engine";
import { createStreamsRoutes } from "./routes";
import { assignPrompts, getPack, listPacks, strandSystemPrompt } from "./packs";
import { parseMetrics, serverDelta } from "./probes";
import { SseParser } from "./sse-parser";

// ── Mock OpenAI server: reasoning + content deltas, finish_reason, trailing usage frame ──

type MockState = {
  requests: Array<Record<string, unknown>>;
  aborted: number;
  importMode: "ok" | "404";
  imported: unknown[];
  /** Honour `stream_options.continuous_usage_stats` (vLLM): `usage` on every chunk. */
  continuousUsage: boolean;
  /** The finish chunk (empty delta) and final usage count one more token: an EOS with no text. */
  eosTail: boolean;
  /** /metrics reads its running count when the request arrives, then answers this much later. */
  metricsLagMs: number;
  /** Streaming completions in flight (the mock's `vllm:num_requests_running`). */
  active: number;
  /** Requests running on the "server" that are not the engine's. */
  foreign: number;
  /** Cumulative spec-decode counters: every finished stream adds 10 drafts / 30 draft tokens / 24 accepted. */
  specDrafts: number;
  /** Streaming completions answer HTTP 500 (the warmup, non-streaming, still succeeds). */
  failStream: boolean;
  leaseMode: "ok" | "busy" | "down";
  leases: string[];
  statusCalls: number;
};
const state: MockState = {
  requests: [],
  aborted: 0,
  importMode: "ok",
  imported: [],
  continuousUsage: false,
  eosTail: false,
  metricsLagMs: 0,
  active: 0,
  foreign: 0,
  specDrafts: 0,
  failStream: false,
  leaseMode: "ok",
  leases: [],
  statusCalls: 0,
};
const isWarmup = (r: Record<string, unknown>) => String((r.messages as Array<{ content: string }>)[0].content).startsWith("Warmup");
const measured = () => state.requests.filter((r) => !isWarmup(r));

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
  if (state.eosTail) outChunks++;
  frames.push(frame({}, "length", usage()));
  const completionTokens = (n + (n > 1 ? 2 : 0)) * tokensPerChunk + (state.eosTail ? tokensPerChunk : 0);
  frames.push(`data: ${JSON.stringify({ id: "cmpl-1", choices: [], usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens } })}\r\n\r\n`);
  frames.push("data: [DONE]\n\n");
  // Split the first content frame across two writes to exercise the parser over HTTP.
  const split = frames.findIndex((f) => f.includes("tok0"));
  const [a, b] = [frames[split].slice(0, 12), frames[split].slice(12)];
  frames.splice(split, 1, a, b);

  const enc = new TextEncoder();
  signal.addEventListener("abort", () => state.aborted++, { once: true });
  const streaming = body.stream === true && !isWarmup(body);
  if (streaming) state.active++;
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
        if (streaming) {
          state.active--;
          state.specDrafts++;
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
      if (state.failStream && body.stream === true && !isWarmup(body)) return new Response("boom", { status: 500 });
      return completions(body, req.signal);
    }
    if (url.pathname === "/metrics") {
      const m = `engine="0",model_name="mock/served"`;
      const running = state.active + state.foreign;
      if (state.metricsLagMs) await Bun.sleep(state.metricsLagMs);
      const d = state.specDrafts;
      return new Response(
        [
          "# HELP vllm:num_requests_running running",
          `vllm:num_requests_running{${m}} ${running}.0`,
          `vllm:num_requests_waiting{${m}} 0.0`,
          `vllm:spec_decode_num_drafts_total{${m}} ${d * 10}.0`,
          `vllm:spec_decode_num_draft_tokens_total{${m}} ${d * 30}.0`,
          `vllm:spec_decode_num_accepted_tokens_total{${m}} ${d * 24}.0`,
          `vllm:spec_decode_num_accepted_tokens_per_pos_total{${m},position="0"} ${d * 9}.0`,
          `vllm:time_to_first_token_seconds_sum{${m}} ${d * 0.05}`,
          `vllm:time_to_first_token_seconds_count{${m}} ${d}.0`,
          "",
        ].join("\n"),
      );
    }
    if (url.pathname === "/api/status") {
      state.statusCalls++;
      return Response.json({
        sampled_at: `2026-10-01T00:00:${String(state.statusCalls).padStart(2, "0")}Z`,
        engine: { flags_fingerprint: "fp-mock" },
        // every node carries its own fresh reading and `sampled_at`; a down peer has none
        cluster: {
          nodes: [
            { id: "spark1", hostname: "spark1", sampled_at: 1000 * state.statusCalls, temperature_c: 50 + state.statusCalls, power_w: 100 + state.statusCalls, available_gib: 20 },
            { id: "spark2", sampled_at: 1000 * state.statusCalls, temperature_c: 45, power_w: 60 + (state.statusCalls % 2) * 10, available_gib: 30 },
            { id: "spark3", online: false, state: "offline", sampled_at: null, temperature_c: null, power_w: null, available_gib: null },
          ],
        },
      });
    }
    if (url.pathname.startsWith("/api/bench/lease")) {
      if (state.leaseMode === "down") return new Response("down", { status: 502 });
      if (req.method === "DELETE") {
        state.leases.push(`release:${url.pathname.split("/").pop()}`);
        return Response.json({ released: true });
      }
      const body = (await req.json()) as { lease_id: string };
      if (state.leaseMode === "busy") {
        return Response.json({ detail: { error: "bench_busy", job_id: "j1", kind: "agentic_tool_eval" } }, { status: 409 });
      }
      state.leases.push(`take:${body.lease_id}`);
      return Response.json({ lease_id: body.lease_id });
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

const engine = new StreamsEngine({ idleAbortMs: 300, idleWaitMs: 200 });
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
  test("six packs; the four decode families; mixed rotates", async () => {
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
      // 8 one-token chunks → 7 decode steps, terminal emit carries all of them
      expect(last.step_ms).toHaveLength(7);
      expect(last.step_tokens).toEqual([1, 1, 1, 1, 1, 1, 1]);
      expect(last.steps_append).toBeUndefined();
      // live decode emits carry only the steps since the previous emit, never one twice
      const live = strands.filter((s) => s.state === "decode" && s.steps_append);
      expect(live.reduce((a, s) => a + s.step_ms!.length, 0)).toBeLessThanOrEqual(7);
      expect(deltas.reduce((a, d) => a + d.tokens, 0)).toBe(8);
    }

    const aggs = byType(events, "agg");
    expect(aggs.length).toBeGreaterThan(0);
    const lastAgg = aggs[aggs.length - 1];
    // No continuous usage in this mock: live counts are chunk counts and the agg says so.
    expect(lastAgg).toMatchObject({ running: 0, waiting: 0, done: 2, tokens: 16, tokens_exact: false });
    expect(lastAgg.peak_tok_s).toBe(0); // the run never decoded for a full rate window
    expect(byType(events, "level")).toHaveLength(0);

    const done = byType(events, "done")[0];
    expect(events[events.length - 1]).toBe(done);
    expect(done.saved_run_id).toBeNull();
    expect(done.summary).toMatchObject({ status: "done", mode: "load", ok: 2, requests: 2, tokens: 16, errors: [] });
    expect(done.summary.aggregate_tok_s).toBeGreaterThan(0);
    expect(done.summary.aggregate_steady_tok_s).toBeGreaterThan(0);
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

  test("a load run nobody ever subscribes to is aborted too", async () => {
    const run_id = await engine.createRun({ mode: "load", pack: "prose", base_url: BASE, n: 1, max_tokens: 120 });
    await Bun.sleep(150);
    expect(engine.snapshot(run_id)!.status).toBe("running");
    await Bun.sleep(400);
    expect(engine.snapshot(run_id)!.status).toBe("cancelled");
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
    state.leases = [];
    const res = await post("/api/streams/runs", { mode: "bench-decode", pack: "prose", base_url: BASE, levels: [2, 1], max_tokens: 4, samples: 1 });
    expect(res.status).toBe(201);
    const { run_id } = (await res.json()) as { run_id: string };
    const events = await collect(run_id);

    const hello = byType(events, "hello")[0];
    expect(hello).toMatchObject({ mode: "bench-decode", levels: [1, 2], n: 3, samples: 1, serve_fingerprint: "fp-mock" });
    expect(hello.prompts.map((p) => p.level)).toEqual([0, 1, 1]);
    // one discarded warmup first (non-streaming, own nonce), then the measured strands
    expect(isWarmup(state.requests[0])).toBe(true);
    expect(state.requests[0]).toMatchObject({ stream: false, max_tokens: 16 });
    expect(measured()).toHaveLength(3);
    expect(measured().every((r) => r.min_tokens === 4 && r.ignore_eos === true)).toBe(true);
    // the serve-engine bench lease is held for the run and released at the end
    expect(state.leases).toEqual([`take:${run_id}`, `release:${run_id}`]);

    const levels = byType(events, "level");
    expect(levels).toHaveLength(2);
    expect(levels[0]).toMatchObject({ index: 0, concurrency: 1, ok: 1, requests: 1, errors: [] });
    expect(levels[1]).toMatchObject({ index: 1, concurrency: 2, ok: 2, requests: 2 });
    expect(levels[1].aggregate_tok_s).toBeGreaterThan(0);
    expect(levels[1].aggregate_steady_tok_s).toBeGreaterThanOrEqual(levels[1].aggregate_tok_s!);
    expect(levels[1].per_stream_median_tok_s).toBeGreaterThan(0);
    expect(levels[1].ttft_p50_ms).toBeGreaterThan(0);
    // nothing else ran on the server; vLLM's own view of the level rode along
    expect(levels[1].foreign_max).toBe(0);
    expect(levels[1].server).toMatchObject({ spec_acceptance: 0.8, spec_tokens_per_step: 3.4, ttft_mean_ms: 50 });
    // wave 2 starts only after wave 1 finished
    const strand0Done = events.findIndex((e) => e.type === "strand" && e.i === 0 && e.state === "done");
    const strand1Start = events.findIndex((e) => e.type === "strand" && e.i === 1);
    expect(strand1Start).toBeGreaterThan(strand0Done);

    const done = byType(events, "done")[0];
    expect(done.saved_run_id).toBe("imported-123");
    const head = done.summary.headline!;
    expect(head).toMatchObject({ aggregate_peak_concurrency: 2 });
    expect(head.decode_tok_per_s_median_c1).toBeGreaterThan(0);
    // run-level numbers are the headline's, never one aggregate across sequential levels
    expect(done.summary.aggregate_tok_s).toBe(head.aggregate_peak_tok_per_s);
    expect(done.summary.per_stream_median_tok_s).toBe(head.decode_tok_per_s_median_c1);
    expect(done.summary.aggregate_steady_tok_s).toBe(levels[1].aggregate_steady_tok_s);

    expect(state.imported).toHaveLength(1);
    const env = state.imported[0] as Record<string, any>;
    expect(env.kind).toBe("decode");
    expect(env.source).toBe("controller-streams");
    expect(env.model).toBe("mock/served");
    expect(env.workload).toEqual({ pack: "prose", levels: [1, 2], samples: 1, max_tokens: 4, thinking: "off", temperature: 0.2, fill_to_max: true, base_url: BASE, serve_fingerprint: "fp-mock" });
    expect(env.metrics.arms.map((a: { concurrency: number }) => a.concurrency)).toEqual([1, 2]);
    expect(env.metrics.arms[1].aggregate_steady_tok_per_s).toBe(levels[1].aggregate_steady_tok_s);
    expect(env.metrics.full_arms[1].per_request).toHaveLength(2);
    expect(env.metrics.full_arms[1].per_request[0].token_times_ms).toHaveLength(6);
    expect(env.metrics.full_arms[1].per_request[0].token_counts).toEqual([1, 2, 3, 4, 5, 6]);
    expect(env.metrics.full_arms[1].per_request[0].completion_tokens).toBe(6);
    expect(env.summary).toEqual({ ...env.metrics.headline, pack: "prose", max_tokens: 4, serve_fingerprint: "fp-mock", energy_j_per_token: env.metrics.hardware.energy_j_per_token });
    // node temperature / power over the run (status snapshots), and energy per token
    const hw = env.metrics.hardware;
    expect(hw.nodes.map((n: { id: string }) => n.id).sort()).toEqual(["spark1", "spark2"]);
    expect(hw.series.length).toBeGreaterThanOrEqual(4);
    expect(hw.nodes.find((n: { id: string }) => n.id === "spark2")).toMatchObject({ available_min_gib: 30, temp_max_c: 45 });
    expect(hw.nodes.find((n: { id: string }) => n.id === "spark1").available_min_gib).toBe(20);
    expect(hw.energy_j).toBeGreaterThanOrEqual(0);
  });

  test("bench-decode repeats: ⌈samples ÷ c⌉ waves per level, medians with min–max", async () => {
    state.importMode = "404";
    const res = await post("/api/streams/runs", { mode: "bench-decode", pack: "prose", base_url: BASE, levels: [1, 2], max_tokens: 4 });
    const { run_id } = (await res.json()) as { run_id: string };
    const events = await collect(run_id);
    const hello = byType(events, "hello")[0];
    expect(hello).toMatchObject({ n: 7, samples: 3 });
    expect(hello.prompts.map((p) => [p.level, p.wave])).toEqual([[0, 0], [0, 1], [0, 2], [1, 0], [1, 0], [1, 1], [1, 1]]);
    const [l1, l2] = byType(events, "level");
    expect(l1).toMatchObject({ concurrency: 1, samples: 3, ok: 3, requests: 3 });
    expect(l2).toMatchObject({ concurrency: 2, samples: 2, ok: 4, requests: 4 });
    expect(l1.aggregate_range![0]).toBeLessThanOrEqual(l1.aggregate_tok_s!);
    expect(l1.aggregate_range![1]).toBeGreaterThanOrEqual(l1.aggregate_tok_s!);
    expect(l1.ttft_p95_ms).toBeNull(); // 3 samples: no tail percentile
    // waves run one after another: the second ×1 strand starts after the first finished
    const firstDone = events.findIndex((e) => e.type === "strand" && e.i === 0 && e.state === "done");
    expect(events.findIndex((e) => e.type === "strand" && e.i === 1)).toBeGreaterThan(firstDone);
    state.importMode = "ok";
  });

  test("foreign requests on the server: waits, then measures and tags the level as contended", async () => {
    state.importMode = "404";
    state.foreign = 1;
    try {
      const t0 = performance.now();
      const res = await post("/api/streams/runs", { mode: "bench-decode", pack: "prose", base_url: BASE, levels: [1], max_tokens: 120, samples: 1 });
      const events = await collect(((await res.json()) as { run_id: string }).run_id);
      expect(performance.now() - t0).toBeGreaterThan(200); // waited idleWaitMs for the server to drain
      expect(byType(events, "level")[0].foreign_max).toBe(1);
    } finally {
      state.foreign = 0;
      state.importMode = "ok";
    }
  });

  test("foreign load: a strand of ours that finishes during the scrape is not counted as foreign", async () => {
    state.importMode = "404";
    state.metricsLagMs = 1500;
    try {
      // one ~2 s strand; the 1 Hz poll reads running=1 at ~1 s and answers at ~2.5 s, after it closed
      const res = await post("/api/streams/runs", { mode: "bench-decode", pack: "prose", base_url: BASE, levels: [1], max_tokens: 120, samples: 1 });
      const events = await collect(((await res.json()) as { run_id: string }).run_id);
      expect(byType(events, "level")[0].foreign_max).toBe(0);
    } finally {
      state.metricsLagMs = 0;
      state.importMode = "ok";
    }
  }, 15000);

  test("a bench is refused while the serve-engine holds the bench lease; runs without it when unreachable", async () => {
    state.leaseMode = "busy";
    try {
      const res = await post("/api/streams/runs", { mode: "bench-decode", pack: "prose", base_url: BASE, levels: [1], max_tokens: 2, samples: 1 });
      expect(res.status).toBe(409);
      expect((await res.json()) as object).toMatchObject({ error: "bench_busy", job_id: "j1" });
      // load runs are not benches
      const load = await post("/api/streams/runs", { mode: "load", pack: "prose", base_url: BASE, n: 1, max_tokens: 1 });
      expect(load.status).toBe(201);
      await collect(((await load.json()) as { run_id: string }).run_id);
      state.leaseMode = "down";
      state.importMode = "404";
      const ok = await post("/api/streams/runs", { mode: "bench-decode", pack: "prose", base_url: BASE, levels: [1], max_tokens: 2, samples: 1 });
      expect(ok.status).toBe(201);
      const done = byType(await collect(((await ok.json()) as { run_id: string }).run_id), "done")[0];
      expect(done.summary.status).toBe("done");
    } finally {
      state.leaseMode = "ok";
      state.importMode = "ok";
    }
  });

  test("a bench where every strand failed is an error, never an imported result", async () => {
    state.imported = [];
    state.failStream = true;
    try {
      const res = await post("/api/streams/runs", { mode: "bench-decode", pack: "prose", base_url: BASE, levels: [1, 2], max_tokens: 4, samples: 1 });
      const events = await collect(((await res.json()) as { run_id: string }).run_id);
      const done = byType(events, "done")[0];
      expect(done.summary.status).toBe("error");
      expect(done.summary.error).toContain("no level completed");
      expect(done.saved_run_id).toBeNull();
      expect(state.imported).toHaveLength(0);
    } finally {
      state.failStream = false;
    }
  });

  test("continuous usage: live tokens are exact per chunk (MTP chunks carry 3), steps carry their token counts", async () => {
    state.continuousUsage = true;
    state.importMode = "404";
    try {
      // max_tokens 5 → the mock reports 3 tokens per chunk
      const res = await post("/api/streams/runs", { mode: "bench-decode", pack: "prose", base_url: BASE, levels: [1, 2], max_tokens: 5, samples: 1 });
      const { run_id } = (await res.json()) as { run_id: string };
      const events = await collect(run_id);
      // The first decode event of every strand already carries the exact count — nothing to calibrate.
      for (const i of [0, 1, 2]) expect(byType(events, "strand").find((s) => s.i === i && s.state === "decode")!.tokens).toBe(3);
      const last = byType(events, "strand").filter((s) => s.i === 2).pop()!;
      expect(last.tokens).toBe(21);
      expect(new Set(last.step_tokens)).toEqual(new Set([3]));
      expect(byType(events, "delta").filter((d) => d.i === 2).reduce((a, d) => a + d.tokens, 0)).toBe(21);
      expect(byType(events, "agg").every((a) => a.tokens_exact)).toBe(true);
      expect(byType(events, "done")[0].summary.tokens).toBe(63);
    } finally {
      state.continuousUsage = false;
      state.importMode = "ok";
    }
  });

  test("continuous usage: an EOS token after the last text chunk is outside the decode span", async () => {
    state.continuousUsage = true;
    state.eosTail = true;
    state.imported = [];
    try {
      const res = await post("/api/streams/runs", { mode: "bench-decode", pack: "prose", base_url: BASE, levels: [1], max_tokens: 4, samples: 1 });
      await collect(((await res.json()) as { run_id: string }).run_id);
      const r = (state.imported[0] as Record<string, any>).metrics.full_arms[0].per_request[0];
      expect(r.completion_tokens).toBe(7); // 2 reasoning + 4 content + EOS
      expect(r.token_counts.at(-1)).toBe(6); // tokens at t_last
      // decode tok/s × decode span = tokens after the first chunk up to t_last: 6 − 1
      expect(Math.round(r.tok_per_s * r.decode_s)).toBe(5);
    } finally {
      state.continuousUsage = false;
      state.eosTail = false;
    }
  });

  test("without per-chunk usage, live counts are chunk counts (tokens_exact false); terminal counts come from usage", async () => {
    const res = await post("/api/streams/runs", { mode: "load", pack: "prose", base_url: BASE, n: 1, max_tokens: 5 });
    const events = await collect(((await res.json()) as { run_id: string }).run_id);
    expect(byType(events, "strand").find((s) => s.state === "decode")!.tokens).toBe(1);
    expect(byType(events, "strand").pop()!.tokens).toBe(21);
    expect(byType(events, "agg").pop()!.tokens_exact).toBe(false);
  });

  test("a cancelled bench keeps the level rows it completed and a partial summary", async () => {
    state.importMode = "404";
    const res = await post("/api/streams/runs", { mode: "bench-decode", pack: "prose", base_url: BASE, levels: [1, 2], max_tokens: 120, samples: 1 });
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
    expect(done.summary.aggregate_tok_s).toBe(byType(all, "level")[0].aggregate_tok_s); // the completed level's
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

    // two requests per size by default
    expect(byType(events, "hello")[0]).toMatchObject({ mode: "bench-prefill", sizes: [64, 8192], max_tokens: 1, n: 4, samples: 2 });
    expect(measured()).toHaveLength(2);
    expect(measured().every((r) => r.max_tokens === 1 && r.min_tokens === 1 && r.ignore_eos === true)).toBe(true);
    // each repeat has its own nonce'd prompt, so prefix caching cannot serve the second
    const prompts = measured().map((r) => (r.messages as Array<{ content: string }>)[0].content);
    expect(prompts[0]).not.toBe(prompts[1]);

    const levels = byType(events, "level");
    expect(levels).toHaveLength(2);
    expect(levels[0]).toMatchObject({ index: 0, size: 64, ok: 2, requests: 2, samples: 2 });
    expect(Math.abs(levels[0].prompt_tokens! - 64) / 64).toBeLessThanOrEqual(0.02);
    expect(levels[0].prefill_tok_s).toBeGreaterThan(0);
    expect(levels[0].per_stream_range).not.toBeNull();
    expect(levels[1]).toMatchObject({ index: 1, size: 8192, ok: 0, requests: 0, skipped: "size 8192 ≥ max_model_len 4096" });
    for (const i of [2, 3]) expect(byType(events, "strand").filter((s) => s.i === i).pop()!.state).toBe("cancelled");

    const done = byType(events, "done")[0];
    expect(done.summary.status).toBe("done");
    expect(done.saved_run_id).toBeNull();
    expect(done.summary.headline!.prefill_tok_per_s_sustained).toBeGreaterThan(0);
    expect(done.summary.headline!.decode_tok_per_s_median_c1).toBeNull();
    // 1-token answers: no energy per output token
    expect(done.summary.hardware!.energy_j_per_token).toBeNull();
    state.importMode = "ok";
  });
});

describe("live rates and fan-out", () => {
  test("per-strand window rate counts tokens in the last window and decays to 0 when the strand stalls", () => {
    // tokens arrive 4 at a time every 50 ms from t=100 to t=3050 (relative to t_start=0)
    const times: number[] = [];
    const counts: number[] = [];
    for (let k = 0, t = 100; t <= 3050; k++, t += 50) {
      times.push(t);
      counts.push((k + 1) * 4);
    }
    const s = { t_start: 0, t_first: 100, token_times: times, token_counts: counts };
    expect(RATE_WINDOW_MS).toBe(3000);
    // before the window has filled: all 60 chunks over the 2.95 s since the first token
    expect(strandWindowRate(s, 3050)).toBeCloseTo(240 / 2.95, 6);
    // full window (100, 3100]: the 59 chunks after t=100 → 236 tokens / 3 s
    expect(strandWindowRate(s, 3100)).toBeCloseTo(236 / 3, 6);
    // 1.55 s into a stall: (1600, 4600] holds 29 chunks; past a full window it reads 0
    expect(strandWindowRate(s, 4600)).toBeCloseTo(116 / 3, 6);
    expect(strandWindowRate(s, 6100)).toBe(0);
    // first second of a strand divides by 1 s, not by 50 ms
    expect(strandWindowRate({ t_start: 0, t_first: 100, token_times: [100, 150], token_counts: [1, 5] }, 150)).toBe(5);
  });

  test("a lagging subscriber's deltas merge per stream instead of being dropped", async () => {
    const sub = new Subscriber();
    for (let k = 0; k < 500; k++) {
      sub.push({ type: "delta", i: 0, text: `t${k} `, reasoning: false, chunks: 1, tokens: 4 });
      sub.push({ type: "delta", i: 0, text: "r", reasoning: true, chunks: 1, tokens: 1 });
      sub.push({ type: "delta", i: 1, text: "x", reasoning: false, chunks: 1, tokens: 2 });
    }
    const got: StreamRunEvent[] = [];
    for (let k = 0; k < 3; k++) got.push((await sub.next())!);
    const d = got as Array<Extract<StreamRunEvent, { type: "delta" }>>;
    expect(d.map((e) => [e.i, e.reasoning])).toEqual([[0, false], [0, true], [1, false]]);
    expect(d[0].text).toBe(Array.from({ length: 500 }, (_, k) => `t${k} `).join(""));
    expect(d[0]).toMatchObject({ chunks: 500, tokens: 2000 });
    expect(d[2]).toMatchObject({ chunks: 500, tokens: 1000 });
    // consumed: the next delta starts a new event
    sub.push({ type: "delta", i: 0, text: "next", reasoning: false, chunks: 1, tokens: 1 });
    expect(((await sub.next()) as { text: string }).text).toBe("next");
  });

  test("vLLM /metrics parsing and per-level deltas", () => {
    const m = (d: number, run: number) =>
      parseMetrics(
        [
          `vllm:num_requests_running{engine="0",model_name="m"} ${run}.0`,
          `vllm:num_requests_running{engine="1",model_name="m"} 1.0`,
          `vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} ${1352 + d}.0`,
          `vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} ${4056 + 3 * d}.0`,
          `vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} ${3492 + 2 * d}.0`,
          `vllm:request_prefill_kv_computed_tokens_sum{engine="0",model_name="m"} ${1397 + 8192 * (d / 100)}`,
          `vllm:request_prefill_time_seconds_sum{engine="0",model_name="m"} ${5.36 + d / 100}`,
          `vllm:num_preemptions_total{engine="0",model_name="m"} 0.0`,
        ].join("\n"),
      );
    expect(m(0, 2)["vllm:num_requests_running"]).toBe(3); // summed over label sets
    expect(serverDelta(m(0, 0), m(100, 0))).toEqual({
      spec_acceptance: 0.667,
      spec_tokens_per_step: 3,
      ttft_mean_ms: null,
      prefill_tok_s: 8192,
      preemptions: 0,
    });
    expect(serverDelta(null, m(1, 0))).toBeNull();
  });

  test("energy is integrated only when every node's power was re-read during the run", async () => {
    const { summarizeHardware } = await import("./probes");
    // spark1 at 100 W then 120 W over 2 s, spark2 at 60 → 70 W: (110 + 65) × 2 = 350 J over 100 tokens
    const fresh = summarizeHardware(
      [
        [0, "spark1", 50, 100, 20],
        [0, "spark2", 45, 60, 30],
        [2000, "spark1", 52, 120, 19],
        [2000, "spark2", 46, 70, 30],
      ],
      [[0, 2000]],
      100,
    );
    expect(fresh).toMatchObject({ energy_j: 350, energy_j_per_token: 3.5 });
    expect(fresh.nodes[0]).toMatchObject({ id: "spark1", temp_max_c: 52, power_mean_w: 110, available_min_gib: 19 });
    // spark2's reading was never refreshed during the run (one reading): no energy
    const stale = summarizeHardware(
      [
        [0, "spark1", 50, 100, 20],
        [0, "spark2", 45, 60, 30],
        [2000, "spark1", 52, 120, 19],
      ],
      [[0, 2000]],
      100,
    );
    expect(stale.energy_j).toBeNull();
    expect(stale.energy_j_per_token).toBeNull();
  });

  test("energy covers only the measured level windows; none per token for prefill", async () => {
    const { summarizeHardware } = await import("./probes");
    const series: Parameters<typeof summarizeHardware>[0] = [
      [0, "spark1", 50, 100, 20],
      [0, "spark2", 45, 60, 30],
      [2000, "spark1", 52, 120, 19],
      [2000, "spark2", 46, 70, 30],
    ];
    // [0.5 s, 1.5 s]: spark1 105 → 115 W, spark2 62.5 → 67.5 W → (110 + 65) × 1 s = 175 J
    expect(summarizeHardware(series, [[500, 1500]], 100)).toMatchObject({ energy_j: 175, energy_j_per_token: 1.75 });
    // two windows with an idle gap between them; a window past the last reading holds it flat
    expect(summarizeHardware(series, [[0, 500], [1500, 2000]], 100).energy_j).toBe(175);
    expect(summarizeHardware(series, [[2000, 3000]], 100).energy_j).toBe(190);
    // prefill: energy, but no per-output-token figure
    expect(summarizeHardware(series, [[0, 2000]], null)).toMatchObject({ energy_j: 350, energy_j_per_token: null });
    // no completed level: nothing measured
    expect(summarizeHardware(series, [], 100).energy_j).toBeNull();
  });
});
