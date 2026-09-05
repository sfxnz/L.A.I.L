/**
 * Reducer tests for the A2 SSE protocol consumer — synthetic events only.
 */
import { describe, expect, test } from "bun:test";
import {
  AGG_WINDOW_MS,
  TEXT_RING_CHARS,
  appendRing,
  initialStreamRunState,
  streamRunReducer,
  type StreamRunAction,
  type StreamRunState,
} from "./use-stream-run";
import type {
  StreamHelloEvent,
  StreamRunSnapshot,
  StreamRunSummary,
} from "./stream-run-types";

const hello: StreamHelloEvent = {
  type: "hello",
  run_id: "r1",
  mode: "load",
  model: "org/model",
  base_url: "http://127.0.0.1:8000",
  n: 2,
  max_tokens: 512,
  started_at: "2026-09-05T18:00:00Z",
  prompts: [
    { i: 0, title: "Prose", text: "Write an essay", pack: "prose" },
    { i: 1, title: "Code", text: "Implement LRU", pack: "code" },
  ],
};

function run(actions: StreamRunAction[], from: StreamRunState = initialStreamRunState()) {
  return actions.reduce(streamRunReducer, from);
}

describe("stream run reducer", () => {
  test("hello seeds one waiting strand per prompt", () => {
    const s = run([hello]);
    expect(s.hello?.run_id).toBe("r1");
    expect(s.strands).toHaveLength(2);
    expect(s.strands[1]).toMatchObject({ i: 1, title: "Code", pack: "code", state: "waiting", text: "" });
  });

  test("delta appends text; reasoning goes to its own buffer", () => {
    const s = run([
      hello,
      { type: "delta", i: 0, text: "Hello ", reasoning: false, chunks: 1 },
      { type: "delta", i: 0, text: "world", reasoning: false, chunks: 2 },
      { type: "delta", i: 0, text: "thinking…", reasoning: true, chunks: 3 },
    ]);
    expect(s.strands[0].text).toBe("Hello world");
    expect(s.strands[0].reasoning).toBe("thinking…");
    expect(s.strands[0].chunks).toBe(3);
  });

  test("text ring buffer holds the last 8k chars", () => {
    const big = "x".repeat(TEXT_RING_CHARS);
    expect(appendRing(big, "tail")).toHaveLength(TEXT_RING_CHARS);
    expect(appendRing(big, "tail").endsWith("tail")).toBe(true);
    const s = run([hello, { type: "delta", i: 1, text: big + "END", reasoning: false, chunks: 9 }]);
    expect(s.strands[1].text).toHaveLength(TEXT_RING_CHARS);
    expect(s.strands[1].text.endsWith("END")).toBe(true);
  });

  test("delta for an unseen strand creates a placeholder instead of dropping text", () => {
    const s = run([{ type: "delta", i: 3, text: "early", reasoning: false, chunks: 1 }]);
    expect(s.strands).toHaveLength(4);
    expect(s.strands[3].text).toBe("early");
    expect(s.strands[0].state).toBe("waiting");
  });

  test("strand events patch per-strand state and metrics", () => {
    const s = run([
      hello,
      { type: "strand", i: 0, state: "decode", ttft_ms: 412, tok_s: 54.2 },
      { type: "strand", i: 0, state: "done", tokens: 512, tok_s: 53.9, peak_tok_s: 61, finish_reason: "length", itl_ms: [18, 19] },
    ]);
    expect(s.strands[0]).toMatchObject({
      state: "done",
      ttft_ms: 412,
      tokens: 512,
      tok_s: 53.9,
      peak_tok_s: 61,
      finish_reason: "length",
      itl_ms: [18, 19],
    });
    expect(s.strands[1].state).toBe("waiting");
  });

  test("agg keeps a 60s sliding window, oldest first, and the latest sample", () => {
    // window floor after the last sample = 89_000 − 60_000 = 29_000 → 0 and 250 fall out
    const points: StreamRunAction[] = [0, 250, 30_000, 61_000, 89_000].map((t_ms) => ({
      type: "agg",
      t_ms,
      tok_s: t_ms / 1000,
      peak_tok_s: 100,
      tokens: t_ms,
      running: 2,
      waiting: 0,
      done: 0,
      tokens_per_chunk: 1,
    }));
    const s = run([hello, ...points]);
    expect(s.agg.map((p) => p.t_ms)).toEqual([30_000, 61_000, 89_000]);
    expect(s.agg[0].t_ms).toBeGreaterThanOrEqual(89_000 - AGG_WINDOW_MS);
    expect(s.latest?.tok_s).toBe(89);
  });

  test("level rows are upserted and sorted ascending by index", () => {
    const level = (index: number, aggregate_tok_s: number): StreamRunAction => ({
      type: "level",
      index,
      concurrency: 2 ** index,
      aggregate_tok_s,
      per_stream_median_tok_s: aggregate_tok_s / 2 ** index,
      ttft_p50_ms: 400,
      ttft_p95_ms: 600,
      ttft_p99_ms: 700,
      tpot_ms: 18,
      ok: 2 ** index,
      requests: 2 ** index,
      errors: [],
    });
    const s = run([hello, level(2, 120), level(0, 34), level(2, 126)]);
    expect(s.levels.map((l) => l.index)).toEqual([0, 2]);
    expect(s.levels[1].aggregate_tok_s).toBe(126);
  });

  test("skipped prefill level lands immediately with its reason", () => {
    const s = run([
      {
        type: "level",
        index: 5,
        size: 262_144,
        aggregate_tok_s: 0,
        per_stream_median_tok_s: 0,
        ttft_p50_ms: 0,
        ttft_p95_ms: 0,
        ttft_p99_ms: 0,
        tpot_ms: 0,
        ok: 0,
        requests: 0,
        errors: [],
        skipped: "exceeds max_model_len 131072",
      },
    ]);
    expect(s.levels[0].skipped).toContain("max_model_len");
  });

  const summary: StreamRunSummary = {
    status: "done",
    mode: "load",
    model: "org/model",
    duration_ms: 77_000,
    tokens: 4096,
    peak_tok_s: 207,
    aggregate_tok_s: 190.4,
    per_stream_median_tok_s: 24.1,
    ttft_p50_ms: 471,
    ttft_p95_ms: 902,
    ok: 8,
    requests: 8,
    errors: [],
  };

  test("done and error terminate; reset clears", () => {
    const done = run([hello, { type: "done", run_id: "r1", summary, saved_run_id: "s1" }]);
    expect(done.done?.saved_run_id).toBe("s1");
    expect(done.done?.summary.peak_tok_s).toBe(207);
    expect(done.error).toBeNull();
    const failed = run([hello, { type: "error", message: "backend 409" }]);
    expect(failed.error).toBe("backend 409");
    expect(run([{ type: "reset" }], failed)).toEqual(initialStreamRunState());
  });

  test("snapshot rebuilds the whole view after a reconnect", () => {
    const { type: _t, ...helloBody } = hello;
    void _t;
    const snapshot: StreamRunSnapshot = {
      hello: helloBody,
      status: "running",
      strands: [
        { i: 0, state: "done", tokens: 300, tok_s: 50, chunks: 300, text: "full text", reasoning_text: "why" },
        { i: 1, state: "decode", tok_s: 44, chunks: 120, text: "", reasoning_text: "" },
      ],
      agg: [
        { t_ms: 1000, tok_s: 40, peak_tok_s: 40, tokens: 40, running: 2, waiting: 0, done: 0, tokens_per_chunk: 1 },
        { t_ms: 70_000, tok_s: 90, peak_tok_s: 95, tokens: 900, running: 1, waiting: 0, done: 1, tokens_per_chunk: 1 },
      ],
      levels: [],
      done: null,
      error: null,
    };
    const stale = run([hello, { type: "delta", i: 0, text: "partial", reasoning: false, chunks: 1 }]);
    const s = streamRunReducer(stale, { type: "snapshot", snapshot });
    expect(s.strands[0].text).toBe("full text");
    expect(s.strands[0].reasoning).toBe("why");
    expect(s.strands[0].state).toBe("done");
    expect(s.strands[1]).toMatchObject({ state: "decode", tok_s: 44, title: "Code" });
    expect(s.agg.map((p) => p.t_ms)).toEqual([70_000]);
    expect(s.latest?.tok_s).toBe(90);
  });
});
