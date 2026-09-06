import { describe, expect, test } from "bun:test";
import { initialStreamRunState, streamRunReducer, type StreamRunAction, type StreamRunState } from "../use-stream-run";
import { aggregateMethod, exportStem, jsonSnapshot, markdownSummary } from "./export";

function build(actions: StreamRunAction[]): StreamRunState {
  return actions.reduce(streamRunReducer, initialStreamRunState());
}

const hello: StreamRunAction = {
  type: "hello",
  run_id: "r9",
  mode: "load",
  model: "nvidia/Qwen3.8-Flash-Next-NVFP4",
  base_url: "http://127.0.0.1:8000",
  n: 2,
  max_tokens: 96,
  max_model_len: 262144,
  started_at: "2026-09-05T20:00:00Z",
  prompts: [
    { i: 0, title: "prose_essay", text: "Continue this | essay", pack: "prose" },
    { i: 1, title: "haiku", text: "Write a haiku", pack: "chat-short" },
  ],
};

const live = build([
  hello,
  { type: "strand", i: 0, state: "decode", ttft_ms: 412 },
  { type: "delta", i: 0, text: "Decode throughput", reasoning: false, chunks: 4 },
  { type: "delta", i: 0, text: "hmm", reasoning: true, chunks: 1 },
  { type: "agg", t_ms: 2500, tok_s: 84.4, peak_tok_s: 90, tokens: 200, running: 1, waiting: 1, done: 0, tokens_per_chunk: 2.7, calibrated: true, ttft_p50_ms: 412, ttft_p95_ms: 412 },
]);

const finished = build([
  hello,
  { type: "strand", i: 0, state: "done", ttft_ms: 412, tokens: 96, tok_s: 54.2, peak_tok_s: 61, finish_reason: "length", itl_ms: [18, 20, 2400, 19] },
  { type: "strand", i: 1, state: "error", error: "HTTP 500: boom | bang" },
  {
    type: "done",
    run_id: "r9",
    summary: {
      status: "done",
      mode: "load",
      model: "nvidia/Qwen3.8-Flash-Next-NVFP4",
      duration_ms: 77_000,
      tokens: 96,
      peak_tok_s: 61,
      aggregate_tok_s: 52.1,
      per_stream_median_tok_s: 54.2,
      ttft_p50_ms: 412,
      ttft_p95_ms: 412,
      ok: 1,
      requests: 2,
      errors: ["#1: HTTP 500: boom | bang"],
    },
    saved_run_id: null,
  },
]);

describe("export shapes", () => {
  test("aggregateMethod names the number it quotes", () => {
    expect(aggregateMethod(live)).toEqual({ value: 84.4, method: "live · usage-calibrated ×2.7 tok/chunk" });
    expect(aggregateMethod(build([hello, { type: "agg", t_ms: 1, tok_s: 3, peak_tok_s: 3, tokens: 3, running: 1, waiting: 0, done: 0, tokens_per_chunk: 1, calibrated: false }])).method).toBe(
      "live · chunk estimate",
    );
    expect(aggregateMethod(finished)).toEqual({ value: 52.1, method: "final · usage" });
    // Cancelled: the engine reports aggregate_tok_s null — keep the last live number, name it.
    const cancelled = build([
      hello,
      { type: "agg", t_ms: 900, tok_s: 40, peak_tok_s: 41, tokens: 30, running: 1, waiting: 0, done: 0, tokens_per_chunk: 2.5, calibrated: true },
      { type: "done", run_id: "r9", summary: { ...finished.done!.summary, status: "cancelled", aggregate_tok_s: null }, saved_run_id: null },
    ]);
    expect(aggregateMethod(cancelled)).toEqual({ value: 40, method: "last sample · usage-calibrated ×2.5 tok/chunk" });
  });

  test("markdown summary: title, facts line, metric table and one strand row per strand", () => {
    const md = markdownSummary(finished, { pack: "prose", arrival: "burst", fill_to_max: false, thinking: "off", packLabel: (id) => (id === "prose" ? "Prose" : id) });
    const lines = md.split("\n");
    expect(lines[0]).toBe("# Streams · load · nvidia/Qwen3.8-Flash-Next-NVFP4");
    expect(lines[2]).toContain("run `r9`");
    expect(lines[2]).toContain("pack Prose");
    expect(lines[2]).toContain("natural EOS");
    expect(md).toContain("| Aggregate tok/s (final · usage) | 52.1 |");
    expect(md).toContain("| Peak tok/s (live 1 s window) | 61.0 |");
    expect(md).toContain("| TTFT p50 / p95 | 412 ms / 412 ms |");
    expect(md).toContain("| Duration | 1 m 17 s |");
    expect(md).toContain("| Strands ok | 1 / 2 |");
    const rows = lines.filter((l) => /^\| \d+ \| /.test(l));
    expect(rows).toHaveLength(2);
    // ITL p50 of [18, 19, 20, 2400] is 19 by nearest rank; p95 is the stall itself
    expect(rows[0]).toContain("| Prose | prose_essay | done | 412 ms | 96 | 54.2 | 61.0 | 19 / 2400 | — | length |");
    // pipes in error text are escaped so the table stays a table
    expect(rows[1]).toContain("HTTP 500: boom \\| bang");
    expect(md).not.toContain("### 1 ·");
  });

  test("live markdown quotes the live estimate and the strand counts", () => {
    const md = markdownSummary(live);
    expect(md).toContain("| Aggregate tok/s (live · usage-calibrated ×2.7 tok/chunk) | 84.4 |");
    expect(md).toContain("| Strands ok | 0 done · 1 streaming · 1 waiting |");
    expect(md).toContain("| 1 | prose | prose_essay | decode | 412 ms |");
    expect(md).toContain("| 20 % |"); // 1 reasoning chunk of 5
  });

  test("transcripts option appends prompt (quoted), thinking (collapsed) and text per strand", () => {
    const md = markdownSummary(live, { transcripts: true });
    expect(md).toContain("### 1 · prose_essay");
    expect(md).toContain("> Continue this | essay");
    expect(md).toContain("<details><summary>thinking</summary>");
    expect(md).toContain("\nhmm\n");
    expect(md).toContain("Decode throughput");
    expect(md).toContain("### 2 · haiku");
    expect(md).toContain("_(no output)_");
  });

  test("json snapshot carries hello, summary, every strand with text, and the agg series", () => {
    const j = jsonSnapshot(finished, { pack: "prose" }) as { strands: Array<Record<string, unknown>>; controls: unknown; agg: unknown[] };
    expect(j.controls).toEqual({ pack: "prose" });
    expect(j.strands).toHaveLength(2);
    expect(j.strands[0]).toMatchObject({ i: 0, state: "done", tokens: 96, itl_ms: [18, 20, 2400, 19], text: "" });
    expect(j.strands[1]).toMatchObject({ error: "HTTP 500: boom | bang", finish_reason: null });
    expect(Array.isArray(j.agg)).toBe(true);
    expect(exportStem(finished)).toBe("streams-Qwen3.8-Flash-Next-NVFP4-r9");
  });
});
