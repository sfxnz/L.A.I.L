import { describe, expect, test } from "bun:test";
import {
  aggregateTokPerS,
  buildEnvelope,
  decodeTimeS,
  decodeTokPerS,
  headline,
  percentile,
  prefillTokPerS,
  summarizeWave,
  toLevelEvent,
  tpotS,
  ttftS,
  type StrandResult,
} from "./metrics";

function result(over: Partial<StrandResult> & { i: number }): StrandResult {
  return {
    ok: true,
    t_start: 0,
    t_first: 100,
    t_last: 2100,
    t_end: 2110,
    completion_tokens: 101,
    prompt_tokens: 70,
    estimated: false,
    finish_reason: "length",
    error: null,
    token_times_ms: [],
    ...over,
  };
}

describe("metric definitions (perf.py)", () => {
  test("nearest-rank percentiles", () => {
    const xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    expect(percentile(xs, 50)).toBe(5);
    expect(percentile(xs, 95)).toBe(10);
    expect(percentile(xs, 99)).toBe(10);
    expect(percentile([3], 50)).toBe(3);
    expect(percentile([1, 2, 3], 50)).toBe(2);
    expect(percentile([], 50)).toBeNull();
  });

  test("per-stream tok/s, TTFT, TPOT from t_start / t_first / t_last", () => {
    // 101 tokens, first at 100 ms, last at 2100 ms → 2.0 s decode, 100 gaps of 20 ms
    const r = result({ i: 0 });
    expect(ttftS(r)).toBe(0.1);
    expect(decodeTimeS(r)).toBe(2);
    expect(decodeTokPerS(r)).toBe(50.5);
    expect(tpotS(r)).toBeCloseTo(0.02, 9);
    expect(prefillTokPerS(r)).toBe(700);
  });

  test("single-chunk stream falls back to stream end for decode time; TPOT undefined", () => {
    const r = result({ i: 0, t_first: 500, t_last: 500, t_end: 600, completion_tokens: 1 });
    expect(decodeTimeS(r)).toBeCloseTo(0.1, 9);
    expect(tpotS(r)).toBeNull();
    expect(decodeTokPerS(r)).toBeCloseTo(10, 9);
  });

  test("wave aggregate = Σ completion_tokens / (max t_last − min t_start), failed strands excluded", () => {
    const wave = [
      result({ i: 0, t_start: 0, t_last: 2000, completion_tokens: 100 }),
      result({ i: 1, t_start: 500, t_last: 4000, completion_tokens: 200 }),
      result({ i: 2, ok: false, error: "HTTP 500", completion_tokens: 999, t_last: 9000 }),
    ];
    expect(aggregateTokPerS(wave)).toBe(300 / 4);
    expect(aggregateTokPerS([])).toBeNull();
  });

  test("summarizeWave and level event carry medians, percentiles and errors", () => {
    const wave = [
      result({ i: 0, t_first: 100, t_last: 2100, completion_tokens: 101 }), // 50.5 tok/s, ttft 0.1
      result({ i: 1, t_first: 300, t_last: 2300, completion_tokens: 201 }), // 100.5 tok/s, ttft 0.3
      result({ i: 2, ok: false, error: "no_output", completion_tokens: null, t_first: null, t_last: null }),
    ];
    const ws = summarizeWave(wave);
    expect(ws.ok).toBe(2);
    expect(ws.requests).toBe(3);
    expect(ws.errors).toEqual(["#2: no_output"]);
    expect(ws.tokens).toBe(302);
    expect(ws.per_stream_median_tok_s).toBe(75.5);
    expect(ws.ttft_s).toEqual({ p50: 0.1, p95: 0.3, p99: 0.3 });
    expect(ws.aggregate_tok_s).toBe(Math.round((302 / 2.3) * 100) / 100);
    expect(ws.tpot_s).toBeCloseTo(0.015, 6); // median of 0.02 and 0.01
    const lvl = toLevelEvent(1, { concurrency: 3 }, ws);
    expect(lvl).toMatchObject({ type: "level", index: 1, concurrency: 3, ok: 2, requests: 3, ttft_p50_ms: 100, ttft_p95_ms: 300, tpot_ms: 15 });
  });

  test("envelope: arms, full_arms with per_request, headline from c1 and the peak arm", () => {
    const c1 = [result({ i: 0, token_times_ms: [100, 120] })];
    const c4 = [0, 1, 2, 3].map((i) => result({ i: i + 1, t_start: 0, t_first: 200, t_last: 3200, completion_tokens: 151 }));
    const env = buildEnvelope({
      kind: "decode",
      model: "m",
      workload: { pack: "prose", levels: [1, 4], max_tokens: 256, thinking: "off", temperature: 0.2, fill_to_max: true, base_url: "http://127.0.0.1:8000" },
      arms: [
        { key: { concurrency: 1 }, ws: summarizeWave(c1), results: c1 },
        { key: { concurrency: 4 }, ws: summarizeWave(c4), results: c4 },
      ],
    });
    expect(env.source).toBe("controller-streams");
    expect(env.metrics.arms.map((a) => a.concurrency)).toEqual([1, 4]);
    expect(env.metrics.arms[0].decode_tok_per_s_median).toBe(50.5);
    expect(env.metrics.arms[1].aggregate_tok_per_s).toBe(Math.round((604 / 3.2) * 100) / 100);
    expect(env.metrics.full_arms[1].per_request).toHaveLength(4);
    expect(env.metrics.full_arms[0].per_request[0]).toMatchObject({ i: 0, ttft_s: 0.1, decode_s: 2, tok_per_s: 50.5, token_times_ms: [100, 120] });
    expect(env.metrics.headline).toEqual({
      decode_tok_per_s_median_c1: 50.5,
      aggregate_peak_tok_per_s: env.metrics.arms[1].aggregate_tok_per_s,
      aggregate_peak_concurrency: 4,
      ttft_p50_s_c1: 0.1,
    });
    expect(env.summary).toEqual(env.metrics.headline);
  });

  test("prefill headline: sustained = largest completed size; skipped arms kept", () => {
    const small = [result({ i: 0, t_first: 1000, t_last: 1000, t_end: 1010, completion_tokens: 1, prompt_tokens: 8192 })];
    const big = [result({ i: 1, t_first: 5000, t_last: 5000, t_end: 5010, completion_tokens: 1, prompt_tokens: 32768 })];
    const arms = [
      { key: { size: 8192 }, ws: summarizeWave(small), results: small },
      { key: { size: 32768 }, ws: summarizeWave(big), results: big },
      { key: { size: 262144 }, ws: null, results: [], skipped: "size 262144 ≥ max_model_len 262144" },
    ];
    const env = buildEnvelope({
      kind: "prefill",
      model: "m",
      workload: { pack: "prose", sizes: [8192, 32768, 262144], max_tokens: 1, thinking: "off", temperature: 0.2, fill_to_max: true, base_url: "http://127.0.0.1:8000" },
      arms,
    });
    expect(env.metrics.arms[0].prefill_tok_per_s).toBe(8192);
    expect(env.metrics.arms[1].prefill_tok_per_s).toBeCloseTo(6553.6, 1);
    expect(env.metrics.arms[2].skipped).toContain("max_model_len");
    expect(headline("prefill", env.metrics.arms).prefill_tok_per_s_sustained).toBeCloseTo(6553.6, 1);
  });
});
