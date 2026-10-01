import { describe, expect, test } from "bun:test";
import { perTokenLatencies } from "@lail/shared";
import {
  aggregateSteadyTokPerS,
  aggregateTokPerS,
  buildEnvelope,
  decodeTimeS,
  decodeTokPerS,
  headline,
  median,
  percentile,
  prefillTokPerS,
  summarizeLevel,
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
    last_tokens: null,
    first_tokens: 1,
    prompt_tokens: 70,
    estimated: false,
    finish_reason: "length",
    error: null,
    token_times_ms: [],
    token_counts: [],
    ...over,
  };
}

const WORKLOAD = { pack: "prose", samples: 1, thinking: "off" as const, temperature: 0.2, fill_to_max: true, base_url: "http://127.0.0.1:8000", serve_fingerprint: "fp1" };

describe("metric definitions", () => {
  test("nearest-rank percentiles; median of an even count is the mean of the middle two", () => {
    const xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    expect(percentile(xs, 50)).toBe(5);
    expect(percentile(xs, 95)).toBe(10);
    expect(percentile(xs, 99)).toBe(10);
    expect(percentile([3], 50)).toBe(3);
    expect(percentile([1, 2, 3], 50)).toBe(2);
    expect(percentile([], 50)).toBeNull();
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([])).toBeNull();
  });

  test("decode tok/s excludes the first chunk's tokens, so it is exactly 1 / TPOT", () => {
    // 101 tokens, the first one at 100 ms, the last at 2100 ms → 100 tokens decoded in 2.0 s
    const r = result({ i: 0 });
    expect(ttftS(r)).toBe(0.1);
    expect(decodeTimeS(r)).toBe(2);
    expect(decodeTokPerS(r)).toBe(50);
    expect(tpotS(r)).toBeCloseTo(0.02, 9);
    expect(decodeTokPerS(r)! * tpotS(r)!).toBeCloseTo(1, 12);
    expect(prefillTokPerS(r)).toBe(700);
  });

  test("MTP: a 4-token first chunk is outside the decode span too", () => {
    // 2 s span; 4 tokens arrived with the first chunk at t_first → 96 tokens decoded
    const r = result({ i: 0, completion_tokens: 100, first_tokens: 4 });
    expect(decodeTokPerS(r)).toBe(48);
    expect(1 / tpotS(r)!).toBeCloseTo(48, 9);
  });

  test("a token after t_last (EOS / stop, empty delta) is outside the decode span", () => {
    // 100 tokens by t_last; usage then counts the EOS that arrived with no text → 101
    const r = result({ i: 0, completion_tokens: 101, last_tokens: 100 });
    expect(decodeTokPerS(r)).toBe(49.5); // (100 − 1) / 2.0 s, not (101 − 1) / 2.0 s
    expect(decodeTokPerS(r)! * tpotS(r)!).toBeCloseTo(1, 12);
    // wall-clock: 100 tokens over max t_last − min t_start = 2.1 s
    expect(aggregateTokPerS([r])).toBeCloseTo(100 / 2.1, 9);
    expect(aggregateSteadyTokPerS([r])).toBe(49.5);
    // totals still count every generated token
    expect(summarizeWave([r]).tokens).toBe(101);
  });

  test("a single chunk has no decode span: rate and TPOT undefined", () => {
    const r = result({ i: 0, t_first: 500, t_last: 500, t_end: 600, completion_tokens: 1 });
    expect(decodeTimeS(r)).toBeNull();
    expect(tpotS(r)).toBeNull();
    expect(decodeTokPerS(r)).toBeNull();
  });

  test("per-token ITL: a step of k tokens is k samples of gap / k", () => {
    // 3 steps of 46 ms carrying 4 tokens (MTP) and one 2 s stall carrying 1
    const lat = perTokenLatencies([46, 46, 46, 2000], [4, 4, 4, 1]);
    expect(lat).toHaveLength(13);
    expect(percentile(lat, 50)).toBe(11.5);
    expect(lat[lat.length - 1]).toBe(2000);
    expect(perTokenLatencies([10, 20], [0, 2])).toEqual([10, 10]);
  });

  test("wave aggregates: wall-clock spans the straggler's tail; decode span is the time-average of the live rate", () => {
    const wave = [
      result({ i: 0, t_start: 0, t_first: 500, t_last: 2000, completion_tokens: 100 }),
      result({ i: 1, t_start: 500, t_first: 1000, t_last: 4000, completion_tokens: 200 }),
      result({ i: 2, ok: false, error: "HTTP 500", completion_tokens: 999, t_last: 9000 }),
    ];
    expect(aggregateTokPerS(wave)).toBe(300 / 4);
    expect(aggregateSteadyTokPerS(wave)).toBeCloseTo((99 + 199) / 3.5, 9);
    expect(aggregateTokPerS([])).toBeNull();
    // ×1: the decode-span aggregate is the strand's own decode rate
    const one = [result({ i: 0 })];
    expect(aggregateSteadyTokPerS(one)).toBe(decodeTokPerS(one[0]));
  });

  test("summarizeWave and level event: medians, errors; p95/p99 withheld under 5 samples", () => {
    const wave = [
      result({ i: 0, t_first: 100, t_last: 2100, completion_tokens: 101 }), // 50 tok/s, ttft 0.1
      result({ i: 1, t_first: 300, t_last: 2300, completion_tokens: 201 }), // 100 tok/s, ttft 0.3
      result({ i: 2, ok: false, error: "no_output", completion_tokens: null, t_first: null, t_last: null }),
    ];
    const ws = summarizeWave(wave);
    expect(ws.ok).toBe(2);
    expect(ws.requests).toBe(3);
    expect(ws.errors).toEqual(["#2: no_output"]);
    expect(ws.tokens).toBe(302);
    expect(ws.per_stream_median_tok_s).toBe(75);
    expect(ws.ttft_s).toEqual({ p50: 0.1, p95: null, p99: null });
    expect(ws.aggregate_tok_s).toBe(Math.round((302 / 2.3) * 100) / 100);
    expect(ws.tpot_s).toBeCloseTo(0.015, 6); // median of 0.02 and 0.01
    expect(ws.samples).toBe(1);
    expect(ws.aggregate_range).toBeNull();
    expect(ws.per_stream_range).toEqual([50, 100]);
    const lvl = toLevelEvent(1, { concurrency: 3 }, ws, { foreign_max: 2, server: null });
    expect(lvl).toMatchObject({ type: "level", index: 1, concurrency: 3, ok: 2, requests: 3, ttft_p50_ms: 100, ttft_p95_ms: null, tpot_ms: 15, foreign_max: 2, samples: 1 });
    expect(lvl).not.toHaveProperty("prefill_tok_s"); // TTFT on a short prompt is not a prefill rate

    const five = [0, 1, 2, 3, 4].map((i) => result({ i, t_first: 100 * (i + 1) }));
    expect(summarizeWave(five).ttft_s).toEqual({ p50: 0.3, p95: 0.5, p99: 0.5 });
  });

  test("a level of repeated waves: per-wave aggregate median with min–max, strands pooled", () => {
    const wave = (k: number, last: number) => [result({ i: k, t_start: 0, t_first: 100, t_last: last, completion_tokens: 101 })];
    // three ×1 waves at 100 / 50 / 40 tok/s of decode
    const ws = summarizeLevel([wave(0, 1100), wave(1, 2100), wave(2, 2600)]);
    expect(ws.samples).toBe(3);
    expect(ws.per_stream_median_tok_s).toBe(50);
    expect(ws.per_stream_range).toEqual([40, 100]);
    expect(ws.aggregate_tok_s).toBe(Math.round((101 / 2.1) * 100) / 100);
    expect(ws.aggregate_range).toEqual([Math.round((101 / 2.6) * 100) / 100, Math.round((101 / 1.1) * 100) / 100]);
    expect(ws.aggregate_steady_tok_s).toBe(50);
  });

  test("envelope: arms, full_arms with per_request, headline, comparability summary", () => {
    const c1 = [result({ i: 0, token_times_ms: [100, 120], token_counts: [1, 5] })];
    const c4 = [0, 1, 2, 3].map((i) => result({ i: i + 1, t_start: 0, t_first: 200, t_last: 3200, completion_tokens: 151, wave: 0 }));
    const env = buildEnvelope({
      kind: "decode",
      model: "m",
      workload: { ...WORKLOAD, levels: [1, 4], max_tokens: 256 },
      arms: [
        { key: { concurrency: 1 }, ws: summarizeWave(c1), results: c1 },
        { key: { concurrency: 4 }, ws: summarizeWave(c4), results: c4, ctx: { foreign_max: 0, server: null } },
      ],
    });
    expect(env.source).toBe("controller-streams");
    expect(env.metrics.arms.map((a) => a.concurrency)).toEqual([1, 4]);
    expect(env.metrics.arms[0].decode_tok_per_s_median).toBe(50);
    expect(env.metrics.arms[1].aggregate_tok_per_s).toBe(Math.round((604 / 3.2) * 100) / 100);
    expect(env.metrics.arms[1].foreign_max).toBe(0);
    expect(env.metrics.full_arms[1].per_request).toHaveLength(4);
    expect(env.metrics.full_arms[1].per_request[0].wave).toBe(0);
    expect(env.metrics.full_arms[0].per_request[0]).toMatchObject({ i: 0, ttft_s: 0.1, decode_s: 2, tok_per_s: 50, token_times_ms: [100, 120], token_counts: [1, 5] });
    expect(env.metrics.headline).toEqual({
      decode_tok_per_s_median_c1: 50,
      aggregate_peak_tok_per_s: env.metrics.arms[1].aggregate_tok_per_s,
      aggregate_peak_concurrency: 4,
      ttft_p50_s_c1: 0.1,
    });
    expect(env.summary).toEqual({ ...env.metrics.headline, pack: "prose", max_tokens: 256, serve_fingerprint: "fp1", energy_j_per_token: null });
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
      workload: { ...WORKLOAD, sizes: [8192, 32768, 262144], max_tokens: 1 },
      arms,
    });
    expect(env.metrics.arms[0].prefill_tok_per_s).toBe(8192);
    expect(env.metrics.arms[1].prefill_tok_per_s).toBeCloseTo(6553.6, 1);
    expect(env.metrics.arms[2].skipped).toContain("max_model_len");
    expect(headline("prefill", env.metrics.arms).prefill_tok_per_s_sustained).toBeCloseTo(6553.6, 1);
  });
});
