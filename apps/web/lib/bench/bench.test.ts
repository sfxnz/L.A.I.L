/**
 * Bench helpers — hand-computed cases. Interpretation/knee, ETA maths, Markdown
 * export shape, preset keys, result shaping from live events and envelopes.
 */
import { describe, expect, test } from "bun:test";
import {
  ETA_PAD,
  doublingRatio,
  etaDecodeMs,
  etaPrefillMs,
  fmtEta,
  predictPerStream,
  reviseEta,
} from "./eta";
import { benchMarkdown, decodeMarkdown, prefillMarkdown, quantFromModelId } from "./export";
import { fmtDuration, fmtMs, fmtSize, fmtTokS } from "./format";
import { findKnee, interpretDecode, interpretPrefill, referenceGain, spread, suggestLevels } from "./interpret";
import { previousComparable } from "./last-sync";
import { appendHardware } from "./live";
import type { RunRow } from "../api";
import {
  CONCURRENCY_LEVELS,
  LEVEL_PRESETS,
  decodeConfigFromQuery,
  decodeConfigToQuery,
  defaultDecodeConfig,
  presetForKey,
  sortConcurrencies,
  toggleLevel,
} from "./levels";
import {
  comparable,
  decodeArmsFromLevels,
  headlineFromResult,
  peakArm,
  resultFromEnvelope,
  steadyPeakArm,
  sustainedArm,
  type DecodeArm,
  type DecodeResult,
  type PrefillArm,
  type PrefillResult,
} from "./result";

// The sparkDash reference run: Prose · 512 tok · 1,2,4,6,8.
const ARM = (c: number, aggregate: number, perStream: number, ttft: number, ok = c): DecodeArm => ({
  samples: 1,
  foreignMax: 0,
  server: null,
  concurrency: c,
  aggregate,
  steady: aggregate,
  aggregateRange: null,
  perStream,
  perStreamRange: null,
  ttftP50: ttft,
  ttftP95: ttft * 1.3,
  ttftP99: ttft * 1.6,
  tpotMs: 1000 / perStream,
  ok,
  requests: c,
  errors: ok < c ? [`#${c - 1}: HTTP 500`] : [],
});
const SPARK = [ARM(1, 54.4, 54.4, 160), ARM(2, 86.5, 45.1, 426), ARM(4, 128.1, 34.2, 471), ARM(6, 169.7, 29.7, 490), ARM(8, 207, 26.7, 432)];

describe("levels and presets", () => {
  test("1–32 grid, keys 1–6 map to ×1 ×2 ×4 ×8 ×16 ×32", () => {
    expect(CONCURRENCY_LEVELS).toEqual(Array.from({ length: 32 }, (_, i) => i + 1));
    expect(LEVEL_PRESETS).toEqual([1, 2, 4, 8, 16, 32]);
    expect(["1", "2", "3", "4", "5", "6"].map(presetForKey)).toEqual([1, 2, 4, 8, 16, 32]);
    expect(presetForKey("7")).toBeNull();
    expect(presetForKey("0")).toBeNull();
    expect(presetForKey("a")).toBeNull();
  });

  test("selection runs ascending, deduped, clamped to 1..32; never empties", () => {
    expect(sortConcurrencies(new Set([16, 1, 4, 4, 0, 33]))).toEqual([1, 4, 16]);
    const one = new Set([4]);
    expect([...toggleLevel(one, 4)]).toEqual([4]);
    expect([...toggleLevel(one, 8)].sort()).toEqual([4, 8]);
    expect([...toggleLevel(new Set([4, 8]), 4)]).toEqual([8]);
  });

  test("config survives the query string round trip; junk falls back to defaults", () => {
    const cfg = { pack: "code", levels: [1, 2, 4], maxTokens: 128, floor: 25, sloMs: 800 };
    const q = decodeConfigToQuery(cfg);
    expect(q).toContain("levels=1%2C2%2C4");
    expect(decodeConfigFromQuery(new URLSearchParams(q))).toEqual(cfg);
    const junk = decodeConfigFromQuery(new URLSearchParams("pack=../x&levels=zz&tokens=-4"));
    expect(junk).toEqual(defaultDecodeConfig());
  });
});

describe("decode interpretation", () => {
  test("sparkDash run: 48 % efficient at ×8, best interactive ×4, no knee, not saturated", () => {
    const it = interpretDecode(SPARK, { floor: 20, sloMs: 500 });
    // 207 / (8 × 54.4) = 0.4756
    expect(it.efficiency).toBeCloseTo(0.4756, 3);
    expect(it.topLevel).toBe(8);
    expect(it.bestInteractive?.concurrency).toBe(8); // 26.7 ≥ 20 and 432 ms ≤ 500
    // reference gain ×1→×2 = 32.1 per stream; ×6→×8 gain = 37.3/2 = 18.65 ≥ 8.0 → no knee, not saturated
    expect(it.knee).toBeNull();
    expect(it.saturated).toBe(false);
    expect(it.suggest).toEqual([16, 32]);
    expect(suggestLevels(4)).toEqual([8, 16]);
    expect(suggestLevels(32)).toEqual([]);
    expect(it.sentence).toBe(
      "Scaling 48 % efficient at ×8; per-stream 26.7 tok/s (÷2.0 vs ×1); best interactive point ×8 — 26.7 tok/s per stream, TTFT p50 432 ms; saturation not reached — run ×16/×32.",
    );
  });

  test("stricter SLO moves the best interactive point down", () => {
    const it = interpretDecode(SPARK, { floor: 30, sloMs: 480 });
    expect(it.bestInteractive?.concurrency).toBe(4); // 34.2 ≥ 30 and 471 ≤ 480; ×6 fails both
    expect(it.knee).toBe(6); // per-stream 29.7 < floor 30
    expect(it.kneeReason).toBe("floor");
  });

  test("a perturbed ×1→×2 step (lost throughput) is not the yardstick — the best step is", () => {
    // Real run: ×2 came in under ×1 while the GPU was shared; ×2→×4 still gained 28 per added stream.
    const arms = [ARM(1, 32.1, 33.41, 156), ARM(2, 27.88, 23.95, 236), ARM(4, 83.97, 23.16, 355)];
    expect(referenceGain(arms)).toBeCloseTo((83.97 - 27.88) / 2, 6);
    const it = interpretDecode(arms, { floor: 20, sloMs: 500 });
    expect(it.saturated).toBe(false);
    expect(it.knee).toBeNull();
    expect(it.suggest).toEqual([8, 16]);
    expect(it.sentence).toContain("saturation not reached — run ×8/×16");
    // Two levels where the second lost throughput: that is saturation.
    expect(interpretDecode(arms.slice(0, 2), { floor: 20, sloMs: 500 }).saturated).toBe(true);
  });

  test("knee from marginal gain: ×8→×16 adds < 25 % of the ×1→×2 gain per stream", () => {
    const arms = [...SPARK, ARM(16, 220, 13.75, 900)];
    // reference 32.1/stream; ×8→×16 gain = 13/8 = 1.6 < 8.0
    const k = findKnee(arms, 5);
    expect(k).toMatchObject({ knee: 16, reason: "marginal" });
    const it = interpretDecode(arms, { floor: 5, sloMs: 5000 });
    expect(it.saturated).toBe(true);
    expect(it.suggest).toEqual([]);
    expect(it.sentence).toContain("knee at ×16");
    expect(it.sentence).toContain("saturated by ×16");
  });

  test("floor violated everywhere: no interactive point, knee at the first level", () => {
    const it = interpretDecode(SPARK, { floor: 60, sloMs: 500 });
    expect(it.bestInteractive).toBeNull();
    expect(it.knee).toBe(1);
    expect(it.kneeReason).toBe("floor");
    expect(it.sentence).toContain("no level holds 60 tok/s per stream under 500 ms TTFT");
  });

  test("single level and empty input degrade honestly", () => {
    const one = interpretDecode([SPARK[0]], { floor: 20, sloMs: 500 });
    expect(one.efficiency).toBeNull();
    expect(one.sentence).toBe("Single stream: 54.4 tok/s, TTFT p50 160 ms — add levels to see scaling; best interactive point ×1 — 54.4 tok/s per stream, TTFT p50 160 ms.");
    // ×1's sentence quotes the per-stream decode rate, not the wall-clock aggregate
    expect(interpretDecode([{ ...SPARK[0], aggregate: 50.1 }], { floor: 20, sloMs: 500 }).sentence).toStartWith("Single stream: 54.4 tok/s");
    expect(interpretDecode([], { floor: 20, sloMs: 500 }).sentence).toContain("nothing to interpret");
  });

  test("a gain drop inside the measured spread is noise, not a knee or saturation", () => {
    // ×8→×16 adds 1.6/stream (< 25 % of 32.1), but the waves of ×8 and ×16 scattered ±30
    const noisy = [...SPARK.slice(0, 4), { ...SPARK[4], aggregateRange: [177, 237] as [number, number] }, { ...ARM(16, 220, 13.75, 900), aggregateRange: [190, 250] as [number, number] }];
    expect(spread(noisy[4])).toBe(30);
    expect(findKnee(noisy, 5).knee).toBeNull();
    expect(interpretDecode(noisy, { floor: 5, sloMs: 5000 }).saturated).toBe(false);
    // the same drop measured tightly is a knee
    const tight = [...SPARK.slice(0, 4), { ...SPARK[4], aggregateRange: [205, 209] as [number, number] }, { ...ARM(16, 220, 13.75, 900), aggregateRange: [218, 222] as [number, number] }];
    expect(findKnee(tight, 5)).toMatchObject({ knee: 16, reason: "marginal" });
  });

  test("contended levels are named and kept out of the knee and saturation calls", () => {
    // ×16 lost throughput only because foreign requests shared the GPU
    const arms = [...SPARK, { ...ARM(16, 150, 9.4, 900), foreignMax: 6 }];
    const it = interpretDecode(arms, { floor: 5, sloMs: 5000 });
    expect(it.contended).toEqual([16]);
    expect(it.knee).toBeNull();
    expect(it.saturated).toBe(false);
    expect(it.sentence).toContain("×16 contended (foreign requests on the server)");
  });

  test("failed levels (ok = 0) are ignored by the maths but kept in the arms", () => {
    const arms = [SPARK[0], SPARK[1], ARM(4, 128.1, 34.2, 471, 0), SPARK[4]];
    const it = interpretDecode(arms, { floor: 20, sloMs: 500 });
    expect(it.topLevel).toBe(8);
    expect(peakArm(arms)?.concurrency).toBe(8);
  });

  test("the hero's peak is the decode-span aggregate; runs without it have none", () => {
    // ×8 has the higher wall-clock aggregate, ×6 the higher decode-span one: the hero
    // follows the live gauge's metric, not the wall-clock goodput.
    const arms = [{ ...SPARK[3], steady: 260 }, { ...SPARK[4], steady: 240 }];
    expect(peakArm(arms)?.concurrency).toBe(8);
    expect(steadyPeakArm(arms)?.concurrency).toBe(6);
    expect(steadyPeakArm(arms.map((a) => ({ ...a, steady: null })))).toBeNull();
  });
});

describe("prefill interpretation", () => {
  const P = (size: number, tokS: number, ttftMs: number): PrefillArm => ({
    samples: 1,
    foreignMax: 0,
    server: null,
    size,
    promptTokens: size + 38,
    prefillTokS: tokS,
    prefillRange: null,
    ttftMs,
    ok: 1,
    requests: 1,
    errors: [],
    skipped: null,
  });
  const REF = [P(8192, 2874.9, 2860), P(16384, 2962.1, 5540), P(32768, 2962.2, 11080), P(65536, 2893.4, 22660), P(131072, 2727.1, 48080)];

  test("sparkDash prefill: holds within 8 % to 128k; TTFT ×2.12 per doubling at 128k vs ×1.94 at 16k", () => {
    const it = interpretPrefill(REF);
    expect(it.peak?.size).toBe(32768);
    expect(it.sustained?.size).toBe(131072);
    expect(it.hold).toBeCloseTo(1 - 2727.1 / 2962.2, 4);
    expect(it.doublings[0]).toMatchObject({ size: 16384 });
    expect(it.doublings[0].ratio).toBeCloseTo(5540 / 2860, 3);
    expect(it.doublings.at(-1)?.ratio).toBeCloseTo(48080 / 22660, 3);
    expect(it.sentence).toBe(
      "Holds within 8 % of peak (2.96k tok/s at 32k) to 128k; TTFT grows ×2.12 per doubling at 128k vs ×1.94 at 16k.",
    );
  });

  test("a rate that never drops holds its peak, not 'within 0 %'", () => {
    const it = interpretPrefill([P(8192, 3255, 2516), P(16384, 3262, 5023)]);
    expect(it.hold).toBe(0);
    expect(it.sentence).toBe("Holds its peak (3.26k tok/s at 16k) to 16k; TTFT grows ×2.00 per doubling at 16k.");
  });

  test("skipped sizes are named, never dropped", () => {
    const skipped: PrefillArm = { ...P(262144, 0, 0), prefillTokS: null, ttftMs: null, ok: 0, requests: 0, skipped: "size 262144 ≥ max_model_len 262144" };
    const it = interpretPrefill([...REF, skipped]);
    expect(it.skipped).toHaveLength(1);
    expect(it.sentence.endsWith("; 256k skipped.")).toBe(true);
    expect(sustainedArm([...REF, skipped])?.size).toBe(131072);
  });
});

describe("ETA maths", () => {
  const measured = [
    { concurrency: 1, perStream: 54.4, ttftMs: 160 },
    { concurrency: 2, perStream: 45.1, ttftMs: 426 },
  ];

  test("per-stream prediction follows the power law through the last two levels, capped at ×1", () => {
    // β = −ln(45.1/54.4)/ln 2 = 0.2705 → ×4: 45.1 × 2^−0.2705 = 37.39
    expect(predictPerStream(measured, 4)).toBeCloseTo(37.39, 1);
    expect(predictPerStream(measured, 1)).toBeCloseTo(54.4, 3);
    expect(predictPerStream([measured[0]], 8)).toBe(54.4);
    expect(predictPerStream([], 8)).toBeNull();
  });

  test("remaining time sums the current wave and every level still to run, padded 12 %", () => {
    const eta = etaDecodeMs({
      levels: [1, 2, 4, 8],
      maxTokens: 512,
      measured,
      current: { concurrency: 4, tokens: [200, 180, 220, 190], rateTokS: 36, waitingFirstToken: 0 },
    });
    // current: (512−180)/36 = 9.22 s; ×8: 512/predict(8) + ttft; predict(8)= 45.1×4^−0.2705 = 31.0 → 16.52 s + 0.8×426×4 = 1.36 s
    const expected = (9222 + 16516 + 1363) * ETA_PAD;
    expect(eta).toBeGreaterThan(expected * 0.97);
    expect(eta).toBeLessThan(expected * 1.03);
    // nothing measured, no live rate → unknown
    expect(etaDecodeMs({ levels: [1, 2], maxTokens: 512, measured: [], current: { concurrency: 1, tokens: [3], rateTokS: null, waitingFirstToken: 1 } })).toBeNull();
    // everything done
    expect(etaDecodeMs({ levels: [1, 2], maxTokens: 512, measured, current: null })).toBe(0);
  });

  test("repeats: a level of concurrency c costs ⌈samples ÷ c⌉ waves, the current level its later waves", () => {
    const one = etaDecodeMs({ levels: [1, 2, 4], maxTokens: 512, measured: [measured[0]], current: { concurrency: 2, tokens: [512, 512], rateTokS: 45.1, waitingFirstToken: 0, wavesLeft: 1 }, samples: 3 })!;
    // current wave done; one more ×2 wave (512/45.1 s + 0.8×160×2 ms), then ×4: 1 wave at predict(4) = 54.4 (one measured level)
    const expected = ((512 / 45.1) * 1000 + 256 + ((512 / 54.4) * 1000 + 0.8 * 160 * 4)) * ETA_PAD;
    expect(one).toBeCloseTo(expected, -1);
  });

  test("revision falls freely and rises only past 25 %", () => {
    expect(reviseEta(60_000, 50_000)).toBe(50_000);
    expect(reviseEta(60_000, 70_000)).toBe(60_000);
    expect(reviseEta(60_000, 80_000)).toBe(80_000);
    expect(reviseEta(null, 12_000)).toBe(12_000);
    expect(reviseEta(12_000, null)).toBe(12_000);
    expect(fmtEta(47_000)).toBe("~45 s");
    expect(fmtEta(77_000)).toBe("~1 m 15 s");
    expect(fmtEta(1_000)).toBe("~5 s");
  });

  test("prefill ETA doubles per context doubling from the measured ratio", () => {
    const m = [
      { size: 8192, ttftMs: 2860 },
      { size: 16384, ttftMs: 5540 },
    ];
    expect(doublingRatio(m)).toBeCloseTo(5540 / 2860, 4);
    expect(doublingRatio([m[0]])).toBe(2);
    const eta = etaPrefillMs({ sizes: [8192, 16384, 32768, 65536], measured: m, skipped: new Set(), current: { size: 32768, elapsedMs: 4000 } });
    const r = 5540 / 2860;
    const expected = (5540 * r - 4000 + 5540 * r * r) * ETA_PAD;
    expect(eta).toBeCloseTo(expected, -2);
    expect(etaPrefillMs({ sizes: [8192, 16384, 262144], measured: m, skipped: new Set([262144]), current: null })).toBe(0);
    expect(etaPrefillMs({ sizes: [8192], measured: [], skipped: new Set(), current: { size: 8192, elapsedMs: 100 } })).toBeNull();
    // two requests per size: the current size's next request and both of every later size
    const two = etaPrefillMs({ sizes: [8192, 16384, 32768], measured: m.slice(0, 1).concat(m[1]), skipped: new Set(), current: null, samples: 2 });
    expect(two).toBeCloseTo(5540 * r * 2 * ETA_PAD, -2);
    const mid = etaPrefillMs({ sizes: [8192, 16384, 32768], measured: m, skipped: new Set(), current: { size: 32768, elapsedMs: 1000, requestsLeft: 1 }, samples: 2 });
    expect(mid).toBeCloseTo((5540 * r - 1000 + 5540 * r) * ETA_PAD, -2);
  });
});

describe("Markdown export", () => {
  const decode: DecodeResult = {
    kind: "decode",
    id: "r1",
    savedRunId: "r1",
    model: "nvidia/Qwen3.8-Flash-Next-NVFP4",
    pack: "prose",
    maxTokens: 512,
    createdAt: "2026-09-05T20:00:00Z",
    durationMs: 77_000,
    engine: "vllm 0.28.1",
    fingerprint: "8488b677",
    hardware: null,
    source: "history",
    levels: [1, 2, 4, 6, 8],
    arms: SPARK,
  };

  test("decode rows follow llama-bench order: model · engine · tg N @ cN · t/s (total) · t/s (req) · TTFT p50", () => {
    const md = decodeMarkdown(decode);
    const lines = md.trim().split("\n");
    expect(lines[0]).toBe("| model | engine | test | t/s (total) | t/s (req) | TTFT p50 |");
    expect(lines[1]).toBe("| --- | --- | --- | --- | --- | --- |");
    expect(lines[2]).toBe("| nvidia/Qwen3.8-Flash-Next-NVFP4 | vllm 0.28.1 · NVFP4 | tg 512 @ c1 | 54.4 | 54.4 | 160 ms |");
    expect(lines[6]).toBe("| nvidia/Qwen3.8-Flash-Next-NVFP4 | vllm 0.28.1 · NVFP4 | tg 512 @ c8 | 207.0 | 26.7 | 432 ms |");
    expect(lines).toHaveLength(7);
  });

  test("engine column drops out when nothing is known; failed strands are listed under the table", () => {
    const md = decodeMarkdown({ ...decode, model: "org/plain-model", engine: null, arms: [ARM(1, 50, 50, 100), ARM(2, 80, 40, 300, 1)] });
    expect(md.split("\n")[0]).toBe("| model | test | t/s (total) | t/s (req) | TTFT p50 |");
    expect(md).toContain("- c2: 1/2 streams ok — #1: HTTP 500");
  });

  test("prefill rows are pp <N> with tokens, prefill t/s and TTFT; skipped sizes say why", () => {
    const prefill: PrefillResult = {
      ...decode,
      kind: "prefill",
      sizes: [8192, 262144],
      arms: [
        { samples: 1, foreignMax: 0, server: null, size: 8192, promptTokens: 8230, prefillTokS: 2874.9, prefillRange: null, ttftMs: 2860, ok: 1, requests: 1, errors: [], skipped: null },
        { samples: null, foreignMax: null, server: null, size: 262144, promptTokens: null, prefillTokS: null, prefillRange: null, ttftMs: null, ok: 0, requests: 0, errors: [], skipped: "size 262144 ≥ max_model_len 262144" },
      ],
    };
    const lines = prefillMarkdown(prefill).trim().split("\n");
    expect(lines[0]).toBe("| model | engine | test | prompt tokens | prefill t/s | TTFT |");
    expect(lines[2]).toBe("| nvidia/Qwen3.8-Flash-Next-NVFP4 | vllm 0.28.1 · NVFP4 | pp 8192 | 8230 | 2874.9 | 2.86 s |");
    expect(lines[3]).toBe("| nvidia/Qwen3.8-Flash-Next-NVFP4 | vllm 0.28.1 · NVFP4 | pp 262144 | skipped: size 262144 ≥ max_model_len 262144 | — | — |");
    expect(benchMarkdown(prefill)).toBe(prefillMarkdown(prefill));
  });

  test("quant is read off the model id", () => {
    expect(quantFromModelId("nvidia/Qwen3.8-Flash-Next-NVFP4")).toBe("NVFP4");
    expect(quantFromModelId("unsloth/Qwen3-27B-Q4_K_M-GGUF")).toBe("Q4_K_M");
    expect(quantFromModelId("meta-llama/Llama-3-8B-Instruct")).toBeNull();
  });
});

describe("formatting", () => {
  test("units never mix inside one column", () => {
    expect(fmtTokS(54.4)).toBe("54.4");
    expect(fmtTokS(207)).toBe("207");
    expect(fmtTokS(2962.2)).toBe("2.96k");
    expect(fmtMs(160)).toBe("160 ms");
    expect(fmtMs(2860)).toBe("2.86 s");
    expect(fmtDuration(77_000)).toBe("1 m 17 s");
    expect(fmtDuration(48_000)).toBe("48 s");
    expect(fmtSize(8192)).toBe("8k");
    expect(fmtSize(262144)).toBe("256k");
  });
});

describe("result shaping", () => {
  test("live level rows become arms in ms", () => {
    const arms = decodeArmsFromLevels([
      { index: 0, concurrency: 1, aggregate_tok_s: 54.4, aggregate_steady_tok_s: 54.4, per_stream_median_tok_s: 54.4, ttft_p50_ms: 160, ttft_p95_ms: 200, ttft_p99_ms: 210, tpot_ms: 18.4, ok: 1, requests: 1, errors: [] },
      { index: 1, concurrency: 2, aggregate_tok_s: null, aggregate_steady_tok_s: null, per_stream_median_tok_s: null, ttft_p50_ms: null, ttft_p95_ms: null, ttft_p99_ms: null, tpot_ms: null, ok: 0, requests: 2, errors: ["#1: HTTP 500", "#2: HTTP 500"] },
    ]);
    expect(arms[0]).toMatchObject({ concurrency: 1, aggregate: 54.4, ttftP50: 160, tpotMs: 18.4 });
    expect(arms[1]).toMatchObject({ concurrency: 2, aggregate: null, ok: 0, requests: 2 });
    expect(arms[1].errors).toHaveLength(2);
  });

  const index = { run_id: "20260905T171607Z_acc454", created_at: "2026-09-05T17:16:07Z", kind: "decode", intent: "attach", model_id: "nvidia/Qwen3.8-Flash-Next-NVFP4", summary: {}, path: "x" };

  test("controller envelope (seconds) → result (ms), headline agrees with the arms", () => {
    const envelope = {
      kind: "decode",
      created_at: "2026-09-05T17:16:07Z",
      model: { id: "nvidia/Qwen3.8-Flash-Next-NVFP4" },
      engine: { name: "vllm", version: "0.28.1" },
      workload: { pack: "prose", levels: [1, 2], max_tokens: 512 },
      metrics: {
        arms: [
          { concurrency: 1, ok: 1, requests: 1, ttft_s: { p50: 0.16, p95: 0.2, p99: 0.21 }, aggregate_tok_per_s: 54.4, decode_tok_per_s_median: 54.4, tpot_s: 0.0184, errors: [] },
          { concurrency: 2, ok: 2, requests: 2, ttft_s: { p50: 0.426, p95: 0.5, p99: 0.51 }, aggregate_tok_per_s: 86.5, decode_tok_per_s_median: 45.1, tpot_s: 0.0222, errors: [] },
        ],
      },
    };
    const r = resultFromEnvelope(index, envelope);
    expect(r?.kind).toBe("decode");
    if (r?.kind !== "decode") throw new Error("kind");
    expect(r.pack).toBe("prose");
    expect(r.engine).toBe("vllm 0.28.1");
    expect(r.levels).toEqual([1, 2]);
    expect(r.arms[1]).toMatchObject({ concurrency: 2, aggregate: 86.5, ttftP50: 426, ttftP99: 510, tpotMs: 22.2 });
    expect(headlineFromResult(r)).toEqual({ c1: 54.4, peak: 86.5, peakAt: 2, sustained: null });
  });

  test("repeats, spread, contention, the server's view, fingerprint and hardware ride along from the envelope", () => {
    const envelope = {
      kind: "decode",
      model: { id: "nvidia/Qwen3.8-Flash-Next-NVFP4" },
      engine: { name: "vllm", version: "0.30.0", flags_fingerprint: "8488b677" },
      workload: { pack: "prose", levels: [1], samples: 3, max_tokens: 512, serve_fingerprint: "8488b677" },
      metrics: {
        arms: [
          {
            concurrency: 1,
            ok: 3,
            requests: 3,
            samples: 3,
            ttft_s: { p50: 0.16, p95: null, p99: null },
            aggregate_tok_per_s: 52.1,
            aggregate_steady_tok_per_s: 54.4,
            aggregate_range: [50.2, 53.0],
            decode_tok_per_s_median: 54.4,
            per_stream_range: [53.1, 55.0],
            foreign_max: 0,
            server: { spec_acceptance: 0.86, spec_tokens_per_step: 3.58, ttft_mean_ms: 150, prefill_tok_s: null, preemptions: 0 },
            tpot_s: 0.0184,
            errors: [],
          },
        ],
        hardware: { series: [], nodes: [{ id: "spark1", samples: 9, temp_max_c: 61, temp_mean_c: 57.2, power_mean_w: 88.1, available_min_gib: 14 }], energy_j: 2210, energy_j_per_token: 1.44 },
      },
    };
    const r = resultFromEnvelope({ ...index, summary: { pack: "prose" } }, envelope);
    if (r?.kind !== "decode") throw new Error("kind");
    expect(r.fingerprint).toBe("8488b677");
    expect(r.hardware).toEqual({ nodes: envelope.metrics.hardware.nodes, energy_j: 2210, energy_j_per_token: 1.44 });
    expect(r.arms[0]).toMatchObject({ samples: 3, steady: 54.4, aggregateRange: [50.2, 53.0], perStreamRange: [53.1, 55.0], foreignMax: 0, ttftP95: null });
    expect(r.arms[0].server?.spec_acceptance).toBe(0.86);
    expect(resultFromEnvelope(index, null)).toBeNull();
  });
});

describe("comparability", () => {
  const key = { model: "m", pack: "prose", maxTokens: 512, fingerprint: "fp" };
  test("like for like: model, pack, tokens/stream and serve fingerprint must all match", () => {
    expect(comparable(key, { ...key })).toBe(true);
    expect(comparable(key, { ...key, model: "other" })).toBe(false);
    expect(comparable(key, { ...key, pack: "code" })).toBe(false);
    expect(comparable(key, { ...key, maxTokens: 256 })).toBe(false);
    expect(comparable(key, { ...key, fingerprint: "fp2" })).toBe(false);
    // unknown fingerprint (runs before it was recorded) compares with nothing
    expect(comparable({ ...key, fingerprint: null }, { ...key, fingerprint: null })).toBe(false);
  });

  test("the ghost is the newest earlier comparable run, past any number of other models' runs", () => {
    const row = (id: string, created_at: string, over: { model?: string; max_tokens?: number; fp?: string } = {}): RunRow => ({
      run_id: id,
      created_at,
      kind: "decode",
      intent: null,
      model_id: over.model ?? "m",
      summary: { pack: "prose", max_tokens: over.max_tokens ?? 512, serve_fingerprint: over.fp ?? "fp" },
      path: "",
    });
    const cur = { id: "c", savedRunId: "c", model: "m", pack: "prose", maxTokens: 512, fingerprint: "fp", createdAt: "2026-09-30T12:00:00Z" };
    // a dozen newer-than-baseline runs of another model would have pushed it out of a 12-row window
    const others = Array.from({ length: 15 }, (_, k) => row(`o${k}`, `2026-09-29T${String(10 + (k % 10)).padStart(2, "0")}:0${k % 6}:00Z`, { model: "other" }));
    const rows = [
      row("c", "2026-09-30T12:00:00Z"),
      row("later", "2026-09-30T13:00:00Z"),
      row("a", "2026-09-29T23:00:00Z", { max_tokens: 256 }),
      row("b", "2026-09-29T22:00:00Z", { fp: "old" }),
      ...others,
      row("d", "2026-09-27T00:00:00Z"),
      row("e", "2026-09-26T00:00:00Z"),
    ];
    expect(previousComparable(rows, cur)?.run_id).toBe("d");
    expect(previousComparable(rows.filter((r) => r.run_id !== "d" && r.run_id !== "e"), cur)).toBeNull();
  });
});

describe("hardware strip sampler (server clock)", () => {
  const s = (t: number, power: number) => ({ t, power, util: 50, temp: 45, mem: 100, rails: null });
  const nodes = [{ id: "spark1", label: "spark1" }];

  test("appends only samples from the run start on, each once, keyed by the server's sampled_at", () => {
    const since = 10_000;
    let series = appendHardware([], { spark1: [s(9_000, 9), s(10_000, 20), s(11_000, 21)] }, nodes, since);
    expect(series[0].samples.map((x) => x.t)).toEqual([10_000, 11_000]);
    const same = appendHardware(series, { spark1: [s(10_000, 20), s(11_000, 21)] }, nodes, since);
    expect(same).toBe(series); // nothing new → same array, no re-render
    series = appendHardware(series, { spark1: [s(11_000, 21), s(12_000, 22)] }, nodes, since);
    expect(series[0].samples.map((x) => x.power)).toEqual([20, 21, 22]);
  });
});
