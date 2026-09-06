/** Status card maths: quant parsing, the KV capacity forecast, uptime, the DNA strand slices. */
import { describe, expect, test } from "bun:test";
import type { RunRow } from "../api";
import { toSlices, sliceColor, sliceHref } from "./dna";
import { FORECAST_SEQ_TOKENS, fmtTokensK, forecastLine, kvForecast, kvFraction, kvUsedTokens } from "./forecast";
import { fmtUptime } from "./format";
import { parseQuant } from "./quant";

describe("parseQuant", () => {
  test("reads the quant token from the model id, most specific first", () => {
    expect(parseQuant("nvidia/Qwen3.8-Flash-Next-NVFP4")).toBe("NVFP4");
    expect(parseQuant("nvidia/DeepSeek-V4-Flash-FP8")).toBe("FP8");
    expect(parseQuant("Qwen/Qwen3-32B-AWQ")).toBe("AWQ");
    expect(parseQuant("TheBloke/Llama-2-13B-GPTQ")).toBe("GPTQ");
    expect(parseQuant("bartowski/gemma-3-27b-it-GGUF")).toBe("GGUF");
    expect(parseQuant("org/model-w4a16")).toBe("INT4");
    expect(parseQuant("org/model-bf16")).toBe("BF16");
  });
  test("null when the id says nothing — never a guess", () => {
    expect(parseQuant("Qwen/Qwen3-32B")).toBeNull();
    expect(parseQuant(null)).toBeNull();
    expect(parseQuant("")).toBeNull();
  });
});

describe("KV capacity forecast", () => {
  test("fits = floor(capacity / 32k) by default", () => {
    const f = kvForecast(524_288)!;
    expect(f.fits).toBe(16);
    expect(f.seqTokens).toBe(FORECAST_SEQ_TOKENS);
    expect(forecastLine(f)).toBe("KV capacity 512k tokens · fits 16 × 32k");
    expect(kvForecast(100_000, 32_768)!.fits).toBe(3);
  });
  test("absent or non-positive capacity → null (the card shows Nil)", () => {
    expect(kvForecast(null)).toBeNull();
    expect(kvForecast(undefined)).toBeNull();
    expect(kvForecast(0)).toBeNull();
    expect(kvForecast(NaN)).toBeNull();
  });
  test("fmtTokensK", () => {
    expect(fmtTokensK(32_768)).toBe("32k");
    expect(fmtTokensK(131_072)).toBe("128k");
    expect(fmtTokensK(50_000)).toBe("50k");
    expect(fmtTokensK(1500)).toBe("1.5k");
    expect(fmtTokensK(800)).toBe("800");
    expect(fmtTokensK(2_100_000)).toBe("2.1M");
  });
  test("kv usage accepts a 0–1 fraction (vLLM) or 0–100 percent (C2)", () => {
    expect(kvFraction(0.11)).toBeCloseTo(0.11);
    expect(kvFraction(11)).toBeCloseTo(0.11);
    expect(kvFraction(140)).toBe(1);
    expect(kvUsedTokens(524_288, 0.5)).toBe(262_144);
    expect(kvUsedTokens(524_288, 50)).toBe(262_144);
    expect(kvUsedTokens(null, 0.5)).toBeNull();
    expect(kvUsedTokens(524_288, null)).toBeNull();
  });
});

describe("fmtUptime", () => {
  test("ladders s → m → h → d", () => {
    expect(fmtUptime(42)).toBe("42 s");
    expect(fmtUptime(95)).toBe("1 m 35 s");
    expect(fmtUptime(4523)).toBe("1 h 15 m");
    expect(fmtUptime(90_000)).toBe("1 d 1 h");
    expect(fmtUptime(null)).toBe("");
    expect(fmtUptime(-1)).toBe("");
  });
});

describe("DNA strand slices", () => {
  const row = (run_id: string, created_at: string, kind: string, summary: Record<string, unknown> = {}): RunRow => ({
    run_id,
    created_at,
    kind,
    intent: null,
    model_id: "nvidia/Qwen3.8-Flash-Next-NVFP4",
    summary,
    path: `${run_id}.json`,
  });

  test("kind → colour token and destination", () => {
    expect(sliceColor("decode")).toBe("bg-lab-line");
    expect(sliceColor("prefill")).toBe("bg-lab-line-2");
    expect(sliceColor("agentic_tool_eval")).toBe("bg-lab-chart-3");
    expect(sliceColor("perf_workflow")).toBe("bg-lab-muted");
    expect(sliceHref({ run_id: "d1", kind: "decode" })).toBe("/bench?run=d1");
    expect(sliceHref({ run_id: "p1", kind: "prefill" })).toBe("/bench?run=p1");
    expect(sliceHref({ run_id: "t1", kind: "agentic_tool_eval" })).toBe("/evals/tool/t1");
    expect(sliceHref({ run_id: "x", kind: "perf_workflow" })).toBe("/evals");
  });

  test("newest first, capped at 30, with a compact headline per kind", () => {
    const rows = Array.from({ length: 35 }, (_, i) =>
      row(`r${i}`, `2026-09-06T${String(i % 24).padStart(2, "0")}:${String(Math.floor(i / 24) * 10).padStart(2, "0")}:00Z`, "decode", {
        aggregate_peak_tok_per_s: 207.4,
        aggregate_peak_concurrency: 8,
      }),
    );
    const s = toSlices(rows);
    expect(s).toHaveLength(30);
    expect(s[0].createdAt >= s[1].createdAt).toBe(true);
    expect(s[0].headline).toBe("207 tok/s @ ×8");
    expect(toSlices([row("p", "2026-09-06T10:00:00Z", "prefill", { prefill_tok_per_s_sustained: 2874.9 })])[0].headline).toBe("2.9k tok/s sustained");
    expect(toSlices([row("t", "2026-09-06T10:00:00Z", "tool_eval", { final_score: 81.6 })])[0].headline).toBe("score 82");
    expect(toSlices([row("x", "2026-09-06T10:00:00Z", "other")])[0].headline).toBeNull();
  });
});
