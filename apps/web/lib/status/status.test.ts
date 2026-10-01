/** Status card maths: quant parsing, the KV capacity forecast, uptime, the DNA strand slices. */
import { describe, expect, test } from "bun:test";
import { FORECAST_SEQ_TOKENS, fmtKvPct, fmtTokensK, forecastLine, kvForecast } from "./forecast";
import { fmtAgo, fmtUptime } from "./format";
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
  test("fmtKvPct: one decimal below 10 % so small real use is visible", () => {
    expect(fmtKvPct(0)).toBe("0%");
    expect(fmtKvPct(0.84)).toBe("0.8%");
    // real but tiny use (~680 tokens of a 1.69M pool) never reads as zero
    expect(fmtKvPct(0.04)).toBe("<0.1%");
    expect(fmtKvPct(0.05)).toBe("0.1%");
    expect(fmtKvPct(9.94)).toBe("9.9%");
    // the branch follows the rounded value: 9.96 is "10%", not "10.0%"
    expect(fmtKvPct(9.96)).toBe("10%");
    expect(fmtKvPct(42.37)).toBe("42%");
    expect(fmtKvPct(100)).toBe("100%");
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
  test("fmtAgo never reads \"now ago\"", () => {
    expect(fmtAgo(0.4)).toBe("just now");
    expect(fmtAgo(3.2)).toBe("3 s ago");
    expect(fmtAgo(null)).toBe("");
  });
});
