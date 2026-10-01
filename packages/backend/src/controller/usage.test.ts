import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { getDb } from "../db/schema";
import { getSettings } from "./settings";
import {
  counterDelta,
  getUsageSummary,
  parseVllmCounters,
  recordCounters,
  sampleUsage,
  type EngineCounters,
} from "./usage";

/** Shape of a live vLLM 0.30 /metrics page (TP=2, MTP): two engines would add per label set. */
const METRICS = `# HELP process_start_time_seconds Start time of the process since unix epoch in seconds.
# TYPE process_start_time_seconds gauge
process_start_time_seconds 1.79080983295e+09
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine="0",model_name="nvidia/Qwen3.8-Flash-Next-NVFP4"} 1397.0
vllm:generation_tokens_total{engine="0",model_name="nvidia/Qwen3.8-Flash-Next-NVFP4"} 4829.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="nvidia/Qwen3.8-Flash-Next-NVFP4"} 15.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="nvidia/Qwen3.8-Flash-Next-NVFP4"} 21.0
vllm:request_success_total{engine="0",finished_reason="abort",model_name="nvidia/Qwen3.8-Flash-Next-NVFP4"} 0.0
vllm:prompt_tokens_created{engine="0",model_name="nvidia/Qwen3.8-Flash-Next-NVFP4"} 1.79e+09
vllm:num_requests_running{engine="0",model_name="nvidia/Qwen3.8-Flash-Next-NVFP4"} 1.0
`;
const M = "nvidia/Qwen3.8-Flash-Next-NVFP4";

function counters(prompt: number, completion: number, requests: number, startTime = 100): EngineCounters {
  return { startTime, models: new Map([[M, { prompt, completion, requests }]]) };
}

beforeEach(() => {
  getDb().exec("DELETE FROM usage_minutes; DELETE FROM usage_counters;");
});

describe("parseVllmCounters", () => {
  test("sums the token and request counters per model and reads the process start", () => {
    const c = parseVllmCounters(METRICS);
    expect(c.startTime).toBe(1790809832.95);
    expect(c.models.get(M)).toEqual({ prompt: 1397, completion: 4829, requests: 36 });
    expect(c.models.size).toBe(1);
  });

  test("sums across data-parallel engines", () => {
    const c = parseVllmCounters(
      'vllm:generation_tokens_total{engine="0",model_name="m"} 10\nvllm:generation_tokens_total{engine="1",model_name="m"} 5\n',
    );
    expect(c.models.get("m")?.completion).toBe(15);
  });

  test("a non-vLLM page yields nothing", () => {
    expect(parseVllmCounters("llamacpp:tokens_predicted_total 9\n").models.size).toBe(0);
    expect(parseVllmCounters("").models.size).toBe(0);
  });
});

describe("counterDelta", () => {
  const prev = { prompt: 100, completion: 400, requests: 4, startTime: 100 };
  test("same process: the difference", () => {
    expect(counterDelta(prev, { prompt: 130, completion: 464, requests: 5 }, 100)).toEqual({
      prompt: 30,
      completion: 64,
      requests: 1,
    });
  });
  test("first sight of a process counts its whole counter", () => {
    expect(counterDelta(null, { prompt: 7, completion: 9, requests: 1 }, 100)).toEqual({
      prompt: 7,
      completion: 9,
      requests: 1,
    });
  });
  test("engine restart (new start time) counts from zero even when counters already passed the old reading", () => {
    expect(counterDelta(prev, { prompt: 500, completion: 900, requests: 9 }, 200)).toEqual({
      prompt: 500,
      completion: 900,
      requests: 9,
    });
  });
  test("a counter going backwards is a reset even without a start time", () => {
    const p = { ...prev, startTime: null };
    expect(counterDelta(p, { prompt: 20, completion: 30, requests: 1 }, null)).toEqual({
      prompt: 20,
      completion: 30,
      requests: 1,
    });
  });
});

describe("recordCounters → getUsageSummary", () => {
  test("accumulates deltas, never double-counts, survives an engine restart", () => {
    const t0 = new Date("2026-10-01T10:00:05Z");
    recordCounters("http://127.0.0.1:8000", counters(100, 400, 4), t0);
    recordCounters("http://127.0.0.1:8000", counters(100, 400, 4), new Date("2026-10-01T10:00:20Z"));
    recordCounters("http://127.0.0.1:8000", counters(130, 464, 5), new Date("2026-10-01T10:01:05Z"));
    // Restart: counters back near zero, new process start.
    recordCounters("http://127.0.0.1:8000", counters(10, 50, 1, 999), new Date("2026-10-02T09:00:00Z"));

    const u = getUsageSummary();
    expect(u.lifetimePrompt).toBe(100 + 30 + 10);
    expect(u.lifetimeCompletion).toBe(400 + 64 + 50);
    expect(u.lifetimeTokens).toBe(140 + 514);
    expect(u.daily).toEqual([
      { date: "2026-10-01", prompt: 130, completion: 464 },
      { date: "2026-10-02", prompt: 10, completion: 50 },
    ]);
    expect(u.topModels).toEqual([{ model: M, tokens: 654, calls: 6 }]);
    const minutes = getDb().query("SELECT COUNT(*) AS n FROM usage_minutes").get() as { n: number };
    expect(minutes.n).toBe(3); // the idle sample adds no row
  });
});

describe("sampleUsage", () => {
  const origFetch = globalThis.fetch;
  afterEach(() => {
    globalThis.fetch = origFetch;
  });

  test("scrapes each enabled backend's /metrics and skips the ones that are down", async () => {
    const { vllm, llamacpp } = getSettings().backends;
    const vllmMetrics = `${vllm.url.replace(/\/$/, "")}/metrics`;
    const seen: string[] = [];
    globalThis.fetch = (async (url: string) => {
      seen.push(String(url));
      if (String(url) === vllmMetrics) return new Response(METRICS);
      throw new Error("ECONNREFUSED");
    }) as unknown as typeof fetch;
    await sampleUsage();
    expect(seen).toEqual([vllmMetrics, `${llamacpp.url.replace(/\/$/, "")}/metrics`]);
    expect(getUsageSummary().lifetimeCompletion).toBe(4829);
  });
});
