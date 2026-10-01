import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { Database } from "bun:sqlite";
import { getDb, migrate } from "../db/schema";
import { getSettings } from "./settings";
import {
  counterDelta,
  getUsageSummary,
  parseEngineCounters,
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

describe("parseEngineCounters", () => {
  test("sums the token and request counters per model and reads the process start", () => {
    const c = parseEngineCounters(METRICS);
    expect(c.startTime).toBe(1790809832.95);
    expect(c.models.get(M)).toEqual({ prompt: 1397, completion: 4829, requests: 36 });
    expect(c.models.size).toBe(1);
  });

  test("sums across data-parallel engines", () => {
    const c = parseEngineCounters(
      'vllm:generation_tokens_total{engine="0",model_name="m"} 10\nvllm:generation_tokens_total{engine="1",model_name="m"} 5\n',
    );
    expect(c.models.get("m")?.completion).toBe(15);
  });

  test("SGLang (--enable-metrics) counters, per model", () => {
    const c = parseEngineCounters(`# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="Qwen/Qwen3-8B",tp_rank="0"} 812.0
sglang:generation_tokens_total{model_name="Qwen/Qwen3-8B",tp_rank="0"} 2048.0
sglang:num_requests_total{model_name="Qwen/Qwen3-8B",tp_rank="0"} 7.0
sglang:cached_tokens_total{model_name="Qwen/Qwen3-8B",tp_rank="0"} 300.0
sglang:num_running_reqs{model_name="Qwen/Qwen3-8B",tp_rank="0"} 1.0
`);
    expect(c.models.get("Qwen/Qwen3-8B")).toEqual({ prompt: 812, completion: 2048, requests: 7 });
    expect(c.startTime).toBeNull();
  });

  test("llama.cpp (--metrics) counters carry no model label: the caller names it", () => {
    const c = parseEngineCounters(
      `# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 120
llamacpp:tokens_predicted_total 345
llamacpp:prompt_seconds_total 1.5
llamacpp:n_decode_total 345
llamacpp:requests_processing 0
`,
      "qwen3-8b-q4_k_m.gguf",
    );
    expect(c.models.get("qwen3-8b-q4_k_m.gguf")).toEqual({ prompt: 120, completion: 345, requests: 0 });
    expect(c.models.size).toBe(1);
  });

  test("a page without engine counters yields nothing", () => {
    expect(parseEngineCounters("process_cpu_seconds_total 9\nhttp_requests_total 3\n").models.size).toBe(0);
    expect(parseEngineCounters("").models.size).toBe(0);
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

describe("recordCounters keyed by process, not by URL", () => {
  test("the same engine reached under an edited backend URL is not counted again", () => {
    recordCounters("http://127.0.0.1:8000", counters(100, 400, 4), new Date("2026-10-01T10:00:00Z"));
    // Configure: 127.0.0.1 → localhost; same process (same start time).
    recordCounters("http://localhost:8000", counters(100, 400, 4), new Date("2026-10-01T10:00:15Z"));
    recordCounters("http://localhost:8000", counters(110, 420, 5), new Date("2026-10-01T10:00:30Z"));
    // …and back again.
    recordCounters("http://127.0.0.1:8000", counters(130, 450, 6), new Date("2026-10-01T10:00:45Z"));
    const u = getUsageSummary();
    expect(u.lifetimePrompt).toBe(130);
    expect(u.lifetimeCompletion).toBe(450);
    expect(u.topModels).toEqual([{ model: M, tokens: 580, calls: 6 }]);
    const rows = getDb().query("SELECT COUNT(*) AS n FROM usage_counters").get() as { n: number };
    expect(rows.n).toBe(1);
  });

  test("a new process at a new URL still counts from zero", () => {
    recordCounters("http://127.0.0.1:8000", counters(100, 400, 4, 100), new Date("2026-10-01T10:00:00Z"));
    recordCounters("http://localhost:8000", counters(5, 9, 1, 200), new Date("2026-10-01T10:00:15Z"));
    expect(getUsageSummary().lifetimePrompt).toBe(105);
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

  test("llama.cpp usage is attributed to the model llama-server lists", async () => {
    const { vllm, llamacpp } = getSettings().backends;
    const lc = llamacpp.url.replace(/\/$/, "").replace(/\/v1$/, "");
    globalThis.fetch = (async (url: string) => {
      const u = String(url);
      if (u === `${lc}/metrics`) return new Response("llamacpp:prompt_tokens_total 10\nllamacpp:tokens_predicted_total 32\n");
      if (u === `${lc}/v1/models`) return Response.json({ object: "list", data: [{ id: "gemma-3-4b-q8.gguf" }] });
      throw new Error(`ECONNREFUSED ${u} (vllm at ${vllm.url})`);
    }) as unknown as typeof fetch;
    await sampleUsage();
    expect(getUsageSummary().topModels).toEqual([{ model: "gemma-3-4b-q8.gguf", tokens: 42, calls: 0 }]);
  });
});

describe("legacy usage_events", () => {
  test("are folded into per-minute buckets once, when usage_minutes is first created", () => {
    const db = new Database(":memory:");
    db.exec(`CREATE TABLE usage_events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, model TEXT NOT NULL,
      prompt_tokens INTEGER NOT NULL DEFAULT 0, completion_tokens INTEGER NOT NULL DEFAULT 0, session_id TEXT,
      source TEXT NOT NULL DEFAULT 'proxy')`);
    const add = db.query("INSERT INTO usage_events (ts, model, prompt_tokens, completion_tokens) VALUES (?, ?, ?, ?)");
    add.run("2026-09-05T17:47:04.074Z", M, 10, 4);
    add.run("2026-09-05T17:47:30.000Z", M, 6, 2);
    add.run("2026-08-14T14:36:49.278Z", "ci-test-model", 9, 9);
    migrate(db);
    migrate(db); // a restart must not fold them again
    expect(db.query("SELECT minute, model, prompt_tokens, completion_tokens, requests FROM usage_minutes").all()).toEqual([
      { minute: "2026-09-05T17:47", model: M, prompt_tokens: 16, completion_tokens: 6, requests: 2 },
    ]);
  });
});
