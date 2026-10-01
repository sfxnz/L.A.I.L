import type { UsageSummary } from "@lail/shared";
import { getDb } from "../db/schema";
import { getSettings } from "./settings";

/**
 * Token usage, metered from the engine's own Prometheus counters. Hermes and every
 * other client talk to :8000 directly, so only the engine sees all traffic; the
 * controller never counts requests itself.
 *
 * Each sample diffs the prompt / generation token counters and the finished-request
 * counter (summed over engines / finish reasons, per model) against the last reading
 * persisted for that backend + model, and adds the delta to a per-minute bucket. A
 * new engine process (process_start_time_seconds changed, or any counter went
 * backwards) starts from zero, so its whole counter is the delta.
 *
 * vLLM and TensorFold always export these; SGLang needs --enable-metrics and
 * llama-server needs --metrics (L.A.I.L launches both with it; llama.cpp has no request
 * counter, so its calls stay 0).
 */

export type Totals = { prompt: number; completion: number; requests: number };
export type EngineCounters = { startTime: number | null; models: Map<string, Totals> };
export type LastReading = Totals & { startTime: number | null };

const SERIES: Record<string, keyof Totals> = {
  "vllm:prompt_tokens_total": "prompt",
  "vllm:generation_tokens_total": "completion",
  "vllm:request_success_total": "requests",
  "sglang:prompt_tokens_total": "prompt",
  "sglang:generation_tokens_total": "completion",
  "sglang:num_requests_total": "requests",
  "llamacpp:prompt_tokens_total": "prompt",
  "llamacpp:tokens_predicted_total": "completion",
  // TensorFold counts a request's tokens when it finishes; the latency histogram's count is
  // its finished requests.
  "tensorfold:prompt_tokens_total": "prompt",
  "tensorfold:generation_tokens_total": "completion",
  "tensorfold:request_latency_seconds_count": "requests",
};

export const USAGE_SAMPLE_MS = 15_000;

/** `unlabeled` names the model for series without a model_name label (llama.cpp, TensorFold). */
export function parseEngineCounters(text: string, unlabeled = "unknown"): EngineCounters {
  const models = new Map<string, Totals>();
  let startTime: number | null = null;
  for (const line of text.split("\n")) {
    if (!line || line.startsWith("#")) continue;
    const m = /^([a-zA-Z_:][\w:]*)(?:\{([^}]*)\})?\s+(\S+)/.exec(line);
    if (!m) continue;
    const value = Number(m[3]);
    if (!Number.isFinite(value)) continue;
    if (m[1] === "process_start_time_seconds") {
      startTime = value;
      continue;
    }
    const field = SERIES[m[1]];
    if (!field) continue;
    const model = /model_name="([^"]*)"/.exec(m[2] || "")?.[1] || unlabeled;
    const t = models.get(model) ?? { prompt: 0, completion: 0, requests: 0 };
    t[field] += value;
    models.set(model, t);
  }
  return { startTime, models };
}

export function counterDelta(prev: LastReading | null, cur: Totals, startTime: number | null): Totals {
  const restarted =
    !prev ||
    (startTime !== null && prev.startTime !== null && startTime !== prev.startTime) ||
    cur.prompt < prev.prompt ||
    cur.completion < prev.completion ||
    cur.requests < prev.requests;
  if (restarted) return { ...cur };
  return {
    prompt: cur.prompt - prev.prompt,
    completion: cur.completion - prev.completion,
    requests: cur.requests - prev.requests,
  };
}

/** Fold one /metrics reading for `backend` into usage_minutes. */
export function recordCounters(backend: string, counters: EngineCounters, now = new Date()): void {
  const db = getDb();
  const minute = now.toISOString().slice(0, 16);
  const readPrev = db.query(
    "SELECT prompt, completion, requests, start_time AS startTime FROM usage_counters WHERE backend = ? AND model = ?",
  );
  // The same process reached under another URL (Configure edited 127.0.0.1 →
  // localhost, a LAN IP…): same model and start time. Its reading moves to the new
  // URL instead of the whole lifetime counter being counted again.
  const readSameProcess = db.query(
    `SELECT backend, prompt, completion, requests, start_time AS startTime FROM usage_counters
     WHERE model = ? AND start_time = ? AND backend != ?`,
  );
  const dropPrev = db.query("DELETE FROM usage_counters WHERE backend = ? AND model = ?");
  const writePrev = db.query(
    `INSERT INTO usage_counters (backend, model, prompt, completion, requests, start_time)
     VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT(backend, model) DO UPDATE SET prompt = excluded.prompt,
       completion = excluded.completion, requests = excluded.requests, start_time = excluded.start_time`,
  );
  const addMinute = db.query(
    `INSERT INTO usage_minutes (minute, model, prompt_tokens, completion_tokens, requests)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(minute, model) DO UPDATE SET
       prompt_tokens = prompt_tokens + excluded.prompt_tokens,
       completion_tokens = completion_tokens + excluded.completion_tokens,
       requests = requests + excluded.requests`,
  );
  db.transaction(() => {
    for (const [model, cur] of counters.models) {
      let prev = readPrev.get(backend, model) as LastReading | null;
      if (counters.startTime !== null && prev?.startTime !== counters.startTime) {
        const moved = readSameProcess.get(model, counters.startTime, backend) as
          | (LastReading & { backend: string })
          | null;
        if (moved) {
          prev = moved;
          dropPrev.run(moved.backend, model);
        }
      }
      const d = counterDelta(prev, cur, counters.startTime);
      if (d.prompt > 0 || d.completion > 0 || d.requests > 0) {
        addMinute.run(minute, model, Math.round(d.prompt), Math.round(d.completion), Math.round(d.requests));
      }
      writePrev.run(backend, model, cur.prompt, cur.completion, cur.requests, counters.startTime);
    }
  })();
}

/** The first id an OpenAI-compatible server lists, for engines whose counters carry no model label. */
async function servedModelId(base: string): Promise<string | null> {
  try {
    const r = await fetch(`${base}/v1/models`, { signal: AbortSignal.timeout(3000) });
    const body = (await r.json()) as { data?: Array<{ id?: string }> };
    return body.data?.[0]?.id || null;
  } catch {
    return null;
  }
}

/** One pass over every enabled backend's /metrics. Unreachable or unmetered backends are skipped. */
export async function sampleUsage(): Promise<void> {
  for (const [kind, b] of Object.entries(getSettings().backends)) {
    if (!b.enabled) continue;
    const base = b.url.replace(/\/$/, "").replace(/\/v1$/, "");
    try {
      const r = await fetch(`${base}/metrics`, { signal: AbortSignal.timeout(3000) });
      if (!r.ok) continue;
      const text = await r.text();
      // llama.cpp and TensorFold series carry no model label: name them by what is served.
      const unlabeled = /^(llamacpp|tensorfold):/m.test(text) ? ((await servedModelId(base)) ?? kind) : kind;
      const counters = parseEngineCounters(text, unlabeled);
      if (counters.models.size) recordCounters(base, counters);
    } catch {
      /* backend down: nothing served, nothing to meter */
    }
  }
}

export function startUsageMeter(): () => void {
  void sampleUsage();
  const t = setInterval(() => void sampleUsage(), USAGE_SAMPLE_MS);
  return () => clearInterval(t);
}

export function getUsageSummary(): UsageSummary {
  const db = getDb();
  const totals = db
    .query(
      `SELECT COALESCE(SUM(prompt_tokens),0) as prompt, COALESCE(SUM(completion_tokens),0) as completion
       FROM usage_minutes`,
    )
    .get() as { prompt: number; completion: number };

  const daily = db
    .query(
      `SELECT substr(minute,1,10) as date,
              SUM(prompt_tokens) as prompt,
              SUM(completion_tokens) as completion
       FROM usage_minutes
       GROUP BY substr(minute,1,10)
       ORDER BY date DESC
       LIMIT 90`,
    )
    .all() as Array<{ date: string; prompt: number; completion: number }>;

  const dailyAsc = [...daily].reverse();

  const topModels = db
    .query(
      `SELECT model,
              SUM(prompt_tokens + completion_tokens) as tokens,
              SUM(requests) as calls
       FROM usage_minutes
       GROUP BY model
       ORDER BY tokens DESC
       LIMIT 10`,
    )
    .all() as Array<{ model: string; tokens: number; calls: number }>;

  const lifetimePrompt = Number(totals.prompt) || 0;
  const lifetimeCompletion = Number(totals.completion) || 0;

  return {
    lifetimeTokens: lifetimePrompt + lifetimeCompletion,
    lifetimePrompt,
    lifetimeCompletion,
    heatmap: dailyAsc.map((d) => ({ date: d.date, tokens: Number(d.prompt) + Number(d.completion) })),
    daily: dailyAsc.map((d) => ({
      date: d.date,
      prompt: Number(d.prompt),
      completion: Number(d.completion),
    })),
    mix: { prompt: lifetimePrompt, completion: lifetimeCompletion },
    topModels: topModels.map((m) => ({
      model: m.model,
      tokens: Number(m.tokens),
      calls: Number(m.calls),
    })),
  };
}
