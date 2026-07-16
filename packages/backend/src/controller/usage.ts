import type { UsageSummary } from "@lail/shared";
import { getDb } from "../db/schema";

export function recordUsage(opts: {
  model: string;
  prompt: number;
  completion: number;
  sessionId?: string | null;
  source?: string;
}) {
  getDb()
    .query(
      `INSERT INTO usage_events (ts, model, prompt_tokens, completion_tokens, session_id, source)
       VALUES (?, ?, ?, ?, ?, ?)`,
    )
    .run(
      new Date().toISOString(),
      opts.model || "unknown",
      opts.prompt || 0,
      opts.completion || 0,
      opts.sessionId || null,
      opts.source || "proxy",
    );
}

export function getUsageSummary(): UsageSummary {
  const db = getDb();
  const totals = db
    .query(
      `SELECT
         COALESCE(SUM(prompt_tokens),0) as prompt,
         COALESCE(SUM(completion_tokens),0) as completion
       FROM usage_events`,
    )
    .get() as { prompt: number; completion: number };

  const daily = db
    .query(
      `SELECT substr(ts,1,10) as date,
              SUM(prompt_tokens) as prompt,
              SUM(completion_tokens) as completion
       FROM usage_events
       GROUP BY substr(ts,1,10)
       ORDER BY date DESC
       LIMIT 90`,
    )
    .all() as Array<{ date: string; prompt: number; completion: number }>;

  const dailyAsc = [...daily].reverse();
  const heatmap = dailyAsc.map((d) => ({
    date: d.date,
    tokens: Number(d.prompt) + Number(d.completion),
  }));

  const topModels = db
    .query(
      `SELECT model,
              SUM(prompt_tokens + completion_tokens) as tokens,
              COUNT(*) as calls
       FROM usage_events
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
    heatmap,
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
