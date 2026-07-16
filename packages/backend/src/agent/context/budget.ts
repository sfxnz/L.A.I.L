import type { ContextChunk } from "./types";

const ELLIPSIS = "…";

/** Head-tail truncate so total length is at most maxChars (including ellipsis). */
export function truncateBody(body: string, maxChars: number): string {
  if (maxChars <= 0) return "";
  if (body.length <= maxChars) return body;
  if (maxChars <= ELLIPSIS.length) return body.slice(0, maxChars);
  const keep = maxChars - ELLIPSIS.length;
  const head = Math.ceil(keep / 2);
  const tail = Math.floor(keep / 2);
  return body.slice(0, head) + ELLIPSIS + body.slice(body.length - tail);
}

export type BudgetResult = {
  chunks: ContextChunk[];
  truncated: boolean;
  droppedLabels: string[];
};

/**
 * Sort by priority ascending (lower = higher priority), greedily take chunks
 * until budget; partially include last chunk via head-tail truncate.
 */
export function applyBudget(
  chunks: ContextChunk[],
  budgetChars: number,
): BudgetResult {
  const sorted = [...chunks].sort((a, b) => a.priority - b.priority);
  const kept: ContextChunk[] = [];
  const droppedLabels: string[] = [];
  let used = 0;
  let truncated = false;

  for (const chunk of sorted) {
    const remaining = budgetChars - used;
    if (remaining <= 0) {
      truncated = true;
      droppedLabels.push(chunk.label);
      continue;
    }
    if (chunk.body.length <= remaining) {
      kept.push(chunk);
      used += chunk.body.length;
      continue;
    }
    // Partial last chunk
    const body = truncateBody(chunk.body, remaining);
    kept.push({ ...chunk, body });
    used += body.length;
    truncated = true;
    // Any further chunks are dropped
  }

  // Labels of chunks that never fit after partial fill already recorded;
  // if we truncated a body, still mark truncated (already set).
  return { chunks: kept, truncated, droppedLabels };
}
