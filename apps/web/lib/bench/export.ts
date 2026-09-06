/**
 * Take-away exports. Markdown follows the llama-bench / llama-benchy column
 * order the community pastes: model · engine · test (`tg 512 @ c8`) · t/s (total)
 * · t/s (req) · TTFT p50; prefill rows are `pp 8192`.
 */
import { fmtMs } from "./format";
import type { BenchResult, DecodeResult, PrefillResult } from "./result";

const QUANT_RE = /\b(nvfp4|fp8|fp4|int8|int4|awq|gptq|bnb|mxfp4|q[2-8]_(?:k_[sml]|k|0|1)|iq[1-4]_\w+|bf16|fp16)\b/i;

/** Quant token from a model id such as `nvidia/Qwen3.8-Flash-Next-NVFP4`; null when none is spelled out. */
export function quantFromModelId(id: string | null | undefined): string | null {
  if (!id) return null;
  const m = id.replace(/[-./]/g, " ").match(QUANT_RE);
  return m ? m[1].toUpperCase() : null;
}

/** Exports keep raw one-decimal numbers — no "2.9k" shorthand in a pasted table. */
function num1(v: number | null): string {
  return v === null || !Number.isFinite(v) ? "" : v.toFixed(1);
}

function cell(v: string): string {
  return v === "" ? "—" : v;
}

function table(header: string[], rows: string[][]): string {
  const line = (cells: string[]) => `| ${cells.map(cell).join(" | ")} |`;
  return [line(header), `|${header.map(() => " --- ").join("|")}|`, ...rows.map(line)].join("\n");
}

function engineColumn(r: BenchResult): string {
  const quant = quantFromModelId(r.model);
  const parts = [r.engine, quant].filter((s): s is string => !!s);
  return parts.join(" · ");
}

export function decodeMarkdown(r: DecodeResult): string {
  const engine = engineColumn(r);
  const header = ["model", ...(engine ? ["engine"] : []), "test", "t/s (total)", "t/s (req)", "TTFT p50"];
  const tokens = r.maxTokens ?? "";
  const rows = r.arms.map((a) => [
    r.model,
    ...(engine ? [engine] : []),
    `tg ${tokens} @ c${a.concurrency}`,
    num1(a.aggregate),
    num1(a.perStream),
    fmtMs(a.ttftP50),
  ]);
  const failures = r.arms.filter((a) => a.ok < a.requests);
  const notes = failures.length
    ? `\n\n${failures.map((a) => `- c${a.concurrency}: ${a.ok}/${a.requests} streams ok — ${a.errors.join("; ") || "errors"}`).join("\n")}`
    : "";
  return `${table(header, rows)}${notes}\n`;
}

export function prefillMarkdown(r: PrefillResult): string {
  const engine = engineColumn(r);
  const header = ["model", ...(engine ? ["engine"] : []), "test", "prompt tokens", "prefill t/s", "TTFT"];
  const rows = r.arms.map((a) => [
    r.model,
    ...(engine ? [engine] : []),
    `pp ${a.size}`,
    a.skipped ? `skipped: ${a.skipped}` : a.promptTokens != null ? String(a.promptTokens) : "",
    a.skipped ? "" : num1(a.prefillTokS),
    a.skipped ? "" : fmtMs(a.ttftMs),
  ]);
  return `${table(header, rows)}\n`;
}

export function benchMarkdown(r: BenchResult): string {
  return r.kind === "decode" ? decodeMarkdown(r) : prefillMarkdown(r);
}

export function benchJson(r: BenchResult, envelope?: Record<string, unknown> | null): string {
  return JSON.stringify(envelope ?? r, null, 2);
}
