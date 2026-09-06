/**
 * Export shapes for a Sequence (a run): a Markdown summary (instrument numbers +
 * a per-strand table, optionally with transcripts) and a JSON snapshot of the
 * client state. Pure — the page supplies the state and the run's controls.
 */
import type { StreamRunState, StrandView } from "../use-stream-run";
import { fmtDuration, fmtInt, fmtMs, fmtRate } from "./format";
import { itlStats } from "./stalls";

export type RunControls = {
  pack?: string;
  arrival?: string;
  fill_to_max?: boolean;
  thinking?: string;
};

export type ExportOptions = RunControls & {
  /** Resolve a pack id to its label (defaults to the id). */
  packLabel?: (id: string) => string;
  /** Append every strand's transcript (Copy all). */
  transcripts?: boolean;
};

const cell = (v: string) => v.replace(/\|/g, "\\|").replace(/\r?\n/g, " ");

/** The method chip's words, so the export says which number it is quoting. */
export function aggregateMethod(state: StreamRunState): { value: number | null; method: string } {
  const tpc = state.latest?.tokens_per_chunk ?? 1;
  const liveMethod = tpc > 1 ? `live · usage-calibrated ×${tpc.toFixed(1)} tok/chunk` : "live · chunk estimate";
  if (state.done) {
    const final = state.done.summary.aggregate_tok_s;
    if (final !== null) return { value: final, method: "final · usage" };
    // A cancelled run has no usage frames to settle on: quote the last live sample and say so.
    return { value: state.latest?.tok_s ?? null, method: `last sample · ${liveMethod.replace(/^live · /, "")}` };
  }
  return { value: state.latest?.tok_s ?? null, method: liveMethod };
}

function strandRow(s: StrandView, packLabel: (id: string) => string): string {
  const itl = itlStats(s.itl_ms);
  const reasoningShare = s.chunks ? s.reasoning_chunks / s.chunks : 0;
  return [
    String(s.i + 1),
    cell(packLabel(s.pack || "") || "—"),
    cell(s.title),
    s.state,
    fmtMs(s.ttft_ms),
    fmtInt(s.tokens),
    fmtRate(s.tok_s),
    fmtRate(s.peak_tok_s),
    itl.p50 === null ? "—" : `${Math.round(itl.p50)} / ${Math.round(itl.p95 ?? itl.p50)}`,
    reasoningShare ? `${Math.round(reasoningShare * 100)} %` : "—",
    s.error ? cell(s.error) : s.finish_reason || "—",
  ].join(" | ");
}

export function markdownSummary(state: StreamRunState, opts: ExportOptions = {}): string {
  const h = state.hello;
  const sum = state.done?.summary ?? null;
  const latest = state.latest;
  const packLabel = opts.packLabel ?? ((id: string) => id);
  const { value: aggregate, method } = aggregateMethod(state);
  const lines: string[] = [];
  lines.push(`# Streams · ${h?.mode ?? "load"} · ${h?.model ?? "—"}`);
  lines.push("");
  const facts = [
    h ? `run \`${h.run_id}\`` : null,
    h ? `endpoint ${h.base_url}` : null,
    opts.pack ? `pack ${packLabel(opts.pack)}` : null,
    h ? `strands ${h.n}` : null,
    h ? `max_tokens ${h.max_tokens}` : null,
    opts.fill_to_max === undefined ? null : opts.fill_to_max ? "fill to max" : "natural EOS",
    opts.thinking ? `thinking ${opts.thinking}` : null,
    opts.arrival ? `arrival ${opts.arrival}` : null,
    h ? `started ${h.started_at}` : null,
  ].filter((f): f is string => !!f);
  lines.push(facts.join(" · "));
  lines.push("");
  lines.push("| Metric | Value |");
  lines.push("|---|---|");
  lines.push(`| Aggregate tok/s (${method}) | ${fmtRate(aggregate)} |`);
  lines.push(`| Peak tok/s (live 1 s window) | ${fmtRate(sum?.peak_tok_s ?? latest?.peak_tok_s)} |`);
  lines.push(`| Tokens | ${fmtInt(sum?.tokens ?? latest?.tokens)} |`);
  lines.push(`| TTFT p50 / p95 | ${fmtMs(sum?.ttft_p50_ms ?? latest?.ttft_p50_ms)} / ${fmtMs(sum?.ttft_p95_ms ?? latest?.ttft_p95_ms)} |`);
  lines.push(`| Per-strand median tok/s | ${fmtRate(sum?.per_stream_median_tok_s)} |`);
  lines.push(`| Duration | ${fmtDuration(sum?.duration_ms ?? latest?.t_ms)} |`);
  lines.push(`| Strands ok | ${sum ? `${sum.ok} / ${sum.requests}` : `${latest?.done ?? 0} done · ${latest?.running ?? 0} streaming · ${latest?.waiting ?? 0} waiting`} |`);
  if (sum?.errors.length) lines.push(`| Errors | ${cell(sum.errors.join("; "))} |`);
  lines.push("");
  lines.push("## Strands");
  lines.push("");
  lines.push("| # | Pack | Prompt | State | TTFT | Tokens | tok/s | Peak | ITL p50 / p95 ms | Thinking | Finish |");
  lines.push("|---|---|---|---|---|---|---|---|---|---|---|");
  for (const s of state.strands) lines.push(`| ${strandRow(s, packLabel)} |`);
  if (opts.transcripts) {
    for (const s of state.strands) {
      lines.push("");
      lines.push(`### ${s.i + 1} · ${s.title}`);
      lines.push("");
      lines.push("> " + s.prompt.split(/\r?\n/).join("\n> "));
      if (s.reasoning) {
        lines.push("");
        lines.push("<details><summary>thinking</summary>");
        lines.push("");
        lines.push(s.reasoning);
        lines.push("");
        lines.push("</details>");
      }
      lines.push("");
      lines.push(s.text || "_(no output)_");
    }
  }
  lines.push("");
  return lines.join("\n");
}

export function jsonSnapshot(state: StreamRunState, controls: RunControls = {}): Record<string, unknown> {
  return {
    exported_at: new Date().toISOString(),
    source: "lail-streams",
    controls,
    hello: state.hello,
    summary: state.done?.summary ?? null,
    saved_run_id: state.done?.saved_run_id ?? null,
    error: state.error,
    strands: state.strands.map((s) => ({
      i: s.i,
      title: s.title,
      pack: s.pack,
      prompt: s.prompt,
      state: s.state,
      ttft_ms: s.ttft_ms ?? null,
      tokens: s.tokens ?? null,
      tok_s: s.tok_s ?? null,
      peak_tok_s: s.peak_tok_s ?? null,
      finish_reason: s.finish_reason ?? null,
      error: s.error ?? null,
      chunks: s.chunks,
      reasoning_chunks: s.reasoning_chunks,
      itl_ms: s.itl_ms ?? [],
      text: s.text,
      reasoning: s.reasoning,
    })),
    agg: state.agg,
  };
}

/** File stem for downloads: `streams-<model>-<run>`. */
export function exportStem(state: StreamRunState): string {
  const model = (state.hello?.model ?? "run").split("/").pop() ?? "run";
  const id = state.hello?.run_id ?? "snapshot";
  return `streams-${model}-${id}`.replace(/[^A-Za-z0-9._-]+/g, "_");
}
