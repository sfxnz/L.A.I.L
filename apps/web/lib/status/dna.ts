import type { RunRow } from "../api";
import { headlineFromIndex } from "../bench/result";

/**
 * The DNA strand: the last 30 runs as slices. Kind → colour token (a `bg-lab-*`
 * class, so both themes resolve), kind → destination, index row → one-line
 * headline. Pure, so the strand component is a renderer with a keyboard.
 */

export type DnaSlice = {
  id: string;
  kind: string;
  model: string | null;
  createdAt: string;
  colorClass: string;
  href: string;
  headline: string | null;
};

export function sliceColor(kind: string | null | undefined): string {
  switch (kind) {
    case "decode":
      return "bg-lab-line";
    case "prefill":
      return "bg-lab-line-2";
    default:
      return kind && /tool|eval/i.test(kind) ? "bg-lab-chart-3" : "bg-lab-muted";
  }
}

export function sliceHref(run: Pick<RunRow, "run_id" | "kind">): string {
  if (run.kind === "decode" || run.kind === "prefill") return `/bench?run=${encodeURIComponent(run.run_id)}`;
  if (run.kind && /tool/i.test(run.kind)) return `/evals/tool/${encodeURIComponent(run.run_id)}`;
  return "/evals";
}

function fmtTokS(v: number): string {
  return v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v >= 100 ? String(Math.round(v)) : v.toFixed(1);
}

export function sliceHeadline(run: RunRow): string | null {
  const h = headlineFromIndex(run);
  if (run.kind === "decode" && h.peak !== null) {
    return `${fmtTokS(h.peak)} tok/s${h.peakAt !== null ? ` @ ×${h.peakAt}` : ""}`;
  }
  if (run.kind === "prefill" && h.sustained !== null) return `${fmtTokS(h.sustained)} tok/s sustained`;
  const s = run.summary as Record<string, unknown> | undefined;
  const score = s?.final_score ?? s?.score;
  if (typeof score === "number") return `score ${Math.round(score)}`;
  return null;
}

/** Newest first, capped at `limit`. */
export function toSlices(runs: RunRow[], limit = 30): DnaSlice[] {
  return runs
    .slice()
    .sort((a, b) => (a.created_at < b.created_at ? 1 : a.created_at > b.created_at ? -1 : 0))
    .slice(0, limit)
    .map((r) => ({
      id: r.run_id,
      kind: r.kind,
      model: r.model_id,
      createdAt: r.created_at,
      colorClass: sliceColor(r.kind),
      href: sliceHref(r),
      headline: sliceHeadline(r),
    }));
}
