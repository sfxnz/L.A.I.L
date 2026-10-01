import type { RunRow } from "./api";

/** Bench runs open in /bench, tool-eval runs on their scorecard; other kinds (legacy_*, …) have no view. */
export function runHref(r: Pick<RunRow, "kind" | "run_id">): string | null {
  if (r.kind === "decode" || r.kind === "prefill") return `/bench?run=${encodeURIComponent(r.run_id)}`;
  if (r.kind === "agentic_tool_eval") return `/evals/tool/${encodeURIComponent(r.run_id)}`;
  return null;
}
