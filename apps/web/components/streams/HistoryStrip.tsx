"use client";

import { Badge, EmptyState, Eyebrow, Panel, SyncRing } from "@/components/ui";
import type { StreamRunRow } from "@/lib/stream-run-types";
import { fmtInt, fmtRate, shortModel } from "@/lib/streams/format";
import { cn } from "@/lib/utils";

function when(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/**
 * Recent Sequences (in-memory, last 50 from the controller). Reuse pre-fills
 * the controls; Open attaches to the run — live ones re-attach through the
 * snapshot path, finished ones replay.
 */
export function HistoryStrip({
  runs,
  currentRunId,
  packLabel,
  onReuse,
  onOpen,
  className,
}: {
  runs: StreamRunRow[];
  currentRunId: string | null;
  packLabel: (id: string) => string;
  onReuse: (row: StreamRunRow) => void;
  onOpen: (row: StreamRunRow) => void;
  className?: string;
}) {
  return (
    <Panel title="Run history" className={cn("streams-history", className)} action={<Eyebrow className="lab-num">{runs.length} recent</Eyebrow>}>
      {!runs.length ? (
        <EmptyState title="No sequences yet">Run to draw the first strand.</EmptyState>
      ) : (
        <ul className="divide-y divide-lab-border-subtle">
          {runs.map((r) => {
            const live = r.status === "running";
            const current = r.run_id === currentRunId;
            return (
              <li
                key={r.run_id}
                className={cn(
                  "animus-notch flex flex-wrap items-center gap-x-3 gap-y-1 px-4 py-2 transition-[background,box-shadow] hover:bg-[color:var(--animus-accent-wash)] hover:shadow-[inset_2px_0_0_var(--color-lab-accent)]",
                  current && "shadow-[inset_2px_0_0_var(--color-lab-line)]",
                )}
              >
                <SyncRing state={live ? "loading" : r.status === "done" ? "serving" : r.status === "error" ? "offline" : "idle"} label={r.status} size={12} />
                <span className="lab-num w-[64px] shrink-0 font-mono text-[10px] text-lab-muted">{when(r.started_at)}</span>
                <span className="min-w-0 max-w-[180px] truncate font-mono text-[11px] text-lab-text" title={r.model}>
                  {shortModel(r.model)}
                </span>
                <Eyebrow className="shrink-0 text-lab-text-dim">{packLabel(r.pack)}</Eyebrow>
                <span className="lab-num shrink-0 font-mono text-[11px] text-lab-text-dim">×{r.n}</span>
                <Badge tone={r.mode === "load" ? "muted" : "accent"}>{r.mode}</Badge>
                <span
                  className="lab-num shrink-0 font-mono text-[11px] text-lab-text-dim"
                  title={r.mode === "load" ? "aggregate (decode span) · live peak tok/s" : "headline: peak level aggregate · ×1 per stream"}
                >
                  {r.summary && r.mode === "bench-prefill" ? (
                    <>
                      {fmtRate(r.summary.headline?.prefill_tok_per_s_sustained)} <span className="text-lab-muted">prefill</span>
                    </>
                  ) : r.summary && r.mode === "bench-decode" ? (
                    <>
                      {fmtRate(r.summary.aggregate_tok_s)} <span className="text-lab-muted">pk @×{r.summary.headline?.aggregate_peak_concurrency ?? "—"}</span> ·{" "}
                      {fmtRate(r.summary.per_stream_median_tok_s)} <span className="text-lab-muted">×1</span>
                    </>
                  ) : r.summary ? (
                    <>
                      {fmtRate(r.summary.aggregate_steady_tok_s)} <span className="text-lab-muted">agg</span> · {fmtRate(r.summary.peak_tok_s)} <span className="text-lab-muted">pk</span>
                      {r.summary.tokens ? <> · {fmtInt(r.summary.tokens)} <span className="text-lab-muted">tok</span></> : null}
                    </>
                  ) : (
                    <span className="text-lab-muted">live</span>
                  )}
                </span>
                <Badge tone={live ? "warn" : r.status === "done" ? "ok" : r.status === "error" ? "danger" : "muted"}>{r.status}</Badge>
                <span className="ml-auto flex shrink-0 items-center gap-1">
                  {r.mode === "load" && (
                    <button type="button" className="strand-link" onClick={() => onReuse(r)} title="Pre-fill the controls from this run">
                      Reuse
                    </button>
                  )}
                  <button type="button" className="strand-link" onClick={() => onOpen(r)} title={live ? "Re-attach to the live run" : "Replay the finished run"} disabled={current}>
                    {current ? "Open" : live ? "Attach" : "Open"}
                  </button>
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </Panel>
  );
}
