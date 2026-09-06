"use client";

import { fmtSize } from "@/lib/bench/format";
import type { LevelRow } from "@/lib/use-stream-run";
import { SyncRing, type SyncState } from "@/components/ui";
import { cn } from "@/lib/utils";

/**
 * One cell per level (×N or a context size): a sync ring for its state and a
 * segmented fill for the wave in progress. Levels the engine skipped appear
 * immediately, hatched, with the reason on hover — never silently dropped.
 */
export type LevelCellState = "pending" | "running" | "done" | "partial" | "failed" | "skipped";

export function levelCellState(row: LevelRow | undefined, isCurrent: boolean, running: boolean): LevelCellState {
  if (row?.skipped) return "skipped";
  if (row) {
    if (row.ok === 0) return "failed";
    return row.ok < row.requests ? "partial" : "done";
  }
  return isCurrent && running ? "running" : "pending";
}

const RING: Record<LevelCellState, SyncState | null> = {
  pending: "idle",
  running: "loading",
  done: "serving",
  partial: "loading",
  failed: "offline",
  skipped: null,
};

export function LevelRings({
  keys,
  kind,
  rows,
  current,
  running,
  fill,
  segments,
  className,
}: {
  /** concurrency per level, or size per level */
  keys: number[];
  kind: "decode" | "prefill";
  rows: LevelRow[];
  current: number;
  running: boolean;
  /** 0–1 fill of the current level */
  fill: number;
  /** segments for the current level's bar (strand count; prefill = 8) */
  segments: number;
  className?: string;
}) {
  return (
    <ol className={cn("grid gap-1.5", className)} style={{ gridTemplateColumns: `repeat(${Math.max(1, keys.length)}, minmax(0, 1fr))` }}>
      {keys.map((k, i) => {
        const row = rows.find((r) => r.index === i);
        const state = levelCellState(row, i === current, running);
        const isCur = i === current && running && !row;
        const segs = Math.max(1, Math.min(32, isCur ? segments : 1));
        const filled = state === "done" || state === "partial" || state === "failed" ? segs : isCur ? Math.round(fill * segs) : 0;
        const label = kind === "decode" ? `×${k}` : fmtSize(k);
        const title =
          state === "skipped"
            ? `${label} — skipped: ${row?.skipped}`
            : state === "partial"
              ? `${label} — ${row?.ok}/${row?.requests} strands`
              : state === "failed"
                ? `${label} — every strand failed`
                : label;
        return (
          <li
            key={k}
            className={cn(
              "min-w-0 border border-lab-border-subtle px-2 py-1.5 transition-colors duration-[var(--dur-exit)]",
              isCur && "border-lab-line",
              state === "skipped" && "bench-hatched opacity-70",
            )}
            title={title}
            aria-current={isCur ? "step" : undefined}
          >
            <div className="flex items-center justify-between gap-1">
              <span
                className={cn(
                  "lab-num truncate font-[family-name:var(--font-display)] text-[12px] font-semibold",
                  state === "pending" || state === "skipped" ? "text-lab-muted" : "text-lab-text",
                )}
              >
                {label}
              </span>
              {state === "skipped" ? (
                <span className="animus-eyebrow text-[8px]">skip</span>
              ) : (
                <SyncRing state={RING[state]} size={12} label={`${label} ${state}`} />
              )}
            </div>
            <div
              className="mt-1.5 grid h-[3px] gap-px"
              style={{ gridTemplateColumns: `repeat(${segs}, minmax(0, 1fr))` }}
              aria-hidden
            >
              {Array.from({ length: segs }).map((_, j) => (
                <span
                  key={j}
                  className={cn(
                    "h-full transition-colors duration-[var(--dur-sync)] ease-[var(--ease-animus-out)]",
                    j < filled
                      ? state === "failed"
                        ? "bg-lab-danger"
                        : state === "partial"
                          ? "bg-lab-warn"
                          : "bg-lab-line-2"
                      : "bg-lab-hover",
                  )}
                />
              ))}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
