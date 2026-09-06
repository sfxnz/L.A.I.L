"use client";

import { Eyebrow } from "@/components/ui";
import type { StrandState } from "@/lib/stream-run-types";
import { fmtInt } from "@/lib/streams/format";
import { cn } from "@/lib/utils";

/**
 * The card's sync bar — SyncBar's grammar (20 segments, 4px, eyebrow + unit)
 * with a phase lead: the first segment is the TTFT phase (line-2) and lights
 * when the first token lands; the rest is decode progress toward the budget
 * (line). Natural-EOS runs have no budget to fill, so decode is an
 * indeterminate "compiling" strand until the model stops.
 */
export function StrandSyncBar({
  state,
  tokens,
  maxTokens,
  fillToMax,
  finishReason,
  desync,
  className,
}: {
  state: StrandState;
  tokens: number | undefined;
  maxTokens: number;
  fillToMax: boolean;
  finishReason?: string;
  desync: boolean;
  className?: string;
}) {
  const SEGMENTS = 20;
  const terminal = state === "done" || state === "error" || state === "cancelled";
  const tok = tokens ?? 0;
  const decodeShare = maxTokens > 0 ? Math.min(1, tok / maxTokens) : 0;
  const decodeSegs = SEGMENTS - 1;
  const filledDecode =
    state === "done" ? decodeSegs : state === "decode" && fillToMax ? Math.round(decodeShare * decodeSegs) : 0;
  const leadLit = state === "decode" || terminal;
  const indeterminate = state === "prefill" || (state === "decode" && !fillToMax);

  const label =
    state === "waiting"
      ? "waiting"
      : state === "prefill"
        ? "prefill · ttft"
        : state === "decode"
          ? fillToMax
            ? "decode"
            : "compiling"
          : desync
            ? "desync"
            : state;
  const unit =
    state === "waiting"
      ? "arrival"
      : state === "prefill"
        ? "awaiting first token"
        : fillToMax && !terminal
          ? `${fmtInt(tok)} / ${fmtInt(maxTokens)} tok`
          : `${fmtInt(tok)} tok${terminal && finishReason ? ` · ${finishReason}` : ""}`;
  const pct = fillToMax || terminal ? Math.round((state === "done" ? 1 : decodeShare) * 100) : null;

  const fill = desync
    ? "bg-lab-danger"
    : state === "cancelled"
      ? "bg-lab-warn"
      : "bg-lab-line";

  return (
    <div className={cn("space-y-1", className)}>
      <div className="flex items-center justify-between gap-2">
        <Eyebrow className={cn("truncate", desync && "text-lab-danger")}>{label}</Eyebrow>
        <span className="lab-num flex shrink-0 items-baseline gap-2 font-mono text-[10px] text-lab-text-dim">
          <span className="text-lab-muted">{unit}</span>
          {pct !== null && !indeterminate ? <span>{pct}%</span> : null}
        </span>
      </div>
      <div
        className="relative grid h-[4px] gap-px overflow-hidden"
        style={{ gridTemplateColumns: `repeat(${SEGMENTS}, minmax(0, 1fr))` }}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct === null || indeterminate ? undefined : pct}
        aria-label={label}
        aria-busy={indeterminate || undefined}
      >
        <span className={cn("h-full transition-colors duration-[var(--dur-sync)]", leadLit ? "bg-lab-line-2" : "bg-lab-hover")} />
        {Array.from({ length: decodeSegs }).map((_, k) => (
          <span
            key={k}
            className={cn(
              "h-full transition-colors duration-[var(--dur-sync)] ease-[var(--ease-animus-out)]",
              k < filledDecode ? fill : "bg-lab-hover",
            )}
          />
        ))}
        {indeterminate && !desync && (
          <span
            aria-hidden
            className={cn(
              "lab-progress-indeterminate absolute inset-y-0 w-1/3 opacity-70",
              state === "prefill" ? "left-0 bg-lab-line-2" : "left-[5%] bg-lab-line",
            )}
          />
        )}
      </div>
    </div>
  );
}
