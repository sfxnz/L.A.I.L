"use client";

import { Eyebrow, Nil } from "@/components/ui";
import { fmtKvPct, fmtTokensK, kvFraction, kvUsedTokens } from "@/lib/status/forecast";
import { fmtGib } from "@/lib/status/format";
import { cn } from "@/lib/utils";

/**
 * The unified-memory story in one bar: engine reservation | other used | free, in GiB,
 * swap beside it and, when the engine reports it, the KV pool's utilisation as a thin
 * sub-track (tooltip in tokens). Warn colour follows the node's memory PRESSURE (RAM and
 * swap running out together), not a fixed "free < 15 GiB" line that a healthy GB10 serve
 * always crosses. Nothing is fabricated — an unknown reading is <Nil/>, and a segment or
 * track is drawn only when its source actually reported.
 */
export function MemoryBar({
  usedGib,
  totalGib,
  reservedGib,
  swapUsedGib,
  swapTotalGib,
  pressure,
  kvUsage,
  kvCapacityTokens,
  source,
  className,
}: {
  usedGib: number | null | undefined;
  totalGib: number | null | undefined;
  /** GiB the serving engine holds (nvidia-smi compute-apps), drawn inside "used" */
  reservedGib?: number | null;
  swapUsedGib?: number | null;
  swapTotalGib?: number | null;
  /** node mem_pressure from the serve-engine */
  pressure?: "ok" | "tight" | "critical" | null;
  /** engine.kv_usage_pct — a percent, 0–100 */
  kvUsage?: number | null;
  kvCapacityTokens?: number | null;
  /** where used/total came from, for the tooltip */
  source?: string;
  className?: string;
}) {
  const known = usedGib != null && totalGib != null && totalGib > 0;
  const usedPct = known ? Math.max(0, Math.min(100, (usedGib / totalGib) * 100)) : null;
  const freeGib = known ? Math.max(0, totalGib - usedGib) : null;
  const reserved = known && reservedGib != null && reservedGib > 0 ? Math.min(reservedGib, usedGib) : null;
  const reservedPct = reserved != null ? (reserved / totalGib!) * 100 : 0;
  const warn = pressure === "tight" || pressure === "critical";
  const kv = kvUsage == null ? null : kvFraction(kvUsage);
  const kvTokens = kvUsedTokens(kvCapacityTokens, kvUsage);
  const kvTip =
    kvUsage == null
      ? undefined
      : kvCapacityTokens && kvTokens != null
        ? `KV ${fmtKvPct(kvUsage)} · ${fmtTokensK(kvTokens)} / ${fmtTokensK(kvCapacityTokens)} tokens`
        : `KV ${fmtKvPct(kvUsage)} of the pool`;
  const swapKnown = swapUsedGib != null && swapTotalGib != null && swapTotalGib > 0;
  const memTip = known
    ? [
        reserved != null
          ? `${fmtGib(reserved)} engine reservation + ${fmtGib(usedGib - reserved)} other used`
          : `${fmtGib(usedGib)} used`,
        `${fmtGib(freeGib)} available of ${fmtGib(totalGib)}${source ? ` (${source})` : ""}`,
        swapKnown ? `swap ${fmtGib(swapUsedGib)} / ${fmtGib(swapTotalGib)}` : null,
        pressure ? `pressure ${pressure}` : null,
      ]
        .filter(Boolean)
        .join(" · ")
    : undefined;

  return (
    <div className={cn("min-w-0", className)}>
      <div className="flex items-baseline justify-between gap-2">
        <Eyebrow>Memory</Eyebrow>
        <span className="lab-num font-mono text-[11px] text-lab-text-dim" title={memTip}>
          {known ? (
            <>
              <span className={cn("text-lab-text", warn && "text-lab-warn")}>{usedGib.toFixed(1)}</span>
              <span className="text-lab-muted"> / {totalGib.toFixed(1)} GiB</span>
              {swapKnown && (
                <span className={cn("text-lab-muted", warn && swapUsedGib > 0 && "text-lab-warn")}>
                  {" "}· swap {swapUsedGib.toFixed(1)}
                </span>
              )}
            </>
          ) : (
            <Nil />
          )}
        </span>
      </div>
      <div
        className="mt-1.5 flex h-[5px] w-full overflow-hidden bg-lab-hover"
        role={known ? "img" : undefined}
        aria-label={memTip}
        title={memTip}
      >
        {usedPct != null && (
          <>
            {reserved != null && (
              <div
                className={cn(
                  "h-full shrink-0 bg-lab-line transition-[width] duration-[var(--dur-sync)] ease-[var(--ease-animus-out)]",
                  warn && "bg-lab-warn",
                )}
                style={{ width: `${reservedPct}%` }}
              />
            )}
            <div
              className={cn(
                "h-full shrink-0 transition-[width] duration-[var(--dur-sync)] ease-[var(--ease-animus-out)]",
                reserved != null ? "bg-lab-line/45" : "bg-lab-line",
                warn && (reserved != null ? "bg-lab-warn/55" : "bg-lab-warn"),
              )}
              style={{ width: `${usedPct - reservedPct}%` }}
            />
          </>
        )}
      </div>
      {kv != null && (
        <div className="mt-px flex items-center gap-2" title={kvTip}>
          <div className="h-[3px] flex-1 overflow-hidden bg-lab-hover" role="img" aria-label={kvTip}>
            <div
              className="h-full origin-left bg-lab-line-2 transition-transform duration-[var(--dur-sync)] ease-[var(--ease-animus-out)]"
              style={{ transform: `scaleX(${kv})`, width: "100%" }}
            />
          </div>
          <span className="lab-num shrink-0 font-mono text-[9px] text-lab-muted">KV {fmtKvPct(kvUsage!)}</span>
        </div>
      )}
    </div>
  );
}
