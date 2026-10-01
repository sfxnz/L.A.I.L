"use client";

import { Eyebrow, Nil, Sparkline, type SparkPoint } from "@/components/ui";
import { fmtGib } from "@/lib/status/format";
import { cn } from "@/lib/utils";

/**
 * The unified-memory story of one Spark: engine reservation | other used | free, in
 * GiB, spelled out under the bar (no hover needed), swap beside it, and the last
 * 60 s of memory in use as a trend line. Warn colour follows the node's memory
 * PRESSURE (RAM and swap running out together), not a fixed "free < 15 GiB" line
 * that a healthy GB10 serve always crosses. An unknown reading is <Nil/>, and a
 * segment is drawn only when its source actually reported.
 */
export function MemoryBar({
  usedGib,
  totalGib,
  reservedGib,
  swapUsedGib,
  swapTotalGib,
  pressure,
  source,
  trend,
  domain,
  offline,
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
  /** where used/total came from, for the tooltip */
  source?: string;
  /** memory in use over time (GiB) */
  trend?: readonly SparkPoint[];
  domain?: readonly [number, number];
  /** the node is down: say so instead of "awaiting" */
  offline?: boolean;
  className?: string;
}) {
  const known = usedGib != null && totalGib != null && totalGib > 0;
  const usedPct = known ? Math.max(0, Math.min(100, (usedGib / totalGib) * 100)) : null;
  const freeGib = known ? Math.max(0, totalGib - usedGib) : null;
  const reserved = known && reservedGib != null && reservedGib > 0 ? Math.min(reservedGib, usedGib) : null;
  const reservedPct = reserved != null ? (reserved / totalGib!) * 100 : 0;
  const warn = pressure === "tight" || pressure === "critical";
  const swapKnown = swapUsedGib != null && swapTotalGib != null && swapTotalGib > 0;
  // Trend scale: a ±1 GiB band around what was seen, so a model loading (tens of GiB)
  // reads as a ramp and 50 MB of page cache churn stays flat.
  const seen = (trend ?? []).map((p) => p.v).filter((v): v is number => v != null);
  const trendLo = seen.length ? Math.max(0, Math.floor(Math.min(...seen) - 1)) : 0;
  const trendHi = seen.length ? Math.ceil(Math.max(...seen) + 1) : undefined;
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
            </>
          ) : (
            <Nil word={offline ? "Offline" : "Awaiting"} />
          )}
        </span>
      </div>
      <div
        className="mt-1.5 flex h-[6px] w-full overflow-hidden bg-lab-hover"
        role={known ? "img" : undefined}
        aria-label={memTip}
        title={memTip}
      >
        {usedPct != null && (
          <>
            {reserved != null && (
              <div
                className={cn("lab-bar h-full shrink-0 bg-lab-line", warn && "bg-lab-warn")}
                style={{ width: `${reservedPct}%` }}
              />
            )}
            <div
              className={cn(
                "lab-bar h-full shrink-0",
                reserved != null ? "bg-lab-line/45" : "bg-lab-line",
                warn && (reserved != null ? "bg-lab-warn/55" : "bg-lab-warn"),
              )}
              style={{ width: `${usedPct - reservedPct}%` }}
            />
          </>
        )}
      </div>
      {known && (
        <div className="lab-num mt-1 flex flex-wrap gap-x-2.5 gap-y-0.5 font-mono text-[10px] text-lab-muted">
          {reserved != null && (
            <span>
              <span aria-hidden className="mr-1 inline-block h-1.5 w-1.5 bg-lab-line align-middle" />
              engine {reserved.toFixed(1)}
            </span>
          )}
          <span>
            <span aria-hidden className="mr-1 inline-block h-1.5 w-1.5 bg-lab-line/45 align-middle" />
            {reserved != null ? "other" : "used"} {(usedGib - (reserved ?? 0)).toFixed(1)}
          </span>
          <span>
            <span aria-hidden className="mr-1 inline-block h-1.5 w-1.5 border border-lab-border bg-lab-hover align-middle" />
            free {freeGib!.toFixed(1)}
          </span>
          {swapKnown && (
            <span className={cn(warn && swapUsedGib > 0 && "text-lab-warn")}>
              swap {swapUsedGib.toFixed(1)} / {swapTotalGib.toFixed(0)}
            </span>
          )}
        </div>
      )}
      {trend && trend.length > 0 && (
        <Sparkline
          points={trend}
          domain={domain}
          width={240}
          height={16}
          min={trendLo}
          max={trendHi}
          area={false}
          className="mt-1 w-full text-lab-line"
          label={`Memory in use over the last 60 s (${trendLo}–${trendHi ?? ""} GiB)`}
        />
      )}
    </div>
  );
}
