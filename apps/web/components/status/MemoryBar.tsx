"use client";

import { Eyebrow, Nil } from "@/components/ui";
import { fmtTokensK, kvFraction, kvUsedTokens } from "@/lib/status/forecast";
import { fmtGib } from "@/lib/status/format";
import { cn } from "@/lib/utils";

/**
 * The unified-memory story in one bar: used / free in GiB on the main track and,
 * when the engine reports it, the KV pool's utilisation as a thin sub-track
 * (tooltip in tokens). Nothing is fabricated — an unknown reading is <Nil/>,
 * and the KV track is drawn only when the engine actually reported it.
 */
export function MemoryBar({
  usedGib,
  totalGib,
  kvUsage,
  kvCapacityTokens,
  source,
  className,
}: {
  usedGib: number | null | undefined;
  totalGib: number | null | undefined;
  /** 0–1 fraction (vLLM) or 0–100 percent (C2) */
  kvUsage?: number | null;
  kvCapacityTokens?: number | null;
  /** where used/total came from, for the tooltip */
  source?: string;
  className?: string;
}) {
  const known = usedGib != null && totalGib != null && totalGib > 0;
  const usedPct = known ? Math.max(0, Math.min(100, (usedGib / totalGib) * 100)) : null;
  const freeGib = known ? Math.max(0, totalGib - usedGib) : null;
  const kv = kvUsage == null ? null : kvFraction(kvUsage);
  const kvTokens = kvUsedTokens(kvCapacityTokens, kvUsage);
  const kvTip =
    kv == null
      ? undefined
      : kvCapacityTokens && kvTokens != null
        ? `KV ${Math.round(kv * 100)} % · ${fmtTokensK(kvTokens)} / ${fmtTokensK(kvCapacityTokens)} tokens`
        : `KV ${Math.round(kv * 100)} % of the pool`;
  const memTip = known
    ? `${fmtGib(usedGib)} used · ${fmtGib(freeGib)} free of ${fmtGib(totalGib)}${source ? ` (${source})` : ""}`
    : undefined;

  return (
    <div className={cn("min-w-0", className)}>
      <div className="flex items-baseline justify-between gap-2">
        <Eyebrow>Memory</Eyebrow>
        <span className="lab-num font-mono text-[11px] text-lab-text-dim" title={memTip}>
          {known ? (
            <>
              <span className={cn("text-lab-text", freeGib != null && freeGib < 15 && "text-lab-warn")}>{usedGib.toFixed(1)}</span>
              <span className="text-lab-muted"> / {totalGib.toFixed(1)} GiB</span>
            </>
          ) : (
            <Nil />
          )}
        </span>
      </div>
      <div
        className="mt-1.5 h-[5px] w-full overflow-hidden bg-lab-hover"
        role={known ? "img" : undefined}
        aria-label={memTip}
        title={memTip}
      >
        {usedPct != null && (
          <div
            className={cn(
              "h-full origin-left bg-lab-line transition-transform duration-[var(--dur-sync)] ease-[var(--ease-animus-out)]",
              freeGib != null && freeGib < 15 && "bg-lab-warn",
            )}
            style={{ transform: `scaleX(${usedPct / 100})`, width: "100%" }}
          />
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
          <span className="lab-num shrink-0 font-mono text-[9px] text-lab-muted">KV {Math.round(kv * 100)}%</span>
        </div>
      )}
    </div>
  );
}
