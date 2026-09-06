"use client";

import { useRef } from "react";
import { fmtTokS } from "@/lib/bench/format";
import { gatedRate } from "@/lib/bench/live";
import type { StrandView } from "@/lib/use-stream-run";
import { Eyebrow, LogView } from "@/components/ui";
import { cn } from "@/lib/utils";

/**
 * ×1: the strand's real text streams at true speed with a gated tok/s pill.
 * ×N: N lanes, each filling left → right with that strand's share of its token
 * budget; error lanes turn danger and the level reads "7/8 strands".
 */

/** Tracks when each strand started decoding, for the pill gate. */
export function useDecodeStarts(strands: StrandView[]): Map<number, number> {
  const starts = useRef(new Map<number, number>());
  const now = Date.now();
  for (const s of strands) {
    if (s.state === "decode" && !starts.current.has(s.i)) starts.current.set(s.i, now);
    if (s.state === "waiting") starts.current.delete(s.i);
  }
  return starts.current;
}

function RatePill({ rate, state }: { rate: number | null; state: StrandView["state"] }) {
  const tone =
    state === "error" || state === "cancelled"
      ? "border-lab-danger text-lab-danger"
      : state === "done"
        ? "border-lab-ok text-lab-ok"
        : "border-lab-line-2 text-lab-text";
  return (
    <span
      className={cn("lab-num animus-chamfer-sm inline-flex h-6 min-w-[4.5rem] items-center justify-center gap-1 border px-2 font-mono text-[11px]", tone)}
      aria-live="off"
    >
      {rate !== null ? (
        <>
          {fmtTokS(rate)} <span className="text-lab-muted">t/s</span>
        </>
      ) : state === "prefill" ? (
        <span className="text-lab-muted">prefill…</span>
      ) : state === "waiting" ? (
        <span className="text-lab-muted">waiting</span>
      ) : (
        <span className="text-lab-muted">…</span>
      )}
    </span>
  );
}

export function StrandLanes({
  strands,
  maxTokens,
  className,
}: {
  strands: StrandView[];
  maxTokens: number;
  className?: string;
}) {
  // The parent re-renders on every agg tick (4 Hz) and its elapsed clock, so the
  // 500 ms gate opens without a timer of its own.
  const starts = useDecodeStarts(strands);
  const now = Date.now();
  const n = strands.length;

  if (n === 0) {
    return (
      <div className={cn("flex h-40 items-center justify-center", className)}>
        <Eyebrow>Awaiting strands</Eyebrow>
      </div>
    );
  }

  if (n === 1) {
    const s = strands[0];
    return (
      <div className={cn("space-y-2", className)}>
        <div className="flex items-center justify-between gap-2">
          <Eyebrow className="truncate">
            Strand {s.i + 1} · {s.title}
          </Eyebrow>
          <RatePill rate={gatedRate(s, starts.get(s.i), now)} state={s.state} />
        </div>
        <LogView
          text={s.text}
          empty={s.state === "prefill" ? "Prefilling…" : "Waiting for the first token…"}
          live={s.state === "decode" || s.state === "prefill"}
          className="h-40 max-h-40 text-[12px] leading-[1.55] text-lab-text"
        />
        {s.error ? (
          <p className="text-[11px] text-lab-danger" role="alert">
            {s.error}
          </p>
        ) : null}
      </div>
    );
  }

  const dense = n > 8;
  const veryDense = n > 16;
  return (
    <div
      className={cn("grid gap-x-3", veryDense ? "grid-cols-2 gap-y-1" : dense ? "gap-y-1" : "gap-y-1.5", className)}
      role="list"
      aria-label={`${n} strands`}
    >
      {strands.map((s, idx) => {
        const frac = Math.max(0, Math.min(1, (s.tokens ?? 0) / Math.max(1, maxTokens)));
        const failed = s.state === "error" || s.state === "cancelled";
        const fill = failed
          ? "bg-lab-danger"
          : s.state === "done"
            ? "bg-lab-line-2"
            : s.state === "decode"
              ? "bg-lab-line"
              : "bg-lab-border-strong";
        const rate = gatedRate(s, starts.get(s.i), now);
        return (
          <div key={s.i} role="listitem" className="flex items-center gap-2" title={s.error ?? `${s.title} · ${s.tokens ?? 0} tok`}>
            {!veryDense && (
              <span className="lab-num w-7 shrink-0 text-right font-mono text-[10px] text-lab-muted">#{idx + 1}</span>
            )}
            <div className={cn("relative flex-1 overflow-hidden bg-lab-hover", veryDense ? "h-[6px]" : dense ? "h-2" : "h-2.5")}>
              {s.state === "prefill" && (
                <span aria-hidden className="lab-progress-indeterminate absolute inset-y-0 left-0 w-1/3 bg-lab-line-2 opacity-60" />
              )}
              <span
                aria-hidden
                className={cn("bench-lane-fill absolute inset-y-0 left-0 w-full", fill)}
                style={{ transform: `scaleX(${failed ? 1 : frac})`, opacity: failed ? 0.6 : 1 }}
              />
            </div>
            {!dense && (
              <span className="lab-num w-[7.5rem] shrink-0 text-right font-mono text-[10px] text-lab-text-dim">
                {failed ? (
                  <span className="text-lab-danger">{s.state === "cancelled" ? "cancelled" : "error"}</span>
                ) : (
                  <>
                    {s.tokens ?? 0}
                    <span className="text-lab-muted"> tok</span>
                    {rate !== null ? (
                      <>
                        {" · "}
                        {fmtTokS(rate)}
                        <span className="text-lab-muted"> t/s</span>
                      </>
                    ) : null}
                  </>
                )}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
