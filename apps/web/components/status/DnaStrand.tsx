"use client";

import Link from "next/link";
import { useRef, useState, type KeyboardEvent } from "react";
import type { RunRow } from "@/lib/api";
import { fmtDate, modelShort } from "@/lib/bench/format";
import { toSlices, type DnaSlice } from "@/lib/status/dna";
import { Badge, Eyebrow, Nil, Panel, Skeleton } from "@/components/ui";
import { cn } from "@/lib/utils";

const LEGEND: ReadonlyArray<[string, string]> = [
  ["decode", "bg-lab-line"],
  ["prefill", "bg-lab-line-2"],
  ["tool / eval", "bg-lab-chart-3"],
  ["other", "bg-lab-muted"],
];

/**
 * The DNA strand — the last 30 runs as one horizontal strand of slices, oldest
 * left, newest right, coloured by kind. Hover or focus a slice and its card
 * compiles beneath; ←/→ scrub when the strand has focus; click/↩ opens the run
 * (/bench for decode/prefill, /evals otherwise).
 */
export function DnaStrand({ runs, loading, className }: { runs: RunRow[]; loading: boolean; className?: string }) {
  const slices = toSlices(runs, 30).reverse(); // oldest → newest
  const [active, setActive] = useState<number | null>(null);
  const refs = useRef<Array<HTMLAnchorElement | null>>([]);
  const shown: DnaSlice | null = active !== null ? slices[active] ?? null : slices[slices.length - 1] ?? null;

  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if (!slices.length) return;
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight" && e.key !== "Home" && e.key !== "End") return;
    e.preventDefault();
    const cur = active ?? slices.length - 1;
    const next =
      e.key === "Home" ? 0 : e.key === "End" ? slices.length - 1 : Math.max(0, Math.min(slices.length - 1, cur + (e.key === "ArrowRight" ? 1 : -1)));
    setActive(next);
    refs.current[next]?.focus();
  }

  return (
    <Panel
      title="Sequences"
      className={cn("flex h-full flex-col", className)}
      action={<Eyebrow className="lab-num">{slices.length ? `last ${slices.length} · ← → scrub` : ""}</Eyebrow>}
    >
      <div className="flex flex-1 flex-col gap-3 p-4">
        {loading && !slices.length ? (
          <Skeleton className="h-10 w-full" aria-hidden />
        ) : !slices.length ? (
          <div className="flex flex-1 flex-col items-start gap-2 py-1">
            <div className="flex h-10 w-full items-stretch gap-px opacity-60" aria-hidden>
              <span className="w-[6%] border border-dashed border-lab-border" />
            </div>
            <p className="text-[12px] leading-snug text-lab-muted">No sequences yet. Run a decode sync to draw the first slice.</p>
          </div>
        ) : (
          <div
            role="listbox"
            aria-label="Run history strand"
            aria-activedescendant={active !== null ? `dna-${slices[active]?.id}` : undefined}
            tabIndex={-1}
            onKeyDown={onKeyDown}
            onMouseLeave={() => setActive(null)}
            className="flex h-10 w-full items-stretch gap-px"
          >
            {slices.map((s, i) => {
              const on = active === i;
              return (
                <Link
                  key={s.id}
                  id={`dna-${s.id}`}
                  ref={(el) => {
                    refs.current[i] = el;
                  }}
                  href={s.href}
                  role="option"
                  aria-selected={on}
                  aria-label={`${s.kind} · ${modelShort(s.model) || "unknown model"} · ${s.headline ?? "no headline"} · ${fmtDate(s.createdAt)}`}
                  tabIndex={on || (active === null && i === slices.length - 1) ? 0 : -1}
                  onMouseEnter={() => setActive(i)}
                  onFocus={() => setActive(i)}
                  className={cn(
                    "relative min-w-0 flex-1 origin-bottom transition-[transform,opacity] duration-[var(--dur-tap)] ease-[var(--ease-animus-out)] focus-visible:outline-none",
                    s.colorClass,
                    on ? "scale-y-100 opacity-100" : "scale-y-[0.72] opacity-70 hover:opacity-100",
                    on && "shadow-[inset_0_0_0_1px_var(--color-lab-text)]",
                  )}
                />
              );
            })}
          </div>
        )}

        {shown ? (
          <div className="lab-num min-h-[52px] border-t border-[color:var(--animus-hairline)] pt-2.5" aria-live="polite">
            <div className="flex flex-wrap items-center gap-2">
              <span aria-hidden className={cn("h-2 w-2 rotate-45", shown.colorClass)} />
              <span className="min-w-0 truncate font-mono text-[12px] text-lab-text" title={shown.model || undefined}>
                {modelShort(shown.model) || <Nil word="None" />}
              </span>
              <Badge tone="muted">{shown.kind}</Badge>
              <span className="ml-auto font-mono text-[10px] text-lab-muted">{fmtDate(shown.createdAt)}</span>
            </div>
            <div className="mt-1 font-mono text-[12px] text-lab-text-dim">{shown.headline ?? <Nil word="None" />}</div>
          </div>
        ) : (
          <div className="mt-auto flex flex-wrap gap-x-3 gap-y-1 border-t border-[color:var(--animus-hairline)] pt-2.5">
            {LEGEND.map(([label, color]) => (
              <span key={label} className="flex items-center gap-1.5">
                <span aria-hidden className={cn("h-1.5 w-1.5 rotate-45", color)} />
                <Eyebrow className="text-[9px]">{label}</Eyebrow>
              </span>
            ))}
          </div>
        )}
      </div>
    </Panel>
  );
}
