"use client";

import { fmtDate, fmtSize, fmtTokS, modelShort } from "@/lib/bench/format";
import { packLabel } from "@/lib/bench/levels";
import type { HistoryEntry } from "@/lib/bench/use-run-history";
import type { StreamPack } from "@/lib/stream-run-types";
import { Eyebrow, Nil, Skeleton } from "@/components/ui";
import { cn } from "@/lib/utils";

/**
 * Run history — model · pack · levels · ×1 tok/s · peak @ ×N · date. `[` `]`
 * step through it; click loads the envelope into the result view. Controller
 * runs whose import failed are listed too, tagged, so nothing measured is lost.
 */
export function HistoryStrand({
  kind,
  entries,
  loading,
  selectedId,
  onSelect,
  packs,
  className,
}: {
  kind: "decode" | "prefill";
  entries: HistoryEntry[];
  loading: boolean;
  selectedId: string | null;
  onSelect: (e: HistoryEntry) => void;
  packs: StreamPack[];
  className?: string;
}) {
  return (
    <div className={cn("space-y-1.5", className)}>
      <div className="flex items-center justify-between gap-2">
        <Eyebrow>History</Eyebrow>
        <Eyebrow className="lab-num text-[9px]">{entries.length ? `${entries.length} · [ ] to step` : ""}</Eyebrow>
      </div>
      {loading && (
        <div className="space-y-2 p-1" aria-busy="true" aria-label="Loading history">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-8 w-full" />
          ))}
        </div>
      )}
      {!loading && !entries.length && (
        <p className="px-1 py-2 text-[12px] leading-snug text-lab-muted">
          No sequences yet. Run a {kind} sync to draw the first slice.
        </p>
      )}
      <ol className="max-h-[22rem] space-y-px overflow-y-auto">
        {entries.map((e) => {
          const on = e.id === selectedId;
          const r = e.result;
          const keys =
            r?.kind === "decode" ? r.levels.map((l) => `×${l}`).join(" ") : r?.kind === "prefill" ? r.sizes.map(fmtSize).join(" ") : null;
          return (
            <li key={e.id}>
              <button
                type="button"
                onClick={() => onSelect(e)}
                aria-pressed={on}
                className={cn(
                  "animus-notch block w-full px-3 py-2 text-left transition-[background,box-shadow] duration-[var(--dur-tap)] hover:bg-[color:var(--animus-accent-wash)]",
                  on && "bg-[color:var(--animus-accent-wash)] shadow-[inset_2px_0_0_var(--color-lab-accent)]",
                  "focus-visible:outline-none! focus-visible:shadow-[inset_0_0_0_2px_var(--color-lab-line)]!",
                )}
                title={e.model}
              >
                <div className="flex items-baseline justify-between gap-2">
                  <span className="truncate font-mono text-[11px] text-lab-text">{modelShort(e.model) || <Nil word="None" />}</span>
                  <span className="lab-num shrink-0 font-mono text-[10px] text-lab-muted">{fmtDate(e.createdAt)}</span>
                </div>
                <div className="mt-1 flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                  <span className="animus-eyebrow text-[9px]">{r ? packLabel(r.pack, packs) : e.failed ? "envelope unavailable" : "loading…"}</span>
                  {keys && <span className="lab-num font-mono text-[10px] text-lab-muted">{keys}</span>}
                  {e.source === "controller" && <span className="animus-eyebrow text-[8px] text-lab-warn!">not in index</span>}
                </div>
                <div className="lab-num mt-1 flex items-baseline gap-3 font-mono text-[11px]">
                  {kind === "decode" ? (
                    <>
                      <span className="text-lab-text-dim">
                        ×1 {e.headline.c1 !== null ? fmtTokS(e.headline.c1) : <Nil word="None" />}
                      </span>
                      <span className="text-lab-text">
                        {e.headline.peak !== null ? (
                          <>
                            {fmtTokS(e.headline.peak)} <span className="text-lab-muted">tok/s</span>
                            {e.headline.peakAt !== null ? <span className="text-lab-target"> @ ×{e.headline.peakAt}</span> : null}
                          </>
                        ) : (
                          <Nil word="None" />
                        )}
                      </span>
                    </>
                  ) : (
                    <span className="text-lab-text">
                      {e.headline.sustained !== null ? (
                        <>
                          {fmtTokS(e.headline.sustained)} <span className="text-lab-muted">tok/s sustained</span>
                        </>
                      ) : (
                        <Nil word="None" />
                      )}
                    </span>
                  )}
                </div>
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
