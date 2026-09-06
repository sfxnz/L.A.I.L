"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { RunRow } from "@/lib/api";
import { fmtDate, fmtPct, fmtTokS, modelShort } from "@/lib/bench/format";
import { lastSync, latestDecodeRuns } from "@/lib/bench/last-sync";
import { packLabel } from "@/lib/bench/levels";
import type { BenchResult } from "@/lib/bench/result";
import { loadEntry } from "@/lib/bench/use-run-history";
import { Badge, Eyebrow, HeroNumber, Nil, Panel, Skeleton, btnClass } from "@/components/ui";
import { cn } from "@/lib/utils";
import { MiniCurve } from "./DecodeCharts";

/** Status → "Last synchronization": the latest decode run's hero, curve, delta and two ways back into /bench. */
export function LastSyncCard({ runs, loading, className }: { runs: RunRow[]; loading: boolean; className?: string }) {
  const [cur, prev] = latestDecodeRuns(runs);
  const [envelopes, setEnvelopes] = useState<{ id: string | null; current: BenchResult | null; previous: BenchResult | null }>({ id: null, current: null, previous: null });
  useEffect(() => {
    if (!cur) return;
    let alive = true;
    Promise.all([loadEntry(cur.run_id, "engine").catch(() => null), prev ? loadEntry(prev.run_id, "engine").catch(() => null) : null]).then(([c, p]) => {
      if (alive) setEnvelopes({ id: cur.run_id, current: c?.result ?? null, previous: p?.result ?? null });
    });
    return () => {
      alive = false;
    };
  }, [cur?.run_id, prev?.run_id]); // eslint-disable-line react-hooks/exhaustive-deps

  const sync = lastSync(runs, envelopes.id === cur?.run_id ? envelopes : { current: null, previous: null });

  return (
    <Panel title="Last synchronization" padded className={className} action={sync ? <Eyebrow className="lab-num">{fmtDate(sync.createdAt)}</Eyebrow> : undefined}>
      {loading && !sync ? (
        <div className="space-y-3" aria-busy="true" aria-label="Loading last synchronization">
          <Skeleton className="h-12 w-40" />
          <Skeleton className="h-20 w-60" />
        </div>
      ) : !sync ? (
        <div className="flex flex-col items-start gap-3 py-2">
          <Eyebrow>No sequences yet</Eyebrow>
          <p className="max-w-[44ch] text-[13px] leading-snug text-lab-text-dim">No sequences yet. Run a decode sync to draw the first slice.</p>
          <Link href="/bench" className={btnClass("primary", "sm")}>
            Open Bench
          </Link>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_auto] md:items-end">
          <div className="min-w-0">
            <Eyebrow>{sync.peakAt !== null ? `peak aggregate @ ×${sync.peakAt}` : "peak aggregate"}</Eyebrow>
            <div className="mt-1 flex flex-wrap items-baseline gap-x-2 gap-y-1">
              {sync.peak !== null ? (
                <HeroNumber value={sync.peak} format={(n) => fmtTokS(n)} label={`${fmtTokS(sync.peak)} tok/s`} />
              ) : (
                <span className="animus-hero">
                  <Nil word="None" />
                </span>
              )}
              <span className="font-[family-name:var(--font-display)] text-[13px] font-semibold uppercase tracking-[0.14em] text-lab-text-dim">tok/s</span>
              {sync.delta !== null && (
                <Badge tone={sync.delta >= 0 ? "ok" : "warn"}>{`${fmtPct(sync.delta, true)} vs previous`}</Badge>
              )}
            </div>
            <dl className="lab-num mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-lab-text-dim">
              <div>
                <dt className="inline text-lab-muted">×1 </dt>
                <dd className="inline">{sync.c1 !== null ? `${fmtTokS(sync.c1)} tok/s` : <Nil word="None" />}</dd>
              </div>
              <div className="min-w-0 truncate">
                <dt className="inline text-lab-muted">model </dt>
                <dd className="inline" title={sync.model}>
                  {modelShort(sync.model) || <Nil word="None" />}
                </dd>
              </div>
              <div>
                <dt className="inline text-lab-muted">pack </dt>
                <dd className="inline">{sync.pack ? packLabel(sync.pack) : <Nil word="None" />}</dd>
              </div>
              {sync.levels.length > 0 && (
                <div>
                  <dt className="inline text-lab-muted">levels </dt>
                  <dd className="inline">{sync.levels.map((l) => `×${l}`).join(" ")}</dd>
                </div>
              )}
            </dl>
            <div className="mt-3 flex flex-wrap gap-2">
              <Link href={sync.runAgainHref} className={btnClass("primary", "sm")}>
                Run again
              </Link>
              <Link href={sync.openHref} className={btnClass("secondary", "sm")}>
                Open
              </Link>
            </div>
          </div>
          <div className={cn("justify-self-start md:justify-self-end", !sync.arms.length && "opacity-50")} aria-hidden={!sync.arms.length}>
            {sync.arms.length ? <MiniCurve arms={sync.arms} ghost={sync.previousArms} /> : <Skeleton className="h-20 w-60" />}
          </div>
        </div>
      )}
    </Panel>
  );
}
