"use client";

import Link from "next/link";
import type { LabStatus } from "@/lib/api";
import type { LiveRun, NodeSample } from "@/lib/lab-status-store";
import { fmtRate } from "@/lib/status/format";
import { Eyebrow, Nil, Panel, Sparkline, Stat, SyncRing } from "@/components/ui";
import { cn } from "@/lib/utils";

/**
 * Streams mini-board: the run's strands and aggregate while one is live (the
 * same `liveRun` the header shows), else the engine's running/waiting and the
 * endpoint rate; a 60 s sparkline of the endpoint rate underneath. Click → /streams.
 */
export function StreamsMiniBoard({
  status,
  liveRun,
  samples,
  healthy,
  className,
}: {
  status: LabStatus | null;
  liveRun: LiveRun | null;
  samples: Record<string, NodeSample[]>;
  healthy: boolean;
  className?: string;
}) {
  const serve = status?.serve;
  const engine = serve?.engine;
  const cluster = status?.cluster || serve?.cluster;
  const liveNode = cluster?.nodes?.find((n) => n.state === "serving" || n.state === "serving_worker");
  const running = liveRun?.running ?? engine?.requests_running ?? serve?.metrics?.requests_running ?? null;
  const waiting = liveRun?.waiting ?? engine?.requests_waiting ?? serve?.metrics?.requests_waiting ?? null;
  const rate = liveRun ? liveRun.tok_s : healthy ? liveNode?.gen_tok_per_s ?? null : null;
  const history = (liveNode ? samples[liveNode.id] ?? [] : []).map((s) => s.tok_s).filter((v): v is number => v !== null);

  return (
    <Link
      href="/streams"
      className={cn("group block h-full focus-visible:outline-none", className)}
      title="Open Streams"
      aria-label={`Streams: ${running ?? 0} running, ${waiting ?? 0} waiting${rate != null ? `, ${fmtRate(rate)} tok/s` : ""}`}
    >
      <Panel
        title="Streams"
        className="flex h-full flex-col transition-[border-color,box-shadow] duration-[var(--dur-tap)] group-hover:border-lab-line group-focus-visible:shadow-[inset_0_0_0_2px_var(--color-lab-line)]"
        action={
          <span className="flex items-center gap-2">
            <SyncRing state={liveRun ? "loading" : healthy ? "serving" : "idle"} label={liveRun ? `Live ${liveRun.source} run` : healthy ? "Observing endpoint" : "Idle"} />
            <Eyebrow className={liveRun ? "text-lab-target" : undefined}>{liveRun ? `${liveRun.source} run` : "endpoint"}</Eyebrow>
          </span>
        }
      >
        <div className="flex flex-1 flex-col gap-3 p-4">
          <div className="flex items-end justify-between gap-3">
            <Stat label="Aggregate" title={liveRun ? "Run aggregate (usage-calibrated)" : "Endpoint counter rate, 2 s"}>
              <div className="lab-num flex items-baseline gap-1.5 font-[family-name:var(--font-display)] text-[30px] font-bold leading-none tabular-nums text-lab-text">
                {rate != null ? fmtRate(rate) : <Nil word={healthy ? "Awaiting" : "None"} />}
                {rate != null && <span className="font-mono text-[10px] font-normal text-lab-muted">tok/s</span>}
              </div>
            </Stat>
            {liveRun && (
              <Stat label="Peak" className="items-end text-right">
                <span className="lab-num font-mono text-[13px] text-lab-target">{fmtRate(liveRun.peak)}</span>
              </Stat>
            )}
          </div>
          <Stat label="Strands" title={liveRun ? "Strands in the live run" : "Requests running / waiting on the engine"}>
            <span className="lab-num font-mono text-[13px] text-lab-text">
              {running != null || waiting != null ? (
                <>
                  {running ?? 0} <span className="text-lab-muted">running</span> · {waiting ?? 0} <span className="text-lab-muted">waiting</span>
                </>
              ) : (
                <Nil word={healthy ? "Awaiting" : "None"} />
              )}
            </span>
          </Stat>
          <div className="mt-auto">
            <div className="flex items-baseline justify-between">
              <Eyebrow className="text-[9px]">Endpoint rate · 60 s</Eyebrow>
              <Eyebrow className="text-[9px] text-lab-accent-bright transition-colors group-hover:text-lab-accent">Open Streams →</Eyebrow>
            </div>
            <Sparkline points={history} width={320} height={44} min={0} className="mt-1 w-full text-lab-line" label="Endpoint decode rate over the last 60 s" />
          </div>
        </div>
      </Panel>
    </Link>
  );
}
