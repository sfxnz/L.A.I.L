"use client";

import { Badge, Eyebrow, HeroNumber, Nil, Stat, SyncRing, Tick, type SyncState } from "@/components/ui";
import type { StreamRunState } from "@/lib/use-stream-run";
import { aggregateMethod } from "@/lib/streams/export";
import { fmtDuration, fmtInt, fmtMs, fmtRate } from "@/lib/streams/format";
import { cn } from "@/lib/utils";

function Cell({ label, children, title, className }: { label: string; children: React.ReactNode; title?: string; className?: string }) {
  return (
    <Stat label={label} title={title} mono className={cn("gap-0.5", className)}>
      {children}
    </Stat>
  );
}

/**
 * The instrument bar: ONE aggregate number with its method chip, then peak,
 * tokens, strand counts, TTFT p50/p95, elapsed. Live and final aggregate are the
 * same metric (window rate → its run average, the decode span), so the payoff
 * count-up from the last live value does not drop to a different definition;
 * wall-clock goodput (incl. TTFT) is its own cell once the run is done.
 */
export function InstrumentBar({ state, live, className }: { state: StreamRunState; live: boolean; className?: string }) {
  const latest = state.latest;
  const sum = state.done?.summary ?? null;
  const { value, method } = aggregateMethod(state);
  const ring: SyncState | null = state.error || sum?.status === "error" ? "offline" : sum ? "serving" : live ? "loading" : latest ? "idle" : null;
  const ringLabel = state.error ? "Run failed" : sum ? (sum.status === "cancelled" ? "Run cancelled" : "Run done") : live ? "Running" : "Idle";
  const peak = sum?.peak_tok_s ?? latest?.peak_tok_s;
  const tokens = sum?.tokens ?? latest?.tokens;
  // Once the run is done its summary is the definition (a null p95 = too few samples),
  // never a fallback to the live pooled value.
  const p50 = sum ? sum.ttft_p50_ms : latest?.ttft_p50_ms;
  const p95 = sum ? sum.ttft_p95_ms : latest?.ttft_p95_ms;
  const elapsed = sum?.duration_ms ?? latest?.t_ms;
  const lastLive = state.agg.length ? state.agg[state.agg.length - 1].tok_s : 0;

  return (
    <div
      className={cn(
        "streams-instrument lab-card animus-bracketed flex flex-wrap items-center gap-x-5 gap-y-2 px-4 py-2.5",
        "before:top-[3px]! before:left-[3px]! after:right-[3px]! after:bottom-[3px]!",
        className,
      )}
      aria-live="polite"
      aria-atomic="false"
    >
      <div className="flex items-center gap-3">
        <SyncRing state={ring} label={ringLabel} />
        <div className="flex flex-col gap-0.5">
          <Eyebrow>Aggregate</Eyebrow>
          <div className="flex items-baseline gap-2">
            {sum && value !== null ? (
              <HeroNumber value={value} from={lastLive} format={(v) => fmtRate(v)} className="text-[30px]! leading-none! text-lab-text" label="Aggregate tok/s" />
            ) : (
              <span className="lab-num font-[family-name:var(--font-display)] text-[30px] leading-none font-bold text-lab-text">
                {value === null ? <Nil /> : fmtRate(value)}
              </span>
            )}
            <span className="font-mono text-[11px] text-lab-muted">tok/s</span>
          </div>
        </div>
        <Badge tone={sum ? "ok" : "muted"}>{method}</Badge>
      </div>

      <Tick className="hidden h-7 sm:block" />

      <Cell label="Peak" title="Highest live window rate once a full window was decoding">
        {peak === undefined ? <Nil /> : <span className="text-lab-target">{fmtRate(peak)}</span>}
      </Cell>
      {sum && (
        <Cell label="Goodput" title="Wall-clock: Σ tokens ÷ (last token − first request), TTFT included">
          {sum.aggregate_tok_s === null ? <Nil /> : fmtRate(sum.aggregate_tok_s)}
        </Cell>
      )}
      <Cell label="Tokens" title="Σ completion tokens (per-chunk usage; chunk count when the server sends none)">
        {tokens === undefined ? <Nil /> : fmtInt(tokens)}
      </Cell>
      <Cell label="Strands" title="streaming · waiting · done">
        {latest ? (
          <>
            {latest.running} <span className="text-lab-muted">streaming</span> · {latest.waiting} <span className="text-lab-muted">waiting</span> · {latest.done}{" "}
            <span className="text-lab-muted">done</span>
          </>
        ) : (
          <Nil />
        )}
      </Cell>
      <Cell label="TTFT p50 / p95" title="Time to first token across strands this run">
        {p50 === undefined || p50 === null ? <Nil /> : `${fmtMs(p50)} / ${fmtMs(p95)}`}
      </Cell>
      <Cell label="Elapsed" className="ml-auto text-right" title="Run clock">
        {elapsed === undefined ? <Nil /> : fmtDuration(elapsed)}
      </Cell>
    </div>
  );
}
