"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { etaDecodeMs, fmtEta, reviseEta, type MeasuredLevel } from "@/lib/bench/eta";
import { fmtDuration, fmtMs, fmtPct, fmtTokS } from "@/lib/bench/format";
import { interpretDecode } from "@/lib/bench/interpret";
import type { DecodeConfig } from "@/lib/bench/levels";
import { currentLevelIndex, rollupLevel, type HardwareSeries } from "@/lib/bench/live";
import { c1Arm, peakArm, type DecodeResult } from "@/lib/bench/result";
import type { StreamRunState } from "@/lib/use-stream-run";
import { Callout, Eyebrow, Nil, SyncRing } from "@/components/ui";
import { cn } from "@/lib/utils";
import { DecodeCharts, type GhostArms } from "./DecodeCharts";
import { DecodeDetails } from "./DetailsTable";
import { Gauge } from "./Gauge";
import { HardwareStrip } from "./HardwareStrip";
import { LevelRings } from "./LevelRings";
import { ResultHero, usePayoff, type Secondary } from "./Payoff";
import { StrandLanes } from "./StrandLanes";
import { TakeawayRow } from "./TakeawayRow";

const NIL = <Nil word="None" />;

export type InstrumentStatus = "idle" | "starting" | "running" | "done" | "cancelled" | "error";

/** The right column for Decode: run phase, then the result — one room, not a modal. */
export function DecodeInstrument({
  cfg,
  status,
  live,
  result,
  ghost,
  fromHistory,
  now,
  hardware,
  errorMessage,
  onRunLevels,
  onRunAgain,
  detailsOpen,
  onToggleDetails,
  copyRequest,
  busy,
}: {
  cfg: DecodeConfig;
  status: InstrumentStatus;
  live: StreamRunState | null;
  result: DecodeResult | null;
  ghost: GhostArms;
  fromHistory: boolean;
  /** wall clock, ticking while a run is live */
  now: number;
  hardware: HardwareSeries[];
  errorMessage: string | null;
  onRunLevels: (levels: number[]) => void;
  onRunAgain: () => void;
  detailsOpen: boolean;
  onToggleDetails: () => void;
  copyRequest: number;
  busy: boolean;
}) {
  const running = status === "running" || status === "starting";
  const levels = live?.hello?.levels ?? result?.levels ?? cfg.levels;
  const maxTokens = live?.hello?.max_tokens ?? result?.maxTokens ?? cfg.maxTokens;
  const rows = live?.levels ?? [];
  const cur = currentLevelIndex(rows.length, levels.length);
  const curStrands = useMemo(() => (live ? live.strands.filter((s) => s.level === cur) : []), [live, cur]);
  const roll = rollupLevel(curStrands, maxTokens);

  const startedAt = live?.hello ? Date.parse(live.hello.started_at) : null;
  const elapsedMs = running && startedAt ? Math.max(0, now - startedAt) : (result?.durationMs ?? live?.done?.summary.duration_ms ?? null);

  // ETA: falls freely, rises only past hysteresis.
  const [eta, setEta] = useState<number | null>(null);
  const etaKey = live?.hello?.run_id ?? null;
  const lastKey = useRef<string | null>(null);
  useEffect(() => {
    if (etaKey !== lastKey.current) {
      lastKey.current = etaKey;
      setEta(null);
    }
  }, [etaKey]);
  useEffect(() => {
    if (!running || !live?.hello) return;
    const measured: MeasuredLevel[] = rows
      .filter((r) => !r.skipped)
      .map((r) => ({ concurrency: r.concurrency ?? 1, perStream: r.per_stream_median_tok_s, ttftMs: r.ttft_p50_ms }));
    const next = etaDecodeMs({
      levels,
      maxTokens,
      measured,
      current:
        cur < levels.length && rows.length < levels.length
          ? { concurrency: levels[cur], tokens: roll.tokens, rateTokS: roll.medianRate, waitingFirstToken: roll.waitingFirstToken }
          : null,
    });
    setEta((shown) => reviseEta(shown, next));
    // now drives the tick; rows/roll derive from live
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [now, running, rows.length, live?.hello]);

  // Payoff plays once per completed live run; history and cancelled results land settled.
  const payoffKey = result ? `${result.id ?? "live"}:${status}` : null;
  const stage = usePayoff(payoffKey, status === "done" && !fromHistory);
  const hitstop = stage === "hitstop";

  const interp = useMemo(() => (result ? interpretDecode(result.arms, { floor: cfg.floor, sloMs: cfg.sloMs }) : null), [result, cfg.floor, cfg.sloMs]);
  const peak = result ? peakArm(result.arms) : null;
  const c1 = result ? c1Arm(result.arms) : null;

  const showResult = !!result && (status === "done" || status === "cancelled" || status === "error" || fromHistory);
  const liveAgg = live?.latest;
  const settledAgg = rows.length ? (rows[rows.length - 1].aggregate_tok_s ?? null) : null;

  const secondaries: Secondary[] = result
    ? [
        { label: "×1 per stream", value: c1?.perStream !== null && c1?.perStream !== undefined ? `${fmtTokS(c1.perStream)} tok/s` : NIL },
        {
          label: interp?.topLevel ? `efficiency @ ×${interp.topLevel}` : "efficiency",
          value: interp?.efficiency !== null && interp?.efficiency !== undefined ? fmtPct(interp.efficiency) : NIL,
          tone: interp?.efficiency !== null && interp?.efficiency !== undefined && interp.efficiency < 0.5 ? "muted" : undefined,
        },
        {
          label: "interactive up to",
          value: interp?.bestInteractive ? `×${interp.bestInteractive.concurrency}` : NIL,
          tone: "target",
        },
        { label: "TTFT p50 @ peak", value: peak?.ttftP50 !== null && peak?.ttftP50 !== undefined ? fmtMs(peak.ttftP50) : NIL },
        { label: "knee", value: interp?.knee ? `×${interp.knee}` : "none", tone: interp?.knee ? undefined : "muted" },
        { label: "duration", value: fmtDuration(result.durationMs), tone: "muted" },
      ]
    : [];

  const header = (() => {
    switch (status) {
      case "starting":
        return { title: "Synchronizing…", ring: "loading" as const };
      case "running":
        return { title: "Synchronizing…", ring: "loading" as const };
      case "done":
        return { title: `Synchronized · ${fmtDuration(elapsedMs)}`, ring: "serving" as const };
      case "cancelled":
        return { title: "Stopped · partial sequence kept", ring: "offline" as const };
      case "error":
        return { title: "Failed", ring: "offline" as const };
      default:
        return fromHistory && result
          ? { title: `Sequence · ${result.createdAt ? new Date(result.createdAt).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" }) : ""}`, ring: "idle" as const }
          : { title: "Ready", ring: "idle" as const };
    }
  })();

  const step = running && live?.hello ? `Level ${Math.min(cur + 1, levels.length)} of ${levels.length} · ×${levels[cur]} · ${maxTokens} tok/stream` : null;

  return (
    <div className={cn("space-y-5", hitstop && "bench-hitstop")} aria-live="polite" aria-busy={running}>
      <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <div className="flex items-center gap-2.5">
          <SyncRing state={header.ring} size={14} label={header.title} />
          <h2 className="font-[family-name:var(--font-display)] text-[15px] font-semibold tracking-tight text-lab-text">{header.title}</h2>
        </div>
        <div className="lab-num flex flex-wrap items-baseline gap-x-3 font-mono text-[11px] text-lab-text-dim">
          {step && <span>{step}</span>}
          {running && elapsedMs !== null && <span>{fmtDuration(elapsedMs)}</span>}
          {running && eta !== null && eta > 0 && <span className="text-lab-muted">ETA {fmtEta(eta)}</span>}
          {running && roll.total > 0 && roll.errors > 0 && (
            <span className="text-lab-danger">
              {roll.total - roll.errors}/{roll.total} strands
            </span>
          )}
        </div>
      </header>

      {status === "error" && errorMessage && <Callout tone="danger">{errorMessage}</Callout>}

      {(running || (live && !fromHistory && status !== "idle")) && (
        <LevelRings keys={levels} kind="decode" rows={rows} current={cur} running={running} fill={roll.fill} segments={Math.max(1, curStrands.length || levels[cur] || 1)} />
      )}

      {running && (
        <div className={cn("grid gap-4", curStrands.length > 1 && "lg:grid-cols-[minmax(0,1fr)_auto]")}>
          <StrandLanes strands={curStrands} maxTokens={maxTokens} />
          {curStrands.length > 1 && (
            <Gauge value={liveAgg?.tok_s ?? 0} peak={liveAgg?.peak_tok_s ?? 0} settled={settledAgg} className="justify-self-center lg:justify-self-end" />
          )}
        </div>
      )}

      {showResult && result && interp && (
        <ResultHero
          stage={stage}
          value={peak?.aggregate ?? null}
          format={(n) => fmtTokS(n)}
          unit="tok/s"
          at={peak ? `peak aggregate @ ×${peak.concurrency}` : "peak aggregate"}
          secondaries={secondaries}
          sentence={interp.sentence}
        />
      )}

      {(running || showResult) && (
        <DecodeCharts
          arms={result?.arms ?? []}
          levels={levels}
          ghost={showResult ? ghost : null}
          knee={showResult ? interp?.knee : null}
          floor={cfg.floor}
          sloMs={cfg.sloMs}
        />
      )}

      {(running || (hardware.length > 0 && status !== "idle" && !fromHistory)) && <HardwareStrip series={hardware} />}

      {showResult && result && (
        <>
          <TakeawayRow
            result={result}
            suggest={interp && !interp.saturated ? interp.suggest : undefined}
            onRunLevels={onRunLevels}
            onRunAgain={onRunAgain}
            detailsOpen={detailsOpen}
            onToggleDetails={onToggleDetails}
            disabled={busy}
            copyRequest={copyRequest}
          />
          {detailsOpen && <DecodeDetails arms={result.arms} className="bench-fade" />}
        </>
      )}

      {status === "idle" && !result && (
        <div className="flex min-h-[18rem] flex-col items-center justify-center gap-2 border border-dashed border-lab-border-subtle text-center">
          <Eyebrow>No sequence drawn</Eyebrow>
          <p className="max-w-[40ch] text-[12px] leading-snug text-lab-muted">
            Pick a pack and the levels, then Run. The curve draws itself as each level settles; the hero is the peak aggregate.
          </p>
        </div>
      )}
    </div>
  );
}
