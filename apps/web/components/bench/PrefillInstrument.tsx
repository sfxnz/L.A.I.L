"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { etaPrefillMs, fmtEta, predictTtftForSize, reviseEta, type MeasuredSize } from "@/lib/bench/eta";
import { fmtDuration, fmtMs, fmtPct, fmtSize, fmtTokS } from "@/lib/bench/format";
import { interpretPrefill } from "@/lib/bench/interpret";
import type { PrefillConfig } from "@/lib/bench/levels";
import type { HardwareSeries } from "@/lib/bench/live";
import { sustainedArm, type PrefillResult } from "@/lib/bench/result";
import type { StrandView, StreamRunState } from "@/lib/use-stream-run";
import { Callout, Eyebrow, Nil, SyncRing } from "@/components/ui";
import { cn } from "@/lib/utils";
import { useNow } from "./Clock";
import type { InstrumentStatus } from "./DecodeInstrument";
import { PrefillDetails } from "./DetailsTable";
import { HardwareStrip } from "./HardwareStrip";
import { LevelRings } from "./LevelRings";
import { ResultHero, usePayoff, type Secondary } from "./Payoff";
import { PrefillCharts, PrefillFillBars } from "./PrefillCharts";
import { TakeawayRow } from "./TakeawayRow";

const NIL = <Nil word="None" />;

/** Wall-clock stamp of each strand's first `prefill` state, as this client saw it. */
function usePrefillStarts(strands: StrandView[]): Map<number, number> {
  const starts = useRef(new Map<number, number>());
  const now = Date.now();
  for (const s of strands) {
    if ((s.state === "prefill" || s.state === "decode") && !starts.current.has(s.i)) starts.current.set(s.i, now);
    if (s.state === "waiting") starts.current.delete(s.i);
  }
  return starts.current;
}

export function PrefillInstrument({
  cfg,
  status,
  live,
  result,
  fromHistory,
  hardware,
  errorMessage,
  onRunAgain,
  detailsOpen,
  onToggleDetails,
  copyRequest,
  busy,
}: {
  cfg: PrefillConfig;
  status: InstrumentStatus;
  live: StreamRunState | null;
  result: PrefillResult | null;
  fromHistory: boolean;
  hardware: HardwareSeries[];
  errorMessage: string | null;
  onRunAgain: () => void;
  detailsOpen: boolean;
  onToggleDetails: () => void;
  copyRequest: number;
  busy: boolean;
}) {
  const running = status === "running" || status === "starting";
  // The fill bars and the TTFT stopwatch animate: this instrument owns its clock.
  const now = useNow(running);
  const sizes = live?.hello?.sizes ?? result?.sizes ?? cfg.sizes;
  const rows = live?.levels ?? [];
  // Skipped sizes report immediately, so "current" is the first size without a row.
  const firstOpen = sizes.findIndex((_, i) => !rows.some((r) => r.index === i));
  const cur = firstOpen < 0 ? Math.max(0, sizes.length - 1) : firstOpen;
  const strands = live?.strands ?? [];
  const starts = usePrefillStarts(strands);
  // Each size runs `samples` requests in turn: per size, show the one in flight (else the last).
  const shown = useMemo(() => {
    const out: StrandView[] = [];
    for (let i = 0; i < sizes.length; i++) {
      const own = strands.filter((s) => s.level === i);
      const open = own.find((s) => s.state === "waiting" || s.state === "prefill" || s.state === "decode");
      const pick = own.find((s) => s.state === "prefill" || s.state === "decode") ?? (own.every((s) => s.state === "waiting") ? open : own.filter((s) => s.state !== "waiting").pop());
      if (pick) out.push(pick);
    }
    return out;
  }, [strands, sizes.length]);
  const curStrand = shown.find((s) => s.level === cur);
  const curStart = curStrand ? (starts.get(curStrand.i) ?? null) : null;
  const curRequestsLeft = strands.filter((s) => s.level === cur && s.state === "waiting" && s !== curStrand).length;

  const measured: MeasuredSize[] = rows.filter((r) => !r.skipped && r.ok > 0).map((r) => ({ size: r.size ?? 0, ttftMs: r.ttft_p50_ms }));
  const skipped = new Set(rows.filter((r) => r.skipped).map((r) => r.size ?? 0));
  const predicted = running ? predictTtftForSize(measured, sizes[cur] ?? 0) : null;

  const startedAt = live?.hello ? Date.parse(live.hello.started_at) : null;
  const elapsedMs = running && startedAt ? Math.max(0, now - startedAt) : (result?.durationMs ?? live?.done?.summary.duration_ms ?? null);
  const samples = live?.hello?.samples ?? 1;

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
    const next = etaPrefillMs({
      sizes,
      measured,
      skipped,
      samples,
      current: firstOpen >= 0 ? { size: sizes[cur], elapsedMs: curStart !== null ? now - curStart : 0, requestsLeft: curRequestsLeft } : null,
    });
    setEta((shown) => reviseEta(shown, next));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [now, running, rows.length, live?.hello]);

  const payoffKey = result ? `${result.id ?? "live"}:${status}` : null;
  const stage = usePayoff(payoffKey, status === "done" && !fromHistory);
  const hitstop = stage === "hitstop";

  const interp = useMemo(() => (result ? interpretPrefill(result.arms) : null), [result]);
  const sustained = result ? sustainedArm(result.arms) : null;
  const showResult = !!result && (status === "done" || status === "cancelled" || status === "error" || fromHistory);

  const lastDoubling = interp?.doublings.length ? interp.doublings[interp.doublings.length - 1] : null;
  const secondaries: Secondary[] = result
    ? [
        {
          label: sustained ? `${fmtSize(sustained.size)} in` : "largest size",
          value: sustained?.ttftMs !== null && sustained?.ttftMs !== undefined ? fmtDuration(sustained.ttftMs) : NIL,
        },
        { label: "peak prefill", value: interp?.peak?.prefillTokS ? `${fmtTokS(interp.peak.prefillTokS)} tok/s` : NIL, tone: "target" },
        { label: "held within", value: interp?.hold !== null && interp?.hold !== undefined ? fmtPct(interp.hold) : NIL },
        { label: lastDoubling ? `TTFT per doubling @ ${fmtSize(lastDoubling.size)}` : "TTFT per doubling", value: lastDoubling ? `×${lastDoubling.ratio.toFixed(2)}` : NIL },
        { label: "skipped", value: interp?.skipped.length ? interp.skipped.map((a) => fmtSize(a.size)).join(" ") : "none", tone: "muted" },
        { label: "duration", value: fmtDuration(result.durationMs), tone: "muted" },
      ]
    : [];

  const header = (() => {
    switch (status) {
      case "starting":
      case "running":
        return { title: "Running…", ring: "loading" as const };
      case "done":
        return { title: `Done · ${fmtDuration(elapsedMs)}`, ring: "serving" as const };
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

  const step =
    running && live?.hello && firstOpen >= 0
      ? `${fmtSize(sizes[cur])} · Level ${cur + 1} of ${sizes.length}${samples > 1 && curStrand?.wave !== undefined ? ` · request ${curStrand.wave + 1}/${samples}` : ""}`
      : null;
  const stopwatch = running && curStart !== null && curStrand && curStrand.state === "prefill" ? now - curStart : null;

  return (
    <div className={cn("space-y-5", hitstop && "bench-hitstop")} aria-live="polite" aria-busy={running}>
      <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <div className="flex items-center gap-2.5">
          <SyncRing state={header.ring} size={14} label={header.title} />
          <h2 className="font-[family-name:var(--font-display)] text-[15px] font-semibold tracking-tight text-lab-text">{header.title}</h2>
        </div>
        <div className="lab-num flex flex-wrap items-baseline gap-x-3 font-mono text-[11px] text-lab-text-dim">
          {step && <span>{step}</span>}
          {stopwatch !== null && <span className="text-lab-text">TTFT {fmtMs(stopwatch)}</span>}
          {running && elapsedMs !== null && <span>{fmtDuration(elapsedMs)}</span>}
          {running && eta !== null && eta > 0 && <span className="text-lab-muted">ETA {fmtEta(eta)}</span>}
        </div>
      </header>

      {status === "error" && errorMessage && <Callout tone="danger">{errorMessage}</Callout>}

      {(running || (live && !fromHistory && status !== "idle")) && (
        <LevelRings keys={sizes} kind="prefill" rows={rows} current={cur} running={running} fill={predicted ? Math.min(0.96, (curStart !== null ? now - curStart : 0) / predicted) : 0} segments={8} />
      )}

      {running && (
        <PrefillFillBars sizes={sizes} rows={rows} strands={shown} current={cur} predicted={predicted} now={now} startedAt={curStart} />
      )}

      {showResult && result && interp && (
        <ResultHero
          stage={stage}
          value={sustained?.prefillTokS ?? null}
          format={(n) => fmtTokS(n)}
          unit="tok/s"
          at={sustained ? `sustained prefill @ ${fmtSize(sustained.size)}` : "sustained prefill"}
          secondaries={secondaries}
          sentence={interp.sentence}
        />
      )}

      {(running || showResult) && <PrefillCharts arms={result?.arms ?? []} sizes={sizes} />}

      {(running || (hardware.length > 0 && status !== "idle" && !fromHistory)) && <HardwareStrip series={hardware} />}

      {showResult && result && (
        <>
          <TakeawayRow
            result={result}
            onRunAgain={onRunAgain}
            detailsOpen={detailsOpen}
            onToggleDetails={onToggleDetails}
            disabled={busy}
            copyRequest={copyRequest}
          />
          {detailsOpen && <PrefillDetails arms={result.arms} className="bench-fade" />}
        </>
      )}

      {status === "idle" && !result && (
        <div className="flex min-h-[18rem] flex-col items-center justify-center gap-2 border border-dashed border-lab-border-subtle text-center">
          <Eyebrow>No sequence drawn</Eyebrow>
          <p className="max-w-[40ch] text-[12px] leading-snug text-lab-muted">
            Pick the context sizes, then Run. Each size fills a bar against its predicted TTFT; the hero is the sustained prefill rate at the largest size.
          </p>
        </div>
      )}
    </div>
  );
}
