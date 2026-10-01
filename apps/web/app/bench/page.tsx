"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, parseApiError } from "@/lib/api";
import { isEditableTarget } from "@/lib/shortcuts";
import { isUnauthorizedError } from "@/lib/auth-token";
import { useShallow } from "zustand/react/shallow";
import { serveHealthy, useLabStatusStore } from "@/lib/lab-status-store";
import type { StreamPack, StreamRunRow } from "@/lib/stream-run-types";
import { useStreamRun } from "@/lib/use-stream-run";
import {
  decodeConfigFromQuery,
  decodeConfigToQuery,
  prefillConfigFromQuery,
  prefillConfigToQuery,
  presetForKey,
  sortConcurrencies,
  toggleLevel,
  type DecodeConfig,
  type PrefillConfig,
} from "@/lib/bench/levels";
import { fmtDate } from "@/lib/bench/format";
import { decodeResultFromLive, prefillResultFromLive, type BenchResult } from "@/lib/bench/result";
import { loadEntry, usePreviousRun, useRunHistory, type HistoryEntry } from "@/lib/bench/use-run-history";
import { Callout, PageSkeleton, Panel, SegmentedControl } from "@/components/ui";
import { DecodeConfigPanel, PrefillConfigPanel } from "@/components/bench/ConfigPanels";
import { DecodeInstrument, type InstrumentStatus } from "@/components/bench/DecodeInstrument";
import { useHardwareSamples } from "@/components/bench/HardwareStrip";
import { HistoryStrand } from "@/components/bench/HistoryStrand";
import { PrefillInstrument } from "@/components/bench/PrefillInstrument";

/*
  /bench — one room, two instruments. Left: configuration and the history
  strand. Right: the run as it happens, then the result with its payoff.
  Every number is the controller engine's `level` / `done` output, shaped by
  lib/bench; the page only sequences state and keys.
*/

type Tab = "decode" | "prefill";
type Notice = { tone: "warn" | "danger" | "muted"; title: string; body?: string };

export default function BenchPage() {
  return (
    <Suspense fallback={<PageSkeleton rows={4} />}>
      <BenchRoom />
    </Suspense>
  );
}

function BenchRoom() {
  const router = useRouter();
  const pathname = usePathname();
  const search = useSearchParams();

  const { healthy, needToken, unreachable } = useLabStatusStore(
    useShallow((s) => ({ healthy: serveHealthy(s.status), needToken: s.needToken, unreachable: s.unreachable })),
  );
  const [tab, setTab] = useState<Tab>(() => (search.get("tab") === "prefill" ? "prefill" : "decode"));
  const [decodeCfg, setDecodeCfg] = useState<DecodeConfig>(() => decodeConfigFromQuery(search));
  const [prefillCfg, setPrefillCfg] = useState<PrefillConfig>(() => prefillConfigFromQuery(search));
  const [packs, setPacks] = useState<StreamPack[]>([]);

  // Live run
  const [runId, setRunId] = useState<string | null>(null);
  const [runMeta, setRunMeta] = useState<{ tab: Tab; pack: string; levels: number[]; sizes: number[]; maxTokens: number } | null>(null);
  const { state } = useStreamRun(runId);
  const [starting, setStarting] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);

  // History and the selected past run
  const history = useRunHistory(tab);
  const [selected, setSelected] = useState<HistoryEntry | null>(null);
  const [selectedResult, setSelectedResult] = useState<BenchResult | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [copyRequest, setCopyRequest] = useState(0);

  const attached = !!runId;
  const status: InstrumentStatus = !attached
    ? "idle"
    : state.error
      ? "error"
      : state.done
        ? state.done.summary.status === "cancelled"
          ? "cancelled"
          : state.done.summary.status === "error"
            ? "error"
            : "done"
        : state.hello
          ? "running"
          : "starting";
  const running = status === "running" || status === "starting";

  const hardware = useHardwareSamples(running, runId);

  const canRun = healthy && !needToken && !unreachable && !running;
  const reason = needToken
    ? "LAIL_TOKEN required — paste it in the banner"
    : unreachable
      ? "Controller unreachable"
      : !healthy
        ? "No model served"
        : undefined;

  // ── Boot: packs, `?run=`, re-attach to a live bench run ─────────────────
  useEffect(() => {
    api.streamPacks().then(setPacks).catch(() => {});
  }, []);
  const urlRun = search.get("run");
  useEffect(() => {
    if (!urlRun) return;
    let alive = true;
    (async () => {
      let entry: HistoryEntry | null = null;
      try {
        const l = await loadEntry(urlRun, "engine");
        if (l.result) entry = { id: urlRun, source: "engine", createdAt: l.result.createdAt ?? "", model: l.result.model, headline: { c1: null, peak: null, peakAt: null, sustained: null }, result: l.result, envelope: l.envelope, failed: false };
      } catch {
        try {
          const l = await loadEntry(urlRun, "controller");
          if (l.result) entry = { id: urlRun, source: "controller", createdAt: l.result.createdAt ?? "", model: l.result.model, headline: { c1: null, peak: null, peakAt: null, sustained: null }, result: l.result, envelope: null, failed: false };
        } catch {
          /* fall through */
        }
      }
      if (!alive) return;
      if (!entry?.result) {
        setNotice({ tone: "warn", title: "Run not found", body: `No run ${urlRun} in the index or on the controller.` });
        return;
      }
      setTab(entry.result.kind);
      setSelected(entry);
      setSelectedResult(entry.result);
    })();
    return () => {
      alive = false;
    };
  }, [urlRun]);
  useEffect(() => {
    if (urlRun) return;
    api
      .listStreamRuns()
      .then((rows) => {
        const live = rows.find((r) => r.status === "running" && r.mode.startsWith("bench-"));
        if (live) attachRow(live);
      })
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function attachRow(row: StreamRunRow) {
    const t: Tab = row.mode === "bench-prefill" ? "prefill" : "decode";
    setTab(t);
    setRunMeta({ tab: t, pack: row.pack, levels: [], sizes: [], maxTokens: 0 });
    setSelected(null);
    setSelectedResult(null);
    setDetailsOpen(false);
    setRunId(row.run_id);
  }

  // ── Query sync (replace, no history entries) ──────────────────────────────
  const syncQuery = useCallback(
    (t: Tab, d: DecodeConfig, p: PrefillConfig) => {
      const q = t === "decode" ? decodeConfigToQuery(d) : prefillConfigToQuery(p);
      router.replace(`${pathname}?${q}`, { scroll: false });
    },
    [router, pathname],
  );
  function changeTab(t: Tab) {
    if (t === tab) return;
    setTab(t);
    if (!running) {
      setSelected(null);
      setSelectedResult(null);
    }
    setDetailsOpen(false);
    syncQuery(t, decodeCfg, prefillCfg);
  }

  // ── Run / Stop ────────────────────────────────────────────────────────────
  const start = useCallback(
    async (t: Tab, d: DecodeConfig, p: PrefillConfig) => {
      if (running || starting) return;
      setStarting(true);
      setNotice(null);
      try {
        const body =
          t === "decode"
            ? { mode: "bench-decode" as const, pack: d.pack, levels: d.levels, max_tokens: d.maxTokens }
            : { mode: "bench-prefill" as const, pack: "prose", sizes: p.sizes };
        const { run_id } = await api.startStreamRun(body);
        setRunMeta({ tab: t, pack: body.pack, levels: d.levels, sizes: p.sizes, maxTokens: d.maxTokens });
        setSelected(null);
        setSelectedResult(null);
        setDetailsOpen(false);
        setRunId(run_id);
        syncQuery(t, d, p);
      } catch (e) {
        if (isUnauthorizedError(e)) return;
        const err = parseApiError(e);
        if (err.error === "run_active" && err.run_id) {
          const rows = await api.listStreamRuns().catch(() => [] as StreamRunRow[]);
          const active = rows.find((r) => r.run_id === err.run_id);
          if (active && active.mode.startsWith("bench-")) {
            attachRow(active);
            setNotice({ tone: "muted", title: "Attached to the bench run already active on this endpoint", body: err.message });
          } else {
            setNotice({ tone: "warn", title: "Another run is active on this endpoint", body: `${err.message} — stop it on /streams first.` });
          }
        } else {
          setNotice({ tone: "danger", title: "Run did not start", body: err.message });
        }
      } finally {
        setStarting(false);
      }
    },
    [running, starting, syncQuery],
  );
  const run = useCallback(() => void start(tab, decodeCfg, prefillCfg), [start, tab, decodeCfg, prefillCfg]);
  const stopRun = useCallback(() => {
    if (!runId || !running) return;
    api.stopStreamRun(runId).catch(() => {});
  }, [runId, running]);

  // When a live run ends, the engine has imported it: refresh the strand. (The
  // EventSource closed itself on `done`; nothing to stop.)
  const doneSeen = useRef<string | null>(null);
  useEffect(() => {
    if (!runId || !state.done || doneSeen.current === runId) return;
    doneSeen.current = runId;
    const t = setTimeout(() => void history.refresh(), 400);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, state.done]);

  // ── The result on show ────────────────────────────────────────────────────
  const liveResult = useMemo<BenchResult | null>(() => {
    if (!runId || !state.hello || !runMeta) return null;
    const summary = state.done?.summary ?? null;
    const savedRunId = state.done?.saved_run_id ?? null;
    if (runMeta.tab === "decode") {
      return decodeResultFromLive({
        runId,
        model: state.hello.model,
        pack: runMeta.pack,
        levels: state.hello.levels ?? runMeta.levels,
        maxTokens: state.hello.max_tokens,
        startedAt: state.hello.started_at,
        rows: state.levels,
        summary,
        savedRunId,
        fingerprint: state.hello.serve_fingerprint ?? null,
      });
    }
    return prefillResultFromLive({
      runId,
      model: state.hello.model,
      pack: runMeta.pack,
      sizes: state.hello.sizes ?? runMeta.sizes,
      startedAt: state.hello.started_at,
      rows: state.levels,
      summary,
      savedRunId,
      fingerprint: state.hello.serve_fingerprint ?? null,
    });
  }, [runId, runMeta, state.hello, state.levels, state.done]);

  const fromHistory = !!selectedResult && !runId;
  const result = runId ? liveResult : selectedResult;
  const shownTab: Tab = runId ? (runMeta?.tab ?? tab) : tab;

  const previous = usePreviousRun(result);
  const ghost = useMemo(
    () => (previous?.result?.kind === "decode" ? { arms: previous.result.arms, label: fmtDate(previous.createdAt) } : null),
    [previous],
  );

  // History selection: load the envelope, keep the row highlighted.
  const select = useCallback(
    (e: HistoryEntry) => {
      if (running) return;
      setRunId(null);
      setSelected(e);
      setDetailsOpen(false);
      if (e.result) setSelectedResult(e.result);
      else {
        setSelectedResult(null);
        loadEntry(e.id, e.source)
          .then((l) => setSelectedResult((cur) => (cur === null ? l.result : cur)))
          .catch(() => setNotice({ tone: "warn", title: "Envelope unavailable", body: `Could not load run ${e.id}.` }));
      }
    },
    [running],
  );
  useEffect(() => {
    // The strand's lazy load may finish after the row was picked.
    if (selected && !selectedResult) {
      const fresh = history.entries.find((x) => x.id === selected.id);
      if (fresh?.result) setSelectedResult(fresh.result);
    }
  }, [history.entries, selected, selectedResult]);
  const selectedId = runId ? (state.done?.saved_run_id ?? runId) : (selected?.id ?? null);
  function stepHistory(delta: number) {
    const list = history.entries;
    if (!list.length) return;
    const idx = list.findIndex((e) => e.id === selectedId);
    const next = idx < 0 ? (delta > 0 ? 0 : list.length - 1) : Math.min(list.length - 1, Math.max(0, idx + delta));
    select(list[next]);
  }

  const runLevels = useCallback(
    (levels: number[]) => {
      const merged = sortConcurrencies(new Set([...decodeCfg.levels, ...levels]));
      const next = { ...decodeCfg, levels: merged };
      setDecodeCfg(next);
      void start("decode", next, prefillCfg);
    },
    [decodeCfg, prefillCfg, start],
  );
  const runAgain = useCallback(() => {
    if (result?.kind === "decode") {
      const next: DecodeConfig = { ...decodeCfg, pack: result.pack || decodeCfg.pack, levels: result.levels.length ? result.levels : decodeCfg.levels, maxTokens: result.maxTokens ?? decodeCfg.maxTokens };
      setDecodeCfg(next);
      void start("decode", next, prefillCfg);
    } else if (result?.kind === "prefill") {
      const next: PrefillConfig = { sizes: result.sizes.length ? result.sizes : prefillCfg.sizes };
      setPrefillCfg(next);
      void start("prefill", decodeCfg, next);
    } else run();
  }, [result, decodeCfg, prefillCfg, start, run]);

  // ── Keys ──────────────────────────────────────────────────────────────────
  // One listener for the page's life; it reads the latest render through a ref.
  const keyState = { canRun, run, stopRun, result, tab, running, stepHistory };
  const keys = useRef(keyState);
  keys.current = keyState;
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const { canRun, run, stopRun, result, tab, running, stepHistory } = keys.current;
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key === "Enter") {
        e.preventDefault();
        if (canRun) run();
        return;
      }
      if (mod && e.key === ".") {
        e.preventDefault();
        stopRun();
        return;
      }
      if (mod && e.key === "/") {
        e.preventDefault();
        if (result) setDetailsOpen((v) => !v);
        return;
      }
      if (mod && e.shiftKey && (e.key === "C" || e.key === "c")) {
        e.preventDefault();
        if (result) setCopyRequest((n) => n + 1);
        return;
      }
      if (mod || e.altKey || isEditableTarget(e.target)) return;
      if (e.key === "r") {
        e.preventDefault();
        if (canRun) run();
      } else if (e.key === "[" || e.key === "]") {
        e.preventDefault();
        stepHistory(e.key === "]" ? 1 : -1);
      } else if (tab === "decode" && !running) {
        const preset = presetForKey(e.key);
        if (preset !== null) {
          e.preventDefault();
          setDecodeCfg((c) => ({ ...c, levels: sortConcurrencies(toggleLevel(new Set(c.levels), preset)) }));
        }
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const errorMessage = state.error ?? state.done?.summary.error ?? null;

  return (
    <div className="lab-fade-in space-y-4">
      <div className="page-header">
        <div className="min-w-0">
          <h1 className="page-title">Bench</h1>
          <p className="page-sub">Decode and prefill synchronization runs against the live endpoint, drawn as they happen.</p>
        </div>
        <SegmentedControl<Tab>
          value={tab}
          onChange={changeTab}
          ariaLabel="Bench mode"
          options={[
            { id: "decode", label: "Decode" },
            { id: "prefill", label: "Prefill" },
          ]}
        />
      </div>

      {notice && (
        <Callout tone={notice.tone} title={notice.title} onDismiss={() => setNotice(null)}>
          {notice.body}
        </Callout>
      )}
      {running && shownTab !== tab && (
        <Callout tone="muted">
          A {shownTab} run is in progress — its instrument stays on the {shownTab} tab.
        </Callout>
      )}

      <div className="grid gap-4 lg:grid-cols-12">
        <div className="space-y-4 lg:col-span-4">
          <Panel title="Configuration" padded>
            {tab === "decode" ? (
              <DecodeConfigPanel
                packs={packs}
                cfg={decodeCfg}
                onChange={setDecodeCfg}
                locked={running}
                running={running && shownTab === "decode"}
                starting={starting}
                canRun={canRun}
                reason={reason}
                onRun={run}
                onStop={stopRun}
              />
            ) : (
              <PrefillConfigPanel
                cfg={prefillCfg}
                onChange={setPrefillCfg}
                locked={running}
                running={running && shownTab === "prefill"}
                starting={starting}
                canRun={canRun}
                reason={reason}
                onRun={run}
                onStop={stopRun}
              />
            )}
          </Panel>
          <Panel padded>
            <HistoryStrand kind={tab} entries={history.entries} loading={history.loading} selectedId={selectedId} onSelect={select} packs={packs} />
          </Panel>
        </div>

        <Panel padded className="lg:col-span-8">
          {shownTab === "decode" ? (
            <DecodeInstrument
              cfg={decodeCfg}
              status={status}
              live={runId ? state : null}
              result={result?.kind === "decode" ? result : null}
              ghost={ghost}
              fromHistory={fromHistory}
              hardware={hardware}
              errorMessage={errorMessage}
              onRunLevels={runLevels}
              onRunAgain={runAgain}
              detailsOpen={detailsOpen}
              onToggleDetails={() => setDetailsOpen((v) => !v)}
              copyRequest={copyRequest}
              busy={!canRun}
            />
          ) : (
            <PrefillInstrument
              cfg={prefillCfg}
              status={status}
              live={runId ? state : null}
              result={result?.kind === "prefill" ? result : null}
              fromHistory={fromHistory}
              hardware={hardware}
              errorMessage={errorMessage}
              onRunAgain={runAgain}
              detailsOpen={detailsOpen}
              onToggleDetails={() => setDetailsOpen((v) => !v)}
              copyRequest={copyRequest}
              busy={!canRun}
            />
          )}
        </Panel>
      </div>
    </div>
  );
}
