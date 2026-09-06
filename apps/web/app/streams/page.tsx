"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { api } from "@/lib/api";
import { isUnauthorizedError } from "@/lib/auth-token";
import { serveHealthy, useLabStatus } from "@/lib/lab-status-store";
import type { StreamPack, StreamRunRow } from "@/lib/stream-run-types";
import { copyText, downloadText } from "@/lib/streams/clipboard";
import { exportStem, jsonSnapshot, markdownSummary, type RunControls } from "@/lib/streams/export";
import { fmtInt } from "@/lib/streams/format";
import { strandDesync, type Desync } from "@/lib/streams/stalls";
import { AGG_WINDOW_MS, useStreamRun } from "@/lib/use-stream-run";
import { Btn, Callout, Corridor, Eyebrow, PageSkeleton, Panel, SegmentedControl, SyncRing, btnClass } from "@/components/ui";
import { ControlBar, type Endpoint, type StreamControls } from "@/components/streams/ControlBar";
import { HelixChart } from "@/components/streams/HelixChart";
import { HistoryStrip } from "@/components/streams/HistoryStrip";
import { InstrumentBar } from "@/components/streams/InstrumentBar";
import { SequenceView } from "@/components/streams/SequenceView";
import { StrandCard } from "@/components/streams/StrandCard";
import { TranscriptSheet } from "@/components/streams/TranscriptSheet";
import { cn } from "@/lib/utils";
import "./streams.css";

/*
  Streams — Strands, the live concurrent-streams view. One strand per stream,
  the Helix (stacked per-strand tok/s whose top edge is the ONE aggregate),
  a Sequence view of the prefill storm, and a card per strand with phase bar,
  transcript, ITL and finish evidence. Every number is the controller engine's;
  the page only splits and draws it.
*/

const DEFAULT_CONTROLS: StreamControls = {
  base_url: "",
  pack: "prose",
  n: 4,
  arrival: "burst",
  max_tokens: 256,
  fill_to_max: false,
  thinking: "off",
};

type Notice = { tone: "ok" | "warn" | "danger" | "muted" | "accent"; title: string; body?: string; action?: React.ReactNode };

function canonUrl(raw: string): string {
  try {
    const u = new URL(raw);
    return `${u.origin}${u.pathname.replace(/\/+$/, "").replace(/\/v1$/, "")}`;
  } catch {
    return raw;
  }
}

function parseApiError(e: unknown): { error?: string; message: string; run_id?: string } {
  const msg = e instanceof Error ? e.message : String(e);
  try {
    const j = JSON.parse(msg) as { error?: string; message?: string; run_id?: string };
    return { error: j.error, message: j.message || msg, run_id: j.run_id };
  } catch {
    return { message: msg };
  }
}

export default function StreamsPage() {
  return (
    <Suspense fallback={<PageSkeleton rows={4} />}>
      <StreamsRoom />
    </Suspense>
  );
}

function StreamsRoom() {
  const router = useRouter();
  const pathname = usePathname();
  const search = useSearchParams();
  const wall = search.get("wall") === "1";
  const urlRun = search.get("run");

  const { status, loading, needToken, unreachable, error: statusError } = useLabStatus();
  const [packs, setPacks] = useState<StreamPack[]>([]);
  const [endpoints, setEndpoints] = useState<Endpoint[]>([]);
  const [controls, setControls] = useState<StreamControls>(DEFAULT_CONTROLS);
  const [runControls, setRunControls] = useState<RunControls>({});
  const [runId, setRunId] = useState<string | null>(urlRun);
  const { state, stop } = useStreamRun(runId);
  const [runs, setRuns] = useState<StreamRunRow[]>([]);
  const [starting, setStarting] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const [focused, setFocused] = useState<number | null>(null);
  const [sheet, setSheet] = useState<number | null>(null);
  const [mutedThinking, setMutedThinking] = useState<ReadonlySet<number>>(() => new Set());
  const [view, setView] = useState<"helix" | "sequence">("helix");
  const [tickAt, setTickAt] = useState(0);
  const desyncLatch = useRef(new Map<number, Desync>());

  const attached = !!runId;
  const finished = !!state.done;
  const live = attached && !finished;
  const strands = state.strands;
  const nowAt = Math.max(tickAt, state.clock?.at ?? 0);

  const anyUp = serveHealthy(status) || Object.values(status?.backends ?? {}).some((b) => b.ok);
  const canRun = anyUp && !needToken && !unreachable && !!controls.base_url;
  const runDisabledReason = needToken
    ? "LAIL_TOKEN required — paste it in the banner"
    : unreachable
      ? "Controller unreachable"
      : !anyUp
        ? "No model served"
        : undefined;

  const packLabel = useCallback((id: string) => packs.find((p) => p.id === id)?.label ?? (id || "—"), [packs]);

  // ── Boot: packs, endpoints, recent runs; re-attach to a live run ──────────
  const refreshRuns = useCallback(() => api.listStreamRuns().then(setRuns).catch(() => {}), []);
  useEffect(() => {
    api.streamPacks().then(setPacks).catch(() => {});
    api.configure
      .get()
      .then((s) => {
        const eps = Object.entries(s.backends)
          .filter(([, b]) => b.enabled && b.url)
          .map(([k, b]) => ({ url: canonUrl(b.url), label: b.label || k }));
        setEndpoints(eps);
        const def = s.backends[s.defaultBackend]?.enabled ? canonUrl(s.backends[s.defaultBackend].url) : eps[0]?.url;
        if (def) setControls((c) => (c.base_url ? c : { ...c, base_url: def }));
      })
      .catch(() => {});
    void refreshRuns();
  }, [refreshRuns]);

  const attach = useCallback(
    (id: string) => {
      desyncLatch.current = new Map();
      setFocused(null);
      setSheet(null);
      setHovered(null);
      setRunId(id);
      const q = new URLSearchParams();
      q.set("run", id);
      if (wall) q.set("wall", "1");
      router.replace(`${pathname}?${q}`);
    },
    [pathname, router, wall],
  );

  // URL is the source of truth for which run we're on (reload, back/forward).
  useEffect(() => {
    if (urlRun && urlRun !== runId) setRunId(urlRun);
  }, [urlRun, runId]);

  // No run in the URL: re-attach to a live load run if there is one.
  const autoAttached = useRef(false);
  useEffect(() => {
    if (autoAttached.current || urlRun || !runs.length) return;
    autoAttached.current = true;
    const liveRun = runs.find((r) => r.status === "running" && r.mode === "load");
    if (liveRun) {
      setRunControls({ pack: liveRun.pack });
      attach(liveRun.run_id);
      setNotice({ tone: "muted", title: "Re-attached", body: `Run ${liveRun.run_id} was still live on ${liveRun.base_url}.` });
    }
  }, [runs, urlRun, attach]);

  // ── Wall mode: attribute on <html> so streams.css can hide the shell ─────
  useEffect(() => {
    if (!wall) return;
    document.documentElement.setAttribute("data-wall", "1");
    return () => document.documentElement.removeAttribute("data-wall");
  }, [wall]);

  // 1 Hz clock while live so stalls/desync surface even if the stream goes quiet.
  useEffect(() => {
    if (!live) return;
    setTickAt(performance.now());
    const id = window.setInterval(() => setTickAt(performance.now()), 1000);
    return () => window.clearInterval(id);
  }, [live]);

  // Run settled: refresh history once.
  useEffect(() => {
    if (finished) void refreshRuns();
  }, [finished, refreshRuns]);

  // ── Desync (latched per strand for the run — evidence stays) ─────────────
  const desyncs = useMemo(() => {
    const latch = desyncLatch.current;
    return strands.map((s) => {
      const held = latch.get(s.i);
      if (held) return held;
      const d = strandDesync(s, nowAt);
      if (d.desync) latch.set(s.i, d);
      return d;
    });
  }, [strands, nowAt]);

  // ── Actions ──────────────────────────────────────────────────────────────
  const run = useCallback(async () => {
    if (!canRun || live || starting) return;
    setStarting(true);
    setNotice(null);
    try {
      const { run_id } = await api.startStreamRun({
        mode: "load",
        base_url: controls.base_url,
        pack: controls.pack,
        n: controls.n,
        max_tokens: controls.max_tokens,
        fill_to_max: controls.fill_to_max,
        thinking: controls.thinking,
        arrival: controls.arrival,
      });
      setRunControls({ pack: controls.pack, arrival: controls.arrival, fill_to_max: controls.fill_to_max, thinking: controls.thinking });
      attach(run_id);
    } catch (e) {
      if (isUnauthorizedError(e)) return; // the shell banner owns 401
      const err = parseApiError(e);
      if (err.error === "run_active" && err.run_id) {
        const rows = await api.listStreamRuns().catch(() => [] as StreamRunRow[]);
        setRuns(rows);
        const active = rows.find((r) => r.run_id === err.run_id);
        if (active && active.mode === "load") {
          setRunControls({ pack: active.pack });
          attach(active.run_id);
          setNotice({ tone: "muted", title: "Attached to the run already active on this endpoint", body: err.message });
        } else {
          setNotice({
            tone: "warn",
            title: "A bench run holds the endpoint",
            body: "Wait for it to finish, or stop it on /bench.",
            action: (
              <Link href="/bench" className={btnClass("secondary", "sm")}>
                Open Bench
              </Link>
            ),
          });
        }
      } else if (/fetch|network|Failed/i.test(err.message) && !err.error) {
        setNotice({ tone: "danger", title: "Controller unreachable", body: err.message });
      } else {
        setNotice({ tone: "danger", title: "Run rejected", body: err.message });
      }
    } finally {
      setStarting(false);
    }
  }, [attach, canRun, controls, live, starting]);

  const doStop = useCallback(() => {
    if (!live) return;
    void stop().catch(() => {});
  }, [live, stop]);

  const exportOpts = useMemo(() => ({ ...runControls, packLabel }), [runControls, packLabel]);

  const copyAll = useCallback(async () => {
    if (!state.hello) return;
    const ok = await copyText(markdownSummary(state, { ...exportOpts, transcripts: true }));
    setNotice(ok ? { tone: "ok", title: `Copied ${strands.length} strands as Markdown` } : { tone: "warn", title: "Clipboard unavailable" });
  }, [exportOpts, state, strands.length]);

  const copyStrand = useCallback(
    async (i: number) => {
      const s = strands[i];
      if (!s) return;
      const ok = await copyText(s.reasoning ? `<thinking>\n${s.reasoning}\n</thinking>\n\n${s.text}` : s.text);
      setNotice(ok ? { tone: "ok", title: `Copied strand ${i + 1}` } : { tone: "warn", title: "Clipboard unavailable" });
    },
    [strands],
  );

  const exportJson = useCallback(() => {
    if (!state.hello) return;
    downloadText(`${exportStem(state)}.json`, JSON.stringify(jsonSnapshot(state, runControls), null, 2), "application/json");
  }, [runControls, state]);
  const exportMarkdown = useCallback(() => {
    if (!state.hello) return;
    downloadText(`${exportStem(state)}.md`, markdownSummary(state, exportOpts), "text/markdown");
  }, [exportOpts, state]);

  const setWall = useCallback(
    (on: boolean) => {
      const q = new URLSearchParams(search.toString());
      if (on) q.set("wall", "1");
      else q.delete("wall");
      const qs = q.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname);
    },
    [pathname, router, search],
  );

  const toggleThinking = useCallback((i: number | null) => {
    setMutedThinking((prev) => {
      const next = new Set(prev);
      if (i === null) {
        // no strand in hand: flip all
        if (next.size) next.clear();
        else for (const s of strands) next.add(s.i);
      } else if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  }, [strands]);

  const reuse = useCallback(
    (row: StreamRunRow) => {
      // The row carries endpoint, pack and n; tokens/arrival/thinking are not in the list row.
      setControls((c) => ({ ...c, base_url: row.base_url, pack: row.pack, n: Math.min(32, Math.max(1, row.n)) }));
      setNotice({ tone: "muted", title: "Controls pre-filled", body: `${packLabel(row.pack)} · ×${row.n} on ${row.base_url}` });
    },
    [packLabel],
  );

  // ── Keyboard ─────────────────────────────────────────────────────────────
  const keys = useRef({ run, doStop, copyAll, copyStrand, toggleThinking, focused, hovered, sheet, wall, setWall, live, strands });
  keys.current = { run, doStop, copyAll, copyStrand, toggleThinking, focused, hovered, sheet, wall, setWall, live, strands };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const k = keys.current;
      const t = e.target as HTMLElement | null;
      const typing = !!t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA" || t.isContentEditable);
      if (e.key === "Escape") {
        if (k.sheet !== null) return; // the dialog closes itself
        if (k.focused !== null) setFocused(null);
        else if (k.wall) k.setWall(false);
        return;
      }
      if (k.sheet !== null) return;
      const meta = e.metaKey || e.ctrlKey;
      if (meta && e.key === ".") {
        e.preventDefault();
        k.doStop();
        return;
      }
      if (meta && e.shiftKey && (e.key === "C" || e.key === "c")) {
        e.preventDefault();
        void k.copyAll();
        return;
      }
      if (typing || meta || e.altKey) return;
      if (e.shiftKey && e.key === "S") {
        e.preventDefault();
        setView((v) => (v === "helix" ? "sequence" : "helix"));
        return;
      }
      const target = k.focused ?? k.hovered;
      switch (e.key) {
        case "r":
          void k.run();
          break;
        case "f":
          if (k.focused !== null) setFocused(null);
          else if (k.hovered !== null) setFocused(k.hovered);
          break;
        case "t":
          k.toggleThinking(target);
          break;
        case "c":
          if (target !== null) void k.copyStrand(target);
          break;
        case "Enter":
          if (target !== null) setSheet(target);
          break;
        default:
          if (/^[1-9]$/.test(e.key)) {
            const i = Number(e.key) - 1;
            if (i < k.strands.length) setFocused((f) => (f === i ? null : i));
          }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ── Derived view bits ────────────────────────────────────────────────────
  const maxTokens = state.hello?.max_tokens ?? controls.max_tokens;
  const fillToMax = runControls.fill_to_max ?? false;
  const tpc = state.latest?.tokens_per_chunk ?? 1;
  const liveRates = state.latest?.strand_tok_s;
  const ringState = needToken ? "token" : unreachable ? "offline" : loading ? null : live ? "loading" : anyUp ? "serving" : "idle";
  const ringWord = needToken
    ? "Token required"
    : unreachable
      ? "Controller unreachable"
      : loading
        ? "Checking…"
        : live
          ? "Synchronizing"
          : finished
            ? state.done?.summary.status === "cancelled"
              ? "Sequence cancelled"
              : state.done?.summary.status === "error"
                ? "Sequence failed"
                : "Sequence synchronized"
            : anyUp
              ? "Endpoint live"
              : "No model serving";
  const sheetStrand = sheet !== null ? (strands[sheet] ?? null) : null;

  return (
    <div className={cn("streams-page lab-fade-in space-y-4", wall && "streams-wall")}>
      <div className="page-header streams-page-header">
        <div className="min-w-0">
          <div className="mb-2.5 flex flex-wrap items-center gap-2.5" aria-live="polite">
            <SyncRing state={ringState} label={ringWord} />
            <Eyebrow className={cn("tracking-[0.18em]", live ? "text-lab-line" : finished ? "text-lab-ok" : unreachable ? "text-lab-danger" : "text-lab-muted")}>
              {ringWord}
            </Eyebrow>
            {state.hello && (
              <>
                <span aria-hidden className="h-3 w-px bg-[color:var(--animus-hairline)]" />
                <Eyebrow className="lab-num" title={state.hello.model}>
                  {state.hello.model.split("/").pop()} · {state.hello.base_url.replace(/^https?:\/\//, "")} · run {state.hello.run_id}
                </Eyebrow>
              </>
            )}
          </div>
          <h1 className="page-title">Streams</h1>
          <p className="page-sub">
            Strands on the endpoint, live: text, TTFT, tok/s and the one aggregate they add up to. The Helix stacks them; the Sequence
            view shows the prefill storm.
          </p>
        </div>
      </div>

      {unreachable && (
        <Callout tone="danger" title="Controller unreachable on :8787">
          {statusError || "No answer from the controller."} Check that <code className="text-lab-text">bun run dev</code> is up.
        </Callout>
      )}

      {notice && (
        <Callout tone={notice.tone} title={notice.title} action={notice.action} onDismiss={() => setNotice(null)}>
          {notice.body}
        </Callout>
      )}

      {state.error && (
        <Callout tone="danger" title="Run failed">
          {state.error}
        </Callout>
      )}

      {!loading && !anyUp && !needToken && !unreachable && !attached ? (
        <Panel>
          <Corridor
            action={
              <Link href="/server" className={btnClass("primary", "sm")}>
                Serve a model
              </Link>
            }
          >
            No memory loaded. Serve a model to begin synchronization — strands need a live endpoint.
          </Corridor>
        </Panel>
      ) : (
        <>
          <Panel padded className="streams-controls-panel">
            <ControlBar
              controls={controls}
              onChange={(patch) => setControls((c) => ({ ...c, ...patch }))}
              endpoints={endpoints.length ? endpoints : controls.base_url ? [{ url: controls.base_url, label: "backend" }] : []}
              packs={packs}
              running={live}
              canRun={canRun}
              runDisabledReason={runDisabledReason}
              starting={starting}
              hasRun={!!state.hello}
              onRun={() => void run()}
              onStop={doStop}
              onCopyAll={() => void copyAll()}
              onExportJson={exportJson}
              onExportMarkdown={exportMarkdown}
              onWall={() => setWall(true)}
            />
          </Panel>

          {attached && <InstrumentBar state={state} live={live} />}

          {attached && (
            <Panel
              title={view === "helix" ? "Helix · last 60 s" : "Sequence · prefill storm"}
              action={
                <div className="flex items-center gap-3">
                  <Eyebrow className="hidden sm:inline">{view === "helix" ? "hover a band → its card" : "waiting · ttft · decode"}</Eyebrow>
                  <SegmentedControl
                    size="sm"
                    ariaLabel="Aggregate view"
                    value={view}
                    onChange={setView}
                    options={[
                      { id: "helix", label: "Helix" },
                      { id: "sequence", label: "Sequence ⇧S" },
                    ]}
                  />
                </div>
              }
            >
              <div className="px-3 pt-2 pb-1">
                {view === "helix" ? (
                  <HelixChart agg={state.agg} n={strands.length} windowMs={AGG_WINDOW_MS} hovered={hovered} focused={focused} onHover={setHovered} live={live} />
                ) : (
                  <SequenceView strands={strands} clock={state.clock} helloAt={state.hello_at} nowAt={nowAt} hovered={hovered} focused={focused} onHover={setHovered} />
                )}
              </div>
            </Panel>
          )}

          {attached ? (
            strands.length ? (
              <section className="strands-grid" aria-label="Strands">
                {strands.map((s, k) => (
                  <div key={s.i} style={{ "--k": k } as CSSProperties} className={cn("contents")}>
                    <StrandCard
                      strand={s}
                      packLabel={packLabel(s.pack)}
                      maxTokens={maxTokens}
                      fillToMax={fillToMax}
                      tokensPerChunk={tpc}
                      liveRate={liveRates?.[s.i] ?? 0}
                      desync={desyncs[k]}
                      focused={focused === s.i}
                      hovered={hovered === s.i}
                      thinkingMuted={mutedThinking.has(s.i)}
                      wall={wall}
                      onHover={setHovered}
                      onFocus={(i) => setFocused((f) => (f === i ? null : i))}
                      onExpand={setSheet}
                      onCopy={(i) => void copyStrand(i)}
                      onToggleThinking={toggleThinking}
                    />
                  </div>
                ))}
              </section>
            ) : (
              <Panel padded>
                <div className="flex items-center gap-3">
                  <SyncRing state="loading" label="Connecting" />
                  <Eyebrow>Connecting to run {runId}…</Eyebrow>
                </div>
              </Panel>
            )
          ) : (
            <section aria-label="Preview" className="space-y-2">
              <div className="flex items-center gap-2.5">
                <Eyebrow className="text-lab-text-dim">Run to synchronize {controls.n} strand{controls.n === 1 ? "" : "s"}</Eyebrow>
                <div aria-hidden className="animus-rule min-w-6 flex-1" />
                <Eyebrow className="lab-num">
                  {packLabel(controls.pack)} · {fmtInt(controls.max_tokens)} tok · {controls.arrival}
                </Eyebrow>
              </div>
              <div className="strands-grid">
                {Array.from({ length: controls.n }).map((_, k) => (
                  <div key={k} className="strand-ghost" style={{ "--k": k } as CSSProperties} aria-hidden>
                    <Eyebrow className="lab-num">strand {String(k + 1).padStart(2, "0")}</Eyebrow>
                  </div>
                ))}
              </div>
            </section>
          )}

          <HistoryStrip
            runs={runs}
            currentRunId={runId}
            packLabel={packLabel}
            onReuse={reuse}
            onOpen={(row) => {
              setRunControls({ pack: row.pack });
              attach(row.run_id);
            }}
          />
        </>
      )}

      {wall && (
        <div className="fixed right-3 bottom-3 z-30">
          <Btn variant="ghost" size="sm" onClick={() => setWall(false)} title="Exit wall mode (esc)">
            exit wall <kbd className="strand-kbd">esc</kbd>
          </Btn>
        </div>
      )}

      <TranscriptSheet
        strand={sheetStrand}
        packLabel={sheetStrand ? packLabel(sheetStrand.pack) : ""}
        tokensPerChunk={tpc}
        open={sheet !== null}
        onOpenChange={(open) => {
          if (!open) setSheet(null);
        }}
        onCopy={(i) => void copyStrand(i)}
      />
    </div>
  );
}
