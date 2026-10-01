"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { useShallow } from "zustand/react/shallow";
import { api, type RunRow } from "@/lib/api";
import { serveHealthy, serverNow, useLabStatusStore, useStale } from "@/lib/lab-status-store";
import { ClusterPanel } from "@/components/ClusterPanel";
import { LastSyncCard } from "@/components/bench/LastSyncCard";
import { EndpointHero } from "@/components/status/EndpointHero";
import { LiveAge } from "@/components/status/LiveAge";
import { Btn, Callout, Corridor, EmptyState, Eyebrow, Panel, btnClass } from "@/components/ui";

/*
  Status — the console's front page, one live instrument:
    Endpoint · the served model: live decode / throughput / TTFT / requests /
               spec acceptance / KV, and how to wire Hermes to it
    Sparks   · per-node hardware and the fabric between them
    Bench    · the last decode bench of the served model
  Every number appears once. Everything updates from the store's live stream
  (one sample per second); when it stops, the page says so and dims instead of
  showing frozen numbers as live.
*/

function Band({ index, label, meta }: { index: string; label: string; meta?: ReactNode }) {
  return (
    <div className="flex items-center gap-2.5">
      <span className="animus-eyebrow shrink-0 tabular-nums text-lab-line!">{index}</span>
      <span className="animus-eyebrow shrink-0 text-lab-text-dim!">{label}</span>
      <div aria-hidden className="animus-rule min-w-6 flex-1" />
      {meta ? <span className="shrink-0">{meta}</span> : null}
    </div>
  );
}

export default function StatusPage() {
  const { status, loading, needToken, unreachable, error, receivedAt, runLive, refresh } = useLabStatusStore(
    useShallow((s) => ({
      status: s.status,
      loading: s.loading,
      needToken: s.needToken,
      unreachable: s.unreachable,
      error: s.error,
      receivedAt: s.receivedAt,
      runLive: !!s.liveRun,
      refresh: s.refresh,
    })),
  );
  const samples = useLabStatusStore((s) => s.samples);
  const stale = useStale();
  const [decodeRuns, setDecodeRuns] = useState<RunRow[]>([]);
  const [runsLoading, setRunsLoading] = useState(true);
  const [retrying, setRetrying] = useState(false);

  // The served model's latest decode bench: on mount, when a run in this tab settles,
  // and every minute while visible (a bench from another tab or Hermes lands here too).
  const loadRuns = useCallback(
    () =>
      api
        .runs({ kind: "decode", limit: 100 })
        .then(setDecodeRuns)
        .catch(() => {})
        .finally(() => setRunsLoading(false)),
    [],
  );
  useEffect(() => {
    void loadRuns();
    const t = setInterval(() => {
      if (!document.hidden) void loadRuns();
    }, 60_000);
    return () => clearInterval(t);
  }, [loadRuns]);
  const wasLive = useRef(runLive);
  useEffect(() => {
    // The engine imports a bench just before its `done`; give the index a moment.
    if (wasLive.current && !runLive) {
      const t = setTimeout(() => void loadRuns(), 1500);
      wasLive.current = runLive;
      return () => clearTimeout(t);
    }
    wasLive.current = runLive;
  }, [runLive, loadRuns]);

  const serve = status?.serve ?? null;
  const cluster = serve?.cluster ?? null;
  const healthy = serveHealthy(status);
  const now = serverNow({ status, receivedAt });
  const nodeCount = cluster?.nodes?.length ?? 0;
  const engineDown = !!serve?.unreachable;

  return (
    <div className="lab-fade-in space-y-4">
      <div className="page-header">
        <div className="min-w-0">
          <h1 className="page-title">Status</h1>
          <p className="page-sub">The served model and the Sparks, live — one sample a second.</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Link href="/server" className={btnClass(healthy ? "secondary" : "primary", "sm")}>
            Serve
          </Link>
          <Link href="/bench" className={btnClass("secondary", "sm")}>
            Bench
          </Link>
        </div>
      </div>

      {needToken && (
        <Callout tone="warn" title="LAIL_TOKEN required">
          The controller is up. Paste the token in the banner — it stays in sessionStorage and is sent as{" "}
          <code className="text-lab-text">X-Lail-Token</code>.
        </Callout>
      )}

      {unreachable && (
        <Callout
          tone="danger"
          title="Controller unreachable"
          action={
            <Btn
              variant="secondary"
              size="sm"
              loading={retrying}
              onClick={() => {
                setRetrying(true);
                void refresh().finally(() => setRetrying(false));
              }}
            >
              Retry
            </Btn>
          }
        >
          {error ? `${error}. ` : ""}Check that <code className="text-lab-text">bun run dev</code> is up on this host
          (ports 3000 / 8787 / 8765). The numbers below are the last ones received.
        </Callout>
      )}

      {!unreachable && engineDown && (
        <Callout tone="danger" title="Serve-engine unreachable">
          The controller answers but serve-engine does not{serve?.error ? ` (${serve.error})` : ""}. Live readings resume
          when it is back on :8765.
        </Callout>
      )}

      <section className="space-y-2.5">
        <Band index="01" label="Endpoint" meta={<LiveAge />} />
        <div className="lab-rise">
          {needToken ? null : healthy && serve ? (
            <EndpointHero
              serve={serve}
              defaultBackend={status?.defaultBackend}
              endpoint={samples.endpoint}
              serverNow={now}
              stale={stale}
            />
          ) : (
            <Panel>
              <Corridor
                title={loading ? "Connecting…" : "No model serving"}
                action={
                  loading ? undefined : (
                    <Link href="/server" className={btnClass("primary", "sm")}>
                      Serve a model
                    </Link>
                  )
                }
              >
                {loading
                  ? "Waiting for the first live sample."
                  : "Start one on Serve — the Sparks below stay live with nothing loaded."}
              </Corridor>
            </Panel>
          )}
        </div>
      </section>

      <section className="space-y-2.5">
        <Band
          index="02"
          label="Sparks"
          meta={<Eyebrow className="lab-num">{nodeCount >= 2 ? `${nodeCount} nodes` : nodeCount === 1 ? "this host" : ""}</Eyebrow>}
        />
        <div className="lab-rise lab-rise-1">
          {needToken ? (
            <Panel title="Sparks" padded>
              <EmptyState title="Token required">Paste LAIL_TOKEN in the banner to load the Sparks.</EmptyState>
            </Panel>
          ) : (
            <ClusterPanel cluster={cluster} loading={loading} samples={samples.nodes} serverNow={now} stale={stale} />
          )}
        </div>
      </section>

      <section className="space-y-2.5">
        <Band index="03" label="Bench" />
        <div className="lab-rise lab-rise-2">
          <LastSyncCard runs={decodeRuns} loading={runsLoading} servingModel={healthy ? (serve?.model_id ?? null) : null} />
        </div>
      </section>
    </div>
  );
}
