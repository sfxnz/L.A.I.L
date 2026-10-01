"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { api, type RunRow } from "@/lib/api";
import { runHref } from "@/lib/run-href";
import { useShallow } from "zustand/react/shallow";
import { serveHealthy, useLabStatusStore } from "@/lib/lab-status-store";
import {
  Badge,
  Btn,
  Callout,
  EmptyState,
  Eyebrow,
  Panel,
  Skeleton,
  SyncRing,
  btnClass,
  eyebrowClass,
} from "@/components/ui";
import { Absent, Cell, CornerTicks, ScoreGauge, Section, scoreTone } from "@/components/evals/parts";

export default function EvalsPage() {
  const { healthy, hasStatus, statusLoading } = useLabStatusStore(
    useShallow((s) => ({ healthy: serveHealthy(s.status), hasStatus: !!s.status, statusLoading: s.loading })),
  );
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [latestTool, setLatestTool] = useState<RunRow | null>(null);
  const [runsTotal, setRunsTotal] = useState<number | null>(null);
  const [runsLoaded, setRunsLoaded] = useState(false);
  const [smokeOut, setSmokeOut] = useState<string | null>(null);
  const [smokeOk, setSmokeOk] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  // Lab status comes from the shell's shared poll; runs load on mount and when the
  // operator asks. The verdict and the total are their own queries (the kind filter
  // and the count run in SQL), never derived from the newest page of runs.
  const loadRuns = useCallback(
    () =>
      Promise.all([
        api.runs({ limit: 20 }).then(setRuns),
        api.runs({ kind: "agentic_tool_eval", limit: 1 }).then((r) => setLatestTool(r[0] ?? null)),
        api.runsCount().then((c) => setRunsTotal(c.count)),
      ])
        .catch(() => {})
        .finally(() => setRunsLoaded(true)),
    [],
  );
  useEffect(() => {
    void loadRuns();
  }, [loadRuns]);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await loadRuns();
    } finally {
      setRefreshing(false);
    }
  }, [loadRuns]);

  const loading = statusLoading || !runsLoaded;

  const latestScore =
    typeof latestTool?.summary?.final_score === "number"
      ? (latestTool.summary.final_score as number)
      : null;
  const latestRating =
    typeof latestTool?.summary?.rating === "string" ? (latestTool.summary.rating as string) : null;

  async function runSmoke() {
    setErr(null);
    setSmokeOut(null);
    setSmokeOk(null);
    setBusy(true);
    try {
      const r = await api.smoke();
      setSmokeOk(!!r.ok);
      setSmokeOut(r.content || JSON.stringify(r));
    } catch (e) {
      setErr(String((e as Error).message || e));
      setSmokeOk(false);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="lab-fade-in space-y-6">
      <div className="page-header">
        <div className="min-w-0">
          <div className="animus-eyebrow mb-1.5 flex items-center gap-2">
            <span aria-hidden className="h-3 w-px bg-lab-accent" />
            Bench control
          </div>
          <h1 className="page-title">Evals</h1>
          <p className="page-sub">Smoke and tool-eval quality vs the live serve; throughput and latency live on Bench</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Btn variant="secondary" size="sm" onClick={() => void refresh()} loading={refreshing}>
            Refresh
          </Btn>
          <Link href="/evals/tool" className={btnClass("primary", "sm")}>
            Tool Eval board
          </Link>
        </div>
      </div>

      {!loading && !healthy && hasStatus && (
        <Callout
          tone="warn"
          title="vLLM isn’t healthy"
          action={
            <Link href="/server" className={btnClass("secondary", "sm")}>
              Open Serve
            </Link>
          }
        >
          Start a model before running smoke, a bench, or tool-eval. Cold loads can take several
          minutes on large NVFP4 weights.
        </Callout>
      )}

      {err && (
        <Callout tone="danger" title="Eval failed" onDismiss={() => setErr(null)}>
          {err}
        </Callout>
      )}

      {/* ── Headline readout ─────────────────────────────────────────────── */}
      <Section className="lab-rise lab-rise-1 space-y-3"
          label="Last verdict"
          meta={
            <span className="flex items-center gap-2">
              <SyncRing
                state={healthy ? "serving" : "idle"}
                label={healthy ? "Endpoint healthy" : "Endpoint idle"}
              />
              <Eyebrow className="tracking-[0.18em]">
                {healthy ? "endpoint live" : "endpoint idle"}
              </Eyebrow>
            </span>
          }>

        <Panel>
          <div className="grid gap-px bg-lab-border-subtle md:grid-cols-[minmax(0,17rem)_minmax(0,1fr)]">
            {/* Score */}
            <div className="relative bg-[color:var(--animus-accent-wash)] px-5 py-4">
              <span aria-hidden className="absolute inset-y-0 left-0 w-[2px] bg-lab-accent" />
              <CornerTicks />
              <div className="animus-eyebrow">Tool-eval score</div>

              {loading ? (
                <div className="mt-3 space-y-2.5" aria-busy="true" aria-label="Loading score">
                  <Skeleton className="h-12 w-28" />
                  <Skeleton className="h-[6px] w-full" />
                </div>
              ) : latestScore != null ? (
                <>
                  <div className="mt-1.5 flex items-end gap-1.5">
                    <span className="font-[family-name:var(--font-display)] text-[62px] font-semibold leading-[0.78] tracking-[0.01em] tabular-nums text-lab-text">
                      {latestScore}
                    </span>
                    <span className="pb-1.5 font-[family-name:var(--font-display)] text-[14px] font-semibold uppercase leading-none tracking-[0.16em] text-lab-muted">
                      /100
                    </span>
                  </div>
                  <div className="mt-3">
                    <ScoreGauge pct={latestScore} label="Tool-eval score" />
                  </div>
                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    <Badge tone={scoreTone(latestScore)}>{latestRating || "scored"}</Badge>
                    <Link
                      href={`/evals/tool/${latestTool?.run_id}`}
                      className={eyebrowClass("text-lab-accent-bright underline-offset-4 hover:underline")}
                    >
                      Open run →
                    </Link>
                  </div>
                </>
              ) : (
                <>
                  <div className="mt-3 font-[family-name:var(--font-display)] text-[28px] font-semibold uppercase leading-[0.9] tracking-[0.14em] text-lab-muted">
                    No runs
                    <br />
                    yet
                  </div>
                  <div className="mt-3">
                    <ScoreGauge pct={null} label="Tool-eval score" />
                  </div>
                  <p className="mt-3 text-[12px] leading-snug text-lab-muted">
                    Run tool-eval from Serve — the verdict lands here score-first.
                  </p>
                </>
              )}
            </div>

            {/* Meta readout — hairline cells, not stat cards */}
            <div className="grid grid-cols-2 gap-px bg-lab-border-subtle sm:grid-cols-2">
              <Cell label="Model" className="bg-lab-panel">
                {loading ? (
                  <Skeleton className="h-3.5 w-32" />
                ) : latestTool?.model_id ? (
                  <span className="font-mono text-[12px]" title={latestTool.model_id}>
                    {latestTool.model_id.split("/").pop()}
                  </span>
                ) : (
                  <Absent>awaiting</Absent>
                )}
              </Cell>
              <Cell label="Recorded" className="bg-lab-panel">
                {loading ? (
                  <Skeleton className="h-3.5 w-24" />
                ) : latestTool?.created_at ? (
                  <span className="font-mono text-[12px] tabular-nums">
                    {latestTool.created_at.slice(0, 19).replace("T", " ")}
                  </span>
                ) : (
                  <Absent>awaiting</Absent>
                )}
              </Cell>
              <Cell label="Runs on record" className="bg-lab-panel">
                {loading ? (
                  <Skeleton className="h-3.5 w-10" />
                ) : (
                  <span className="font-[family-name:var(--font-display)] text-[18px] font-semibold leading-none tabular-nums">
                    {runsTotal ?? <Absent>unknown</Absent>}
                  </span>
                )}
              </Cell>
              <Cell label="Endpoint" className="bg-lab-panel">
                {loading ? (
                  <Skeleton className="h-3.5 w-20" />
                ) : (
                  <span className="inline-flex items-center gap-2">
                    <SyncRing state={healthy ? "serving" : "idle"} />
                    <span className="font-[family-name:var(--font-display)] text-[12px] font-semibold uppercase tracking-[0.14em]">
                      {healthy ? "serving" : "no serve"}
                    </span>
                  </span>
                )}
              </Cell>
            </div>
          </div>
        </Panel>
      </Section>

      {/* ── Instruments ──────────────────────────────────────────────────── */}
      <Section className="lab-rise lab-rise-2 space-y-3"
          label="Instruments"
          meta={
            <Eyebrow className="tracking-[0.18em]">
              {healthy ? "armed" : "locked · start a model"}
            </Eyebrow>
          }>

        <div className="grid gap-4 lg:grid-cols-2">
          <Panel title="Smoke" className="flex h-full flex-col">
            <div className="flex flex-1 flex-col gap-3.5 p-4">
              <p className="text-[13px] leading-relaxed text-lab-muted">
                Quick completion check (12×17 → 204). Catches empty / garbage output before you
                bench.
              </p>
              <Btn
                onClick={() => void runSmoke()}
                disabled={!healthy}
                loading={busy && smokeOk == null}
                title={!healthy ? "Start a model on Serve first" : undefined}
              >
                Run smoke
              </Btn>
              {smokeOk != null && (
                <div
                  className="animus-chamfer-sm flex flex-wrap items-center gap-2.5 border border-lab-border-subtle bg-lab-editor px-3 py-2.5"
                  role="status"
                >
                  <SyncRing
                    state={smokeOk ? "serving" : "offline"}
                    label={smokeOk ? "Smoke passed" : "Smoke failed"}
                  />
                  <Badge tone={smokeOk ? "ok" : "danger"}>{smokeOk ? "PASS" : "FAIL"}</Badge>
                  <span className="min-w-0 break-all font-mono text-[12px] text-lab-text-dim">
                    {smokeOut || <Absent>no output</Absent>}
                  </span>
                </div>
              )}
            </div>
          </Panel>

          <Panel title="Throughput & latency" className="flex h-full flex-col">
            <div className="flex flex-1 flex-col gap-3.5 p-4">
              <p className="text-[13px] leading-relaxed text-lab-muted">
                One bench: decode tok/s and TTFT across concurrency levels, and prefill throughput on long
                prompts — warmed up, repeated, and checked for foreign load on the server.
              </p>
              <div className="flex flex-wrap items-center gap-2">
                <Link href="/bench?tab=decode" className={btnClass("primary", "md")}>
                  Decode bench
                </Link>
                <Link href="/bench?tab=prefill" className={btnClass("secondary", "md")}>
                  Prefill bench
                </Link>
              </div>
            </div>
          </Panel>
        </div>
      </Section>

      {/* ── Run log ──────────────────────────────────────────────────────── */}
      <Section className="lab-rise lab-rise-3 space-y-3"
          label="Run log"
          meta={
            <Eyebrow className="lab-num tracking-[0.18em]">
              {loading ? "loading" : runsTotal !== null ? `${runs.length} newest of ${runsTotal}` : `${runs.length} newest`}
            </Eyebrow>
          }>

        <Panel>
          <div className="overflow-x-auto">
            <table className="lab-table">
              <thead>
                <tr>
                  <th scope="col">Run</th>
                  <th scope="col">Kind</th>
                  <th scope="col">Intent</th>
                  <th scope="col">Model</th>
                  <th scope="col">When</th>
                </tr>
              </thead>
              <tbody>
                {loading && (
                  <tr>
                    <td colSpan={5} className="!p-3">
                      <div className="space-y-2.5" aria-busy="true" aria-label="Loading runs">
                        {[0, 1, 2, 3].map((i) => (
                          <div key={i} className="grid grid-cols-5 gap-3">
                            <Skeleton className="h-3 w-full" />
                            <Skeleton className="h-3 w-[70%]" />
                            <Skeleton className="h-3 w-[50%]" />
                            <Skeleton className="h-3 w-[80%]" />
                            <Skeleton className="h-3 w-[60%]" />
                          </div>
                        ))}
                      </div>
                    </td>
                  </tr>
                )}
                {!loading &&
                  runs.map((r) => {
                    const isTool = r.kind === "agentic_tool_eval";
                    const href = runHref(r);
                    const s =
                      typeof r.summary?.final_score === "number"
                        ? (r.summary.final_score as number)
                        : null;
                    return (
                      <tr key={r.run_id}>
                        <td className="font-mono text-[12px]">
                          {href ? (
                            <Link
                              href={href}
                              className="text-lab-accent-bright underline-offset-4 hover:underline"
                            >
                              {r.run_id}
                            </Link>
                          ) : (
                            <span className="text-lab-text-dim">{r.run_id}</span>
                          )}
                        </td>
                        <td>
                          {isTool ? (
                            <span className="inline-flex items-center gap-2">
                              <span className="font-[family-name:var(--font-display)] text-[11px] font-semibold uppercase tracking-[0.14em] text-lab-text-dim">
                                tool-eval
                              </span>
                              {s != null ? (
                                <span className="font-[family-name:var(--font-display)] text-[15px] font-semibold leading-none tabular-nums text-lab-text">
                                  {s}
                                </span>
                              ) : (
                                <Absent>unscored</Absent>
                              )}
                            </span>
                          ) : (
                            <span className="font-mono text-[11px] text-lab-muted">{r.kind}</span>
                          )}
                        </td>
                        <td>
                          {r.intent ? (
                            <span className="font-mono text-[12px]">{r.intent}</span>
                          ) : (
                            <Absent>unset</Absent>
                          )}
                        </td>
                        <td className="max-w-[220px] truncate">
                          {r.model_id ? (
                            <span title={r.model_id}>{r.model_id.split("/").pop()}</span>
                          ) : (
                            <Absent>unknown</Absent>
                          )}
                        </td>
                        <td className="whitespace-nowrap font-mono text-[11px] tabular-nums text-lab-muted">
                          {r.created_at ? (
                            r.created_at.slice(0, 19).replace("T", " ")
                          ) : (
                            <Absent>no stamp</Absent>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                {!loading && !runs.length && (
                  <tr>
                    <td colSpan={5}>
                      <EmptyState
                        title="No runs yet"
                        action={
                          <Btn size="sm" disabled={!healthy || busy} onClick={() => void runSmoke()}>
                            Run smoke first
                          </Btn>
                        }
                      >
                        Smoke, bench or tool-eval when the endpoint is healthy. Bench runs open on
                        Bench, tool-eval results on the board.
                      </EmptyState>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Panel>
      </Section>
    </div>
  );
}
