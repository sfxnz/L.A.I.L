"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, watchJob, type LabStatus, type RunRow } from "@/lib/api";
import { isUnauthorizedError } from "@/lib/auth-token";
import {
  Badge,
  Btn,
  Callout,
  EmptyState,
  Field,
  Input,
  LogView,
  Panel,
  ProgressBar,
  Skeleton,
  inputCls,
  btnClass,
} from "@/components/ui";
import { usePageTitle } from "@/lib/usePageTitle";
import { cn } from "@/lib/utils";

/* ═══════════════════════════════════════════════════════════════════════════
   Animus readout atoms — local to the Evals surfaces.
   Every colour is a lab-* token so both worlds (dark void / light plate)
   resolve. Crimson is the only chromatic accent; lab-line is structure.
   ═══════════════════════════════════════════════════════════════════════════ */

/** An absent reading is a deliberate HUD state — never a bare em-dash. */
function Absent({ children = "none" }: { children?: ReactNode }) {
  return (
    <span className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.18em] text-lab-muted">
      {children}
    </span>
  );
}

/**
 * Restrained corner ticks for the score frame.
 *
 * Deliberately NOT .animus-bracketed: that utility pins its brackets at -1px,
 * which the Panel's `overflow-hidden` clips, and its `border-top: 1px solid`
 * shorthand resets the colour to currentColor. These sit inside the box and
 * ride --animus-tick, so they read as hairline structure in both worlds.
 */
function CornerTicks() {
  const arm = "pointer-events-none absolute h-2.5 w-2.5 border-[color:var(--animus-tick)]";
  return (
    <span aria-hidden>
      <span className={cn(arm, "left-1.5 top-1.5 border-l border-t")} />
      <span className={cn(arm, "right-1.5 top-1.5 border-r border-t")} />
      <span className={cn(arm, "bottom-1.5 left-1.5 border-b border-l")} />
      <span className={cn(arm, "bottom-1.5 right-1.5 border-b border-r")} />
    </span>
  );
}

/**
 * A page section hung off the vertical spine.
 *
 * The spine is a hairline rail down the left edge; every section branches off
 * it with a crimson node + eyebrow + horizontal rule, so the eye is carried
 * score → categories → scenarios with no dead vertical gap.
 */
function Section({
  label,
  meta,
  children,
  className,
}: {
  label: string;
  meta?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("relative space-y-3 pl-4", className)}>
      {/* spine rail */}
      <span
        aria-hidden
        className="absolute bottom-1 left-0 top-2 w-px bg-[color:var(--animus-hairline)]"
      />
      {/* branch node */}
      <span aria-hidden className="absolute left-0 top-2 h-3 w-px bg-lab-accent" />
      <span aria-hidden className="absolute left-0 top-[13px] h-px w-2 bg-[color:var(--animus-hairline)]" />

      <div className="flex items-center gap-3">
        <h2 className="animus-eyebrow shrink-0 text-lab-text-dim">{label}</h2>
        <span aria-hidden className="animus-rule min-w-8 flex-1" />
        {meta ? <div className="shrink-0">{meta}</div> : null}
      </div>
      {children}
    </section>
  );
}

/** 0–100 gauge with division ticks. Crimson fill, hairline graticule. */
function Gauge({
  pct,
  label,
  divisions = 10,
  className,
}: {
  pct: number | null;
  label: string;
  divisions?: number;
  className?: string;
}) {
  const v = Math.max(0, Math.min(100, pct ?? 0));
  const step = 100 / divisions;
  return (
    <div
      className={cn("relative h-[6px] w-full overflow-hidden bg-lab-hover", className)}
      role="img"
      aria-label={pct == null ? `${label}: no reading` : `${label}: ${Math.round(v)} of 100`}
    >
      {pct != null && (
        <div
          className="h-full bg-lab-accent transition-[width] duration-700 ease-out"
          style={{ width: `${v}%` }}
        />
      )}
      <span
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage: `repeating-linear-gradient(90deg, transparent 0 calc(${step}% - 1px), var(--animus-hairline) calc(${step}% - 1px), var(--animus-hairline) ${step}%)`,
        }}
      />
    </div>
  );
}

/** Hairline readout cell — the HUD replacement for a stat card. */
function Cell({
  label,
  children,
  className,
}: {
  label: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("min-w-0 px-4 py-3", className)}>
      <div className="animus-eyebrow truncate text-[10px]">{label}</div>
      <div className="mt-1.5 truncate text-[13px] text-lab-text">{children}</div>
    </div>
  );
}

function scoreTone(score: number | null | undefined) {
  if (score == null) return "muted" as const;
  if (score >= 90) return "ok" as const;
  if (score >= 75) return "accent" as const;
  if (score >= 50) return "warn" as const;
  return "danger" as const;
}

export default function EvalsPage() {
  usePageTitle("Evals");
  const [status, setStatus] = useState<LabStatus | null>(null);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [logs, setLogs] = useState("");
  const [jobMsg, setJobMsg] = useState("");
  const [jobStatus, setJobStatus] = useState("");
  const [jobProgress, setJobProgress] = useState(0);
  const [smokeOut, setSmokeOut] = useState<string | null>(null);
  const [smokeOk, setSmokeOk] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [runner, setRunner] = useState<"workflow" | "prefill" | "concurrency">("workflow");
  const [conc, setConc] = useState("1,2,4");
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const refresh = useCallback(async (opts?: { soft?: boolean }) => {
    if (!opts?.soft) setRefreshing(true);
    try {
      const [s, r] = await Promise.all([
        api.labStatus(),
        api.runs().catch(() => [] as RunRow[]),
      ]);
      setStatus(s);
      setRuns(r);
      setErr(null);
    } catch (e) {
      if (!isUnauthorizedError(e)) setErr(String((e as Error).message || e));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh({ soft: true }), 8000);
    return () => clearInterval(t);
  }, [refresh]);

  const healthy = !!(status?.serve && !status.serve.unreachable && status.serve.healthy);
  const jobRunning = jobStatus === "running" || jobStatus === "queued";

  /* Purely derived from the runs already in state — no extra fetch. */
  const latestTool = useMemo(
    () =>
      runs.find(
        (r) =>
          (r.kind === "agentic_tool_eval" || String(r.kind || "").includes("tool")) &&
          typeof r.summary?.final_score === "number",
      ) || null,
    [runs],
  );
  const latestScore =
    typeof latestTool?.summary?.final_score === "number"
      ? (latestTool.summary.final_score as number)
      : null;
  const latestRating =
    typeof latestTool?.summary?.rating === "string" ? (latestTool.summary.rating as string) : null;

  function track(jobId: string) {
    setBusy(true);
    setLogs("");
    setJobMsg("starting…");
    setJobStatus("running");
    setJobProgress(0);
    watchJob(
      jobId,
      (chunk) => setLogs((l) => (l + chunk).slice(-80_000)),
      (s) => {
        setJobStatus(s.status);
        setJobMsg(s.message);
        setJobProgress(s.progress ?? 0);
        if (s.status === "done" || s.status === "error" || s.status === "failed") {
          setBusy(false);
          void refresh({ soft: true });
        }
      },
      () => {
        setBusy(false);
        void refresh({ soft: true });
      },
    );
  }

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

  async function runPerf() {
    setErr(null);
    setBusy(true);
    try {
      const concurrencies = conc
        .split(/[,\s]+/)
        .map((x) => parseInt(x, 10))
        .filter((n) => Number.isFinite(n) && n > 0);
      const { job_id } = await api.benchPerf({
        runner,
        concurrencies: concurrencies.length ? concurrencies : [1, 2, 4],
        concurrency: concurrencies[0] || 4,
        intent: "attach",
      });
      track(job_id);
    } catch (e) {
      setErr(String((e as Error).message || e));
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
          <p className="page-sub">Smoke, perf, and tool-eval quality vs the live serve</p>
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

      {!loading && !healthy && status && (
        <Callout
          tone="warn"
          title="vLLM isn’t healthy"
          action={
            <Link href="/server" className={btnClass("secondary", "sm")}>
              Open Serve
            </Link>
          }
        >
          Start a model before running smoke, perf, or tool-eval. Cold loads can take several
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
              <span
                className={cn("lab-dot", healthy ? "lab-dot-live" : "lab-dot-idle")}
                role="img"
                aria-label={healthy ? "Endpoint healthy" : "Endpoint idle"}
              />
              <span className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.18em] text-lab-muted">
                {healthy ? "endpoint live" : "endpoint idle"}
              </span>
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
                    <Gauge pct={latestScore} label="Tool-eval score" />
                  </div>
                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    <Badge tone={scoreTone(latestScore)}>{latestRating || "scored"}</Badge>
                    <Link
                      href={`/evals/tool/${latestTool?.run_id}`}
                      className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.16em] text-lab-accent-bright underline-offset-4 hover:underline"
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
                    <Gauge pct={null} label="Tool-eval score" />
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
                    {runs.length}
                  </span>
                )}
              </Cell>
              <Cell label="Endpoint" className="bg-lab-panel">
                {loading ? (
                  <Skeleton className="h-3.5 w-20" />
                ) : (
                  <span className="inline-flex items-center gap-2">
                    <span
                      className={cn("lab-dot", healthy ? "lab-dot-live" : "lab-dot-idle")}
                      aria-hidden
                    />
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
            <span className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.18em] text-lab-muted">
              {healthy ? "armed" : "locked · start a model"}
            </span>
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
                loading={busy && smokeOk == null && !jobRunning}
                title={!healthy ? "Start a model on Serve first" : undefined}
              >
                Run smoke
              </Btn>
              {smokeOk != null && (
                <div
                  className="animus-chamfer-sm flex flex-wrap items-center gap-2.5 border border-lab-border-subtle bg-lab-editor px-3 py-2.5"
                  role="status"
                >
                  <span
                    className={cn("lab-dot", smokeOk ? "bg-lab-ok" : "bg-lab-danger")}
                    aria-hidden
                  />
                  <Badge tone={smokeOk ? "ok" : "danger"}>{smokeOk ? "PASS" : "FAIL"}</Badge>
                  <span className="min-w-0 break-all font-mono text-[12px] text-lab-text-dim">
                    {smokeOut || <Absent>no output</Absent>}
                  </span>
                </div>
              )}
            </div>
          </Panel>

          <Panel title="Perf bench" className="flex h-full flex-col">
            <div className="flex flex-1 flex-col gap-3.5 p-4">
              <Field label="Runner" htmlFor="eval-runner">
                <select
                  id="eval-runner"
                  className={inputCls}
                  value={runner}
                  onChange={(e) => setRunner(e.target.value as typeof runner)}
                  disabled={busy}
                >
                  <option value="workflow">workflow (concurrency sweep)</option>
                  <option value="prefill">prefill</option>
                  <option value="concurrency">concurrency (single N)</option>
                </select>
              </Field>
              <Field
                label="Concurrencies (comma-separated)"
                htmlFor="eval-conc"
                hint="Prefer concurrency + p95, not single-user tok/s only."
              >
                <Input
                  id="eval-conc"
                  value={conc}
                  onChange={(e) => setConc(e.target.value)}
                  placeholder="1,2,4"
                  disabled={busy}
                />
              </Field>
              <Btn
                onClick={() => void runPerf()}
                disabled={!healthy || busy}
                loading={busy && jobRunning}
                title={
                  !healthy ? "Start a model on Serve first" : busy ? "Job in progress" : undefined
                }
              >
                Start perf job
              </Btn>
            </div>
          </Panel>
        </div>
      </Section>

      {(jobStatus || logs) && (
        <Section className="lab-fade-in space-y-3" label="Job telemetry">
          <Panel
            title="Job log"
            action={
              <Badge
                tone={
                  jobStatus === "done"
                    ? "ok"
                    : jobStatus === "error" || jobStatus === "failed"
                      ? "danger"
                      : "accent"
                }
                dot={jobRunning}
              >
                {jobStatus || "idle"}
              </Badge>
            }
          >
            <div className="space-y-3 p-4">
              <ProgressBar
                value={Math.round((jobProgress || 0) * 100)}
                indeterminate={jobRunning && !(jobProgress > 0)}
                label={jobMsg || "Running…"}
              />
              <LogView text={logs} live={jobRunning} />
            </div>
          </Panel>
        </Section>
      )}

      {/* ── Run log ──────────────────────────────────────────────────────── */}
      <Section className="lab-rise lab-rise-3 space-y-3"
          label="Run log"
          meta={
            <span className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.18em] text-lab-muted tabular-nums">
              {loading ? "loading" : `${runs.length} recorded`}
            </span>
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
                  runs.slice(0, 20).map((r) => {
                    const isTool =
                      r.kind === "agentic_tool_eval" || String(r.kind || "").includes("tool");
                    const s =
                      typeof r.summary?.final_score === "number"
                        ? (r.summary.final_score as number)
                        : null;
                    return (
                      <tr key={r.run_id}>
                        <td className="font-mono text-[12px]">
                          {isTool ? (
                            <Link
                              href={`/evals/tool/${r.run_id}`}
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
                        Smoke or perf when the endpoint is healthy. Tool-eval results open on the
                        board.
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
