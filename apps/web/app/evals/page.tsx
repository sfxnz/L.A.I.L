"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { api, watchJob, type LabStatus, type RunRow } from "@/lib/api";
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
      setErr(String((e as Error).message || e));
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
    <div className="space-y-5 lab-fade-in">
      <div className="page-header">
        <div>
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

      {!loading && !healthy && (
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

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Smoke" className="flex h-full flex-col">
          <div className="flex flex-1 flex-col space-y-3 p-4">
            <p className="text-[13px] text-lab-muted">
              Quick completion check (12×17 → 204). Catches empty / garbage output before you bench.
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
                className="flex flex-wrap items-center gap-2 rounded-[10px] border border-lab-border-subtle bg-lab-editor px-3 py-2"
                role="status"
              >
                <Badge tone={smokeOk ? "ok" : "danger"}>{smokeOk ? "PASS" : "FAIL"}</Badge>
                <span className="min-w-0 break-all font-mono text-[12px] text-lab-text-dim">
                  {smokeOut}
                </span>
              </div>
            )}
          </div>
        </Panel>

        <Panel title="Perf bench" className="flex h-full flex-col">
          <div className="flex flex-1 flex-col space-y-3 p-4">
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
              title={!healthy ? "Start a model on Serve first" : busy ? "Job in progress" : undefined}
            >
              Start perf job
            </Btn>
          </div>
        </Panel>
      </div>

      {(jobStatus || logs) && (
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
              {jobStatus || "—"}
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
      )}

      <Panel title="Recent runs">
        <div className="overflow-x-auto p-2">
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
                    <div className="space-y-2" aria-busy="true" aria-label="Loading runs">
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
                  return (
                    <tr key={r.run_id}>
                      <td className="font-mono text-[12px]">
                        {isTool ? (
                          <Link
                            href={`/evals/tool/${r.run_id}`}
                            className="text-lab-accent-bright underline-offset-2 hover:underline"
                          >
                            {r.run_id}
                          </Link>
                        ) : (
                          r.run_id
                        )}
                      </td>
                      <td>
                        {r.kind === "agentic_tool_eval" ? (
                          <span className="inline-flex items-center gap-1.5">
                            tool-eval{" "}
                            {typeof r.summary?.final_score === "number" ? (
                              <Badge tone="ok">{String(r.summary.final_score)}</Badge>
                            ) : null}
                          </span>
                        ) : (
                          <span className="font-mono text-[11px] text-lab-muted">{r.kind}</span>
                        )}
                      </td>
                      <td>{r.intent || "—"}</td>
                      <td className="max-w-[220px] truncate">
                        {r.model_id?.split("/").pop() || "—"}
                      </td>
                      <td className="text-lab-muted">{r.created_at?.slice(0, 19) || "—"}</td>
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
                      Smoke or perf when the endpoint is healthy. Tool-eval results open on the board.
                    </EmptyState>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
