"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, watchJob, type LabStatus, type RunRow } from "@/lib/api";
import { Badge, Btn, Field, Input, LogView, Panel, inputCls, btnClass } from "@/components/ui";

export default function EvalsPage() {
  const [status, setStatus] = useState<LabStatus | null>(null);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [logs, setLogs] = useState("");
  const [jobMsg, setJobMsg] = useState("");
  const [jobStatus, setJobStatus] = useState("");
  const [smokeOut, setSmokeOut] = useState<string | null>(null);
  const [smokeOk, setSmokeOk] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [runner, setRunner] = useState<"workflow" | "prefill" | "concurrency">("workflow");
  const [conc, setConc] = useState("1,2,4");
  const [err, setErr] = useState<string | null>(null);

  const refresh = () => {
    api.labStatus().then(setStatus).catch((e) => setErr(String(e.message || e)));
    api.runs().then(setRuns).catch(() => {});
  };

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 8000);
    return () => clearInterval(t);
  }, []);

  const healthy = !!(status?.serve && !status.serve.unreachable && status.serve.healthy);

  function track(jobId: string) {
    setBusy(true);
    setLogs("");
    setJobMsg("starting…");
    setJobStatus("running");
    watchJob(
      jobId,
      (chunk) => setLogs((l) => (l + chunk).slice(-80_000)),
      (s) => {
        setJobStatus(s.status);
        setJobMsg(s.message);
        if (s.status === "done" || s.status === "error" || s.status === "failed") {
          setBusy(false);
          refresh();
        }
      },
      () => {
        setBusy(false);
        refresh();
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
    <div className="space-y-5">
      <div className="page-header">
        <div>
          <h1 className="page-title">Evals</h1>
          <p className="page-sub">Smoke, perf, and tool-eval quality vs the live serve</p>
        </div>
        <div className="flex gap-2">
          <Link href="/evals/tool" className={btnClass("secondary", "sm")}>
            Tool Eval board
          </Link>
          <Btn variant="secondary" size="sm" onClick={refresh}>
            Refresh
          </Btn>
          <Link href="/server" className={btnClass("primary", "sm")}>
            Open Serve
          </Link>
        </div>
      </div>

      {!healthy && (
        <div className="rounded-[12px] border border-[rgba(255,214,10,0.22)] bg-[rgba(255,214,10,0.08)] px-4 py-3 text-[13px] text-lab-text-dim">
          vLLM is not healthy. Start a model on{" "}
          <Link href="/server" className="font-semibold text-lab-warn underline-offset-2 hover:underline">
            Serve
          </Link>{" "}
          before running evals.
        </div>
      )}

      {err && (
        <div className="rounded-[12px] border border-[rgba(255,69,58,0.28)] bg-[rgba(255,69,58,0.1)] px-4 py-3 text-[13px] text-lab-danger">
          {err}
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Smoke">
          <div className="space-y-3 p-4">
            <p className="text-[13px] text-lab-muted">
              Quick completion check (12×17 → 204). Catches empty / garbage output before you bench.
            </p>
            <Btn onClick={runSmoke} disabled={!healthy || busy}>
              Run smoke
            </Btn>
            {smokeOk != null && (
              <div className="flex items-center gap-2">
                <Badge tone={smokeOk ? "ok" : "danger"}>{smokeOk ? "PASS" : "FAIL"}</Badge>
                <span className="font-mono text-[12px] text-lab-text-dim">{smokeOut}</span>
              </div>
            )}
          </div>
        </Panel>

        <Panel title="Perf bench">
          <div className="space-y-3 p-4">
            <Field label="Runner">
              <select
                className={inputCls}
                value={runner}
                onChange={(e) => setRunner(e.target.value as typeof runner)}
              >
                <option value="workflow">workflow (concurrency sweep)</option>
                <option value="prefill">prefill</option>
                <option value="concurrency">concurrency (single N)</option>
              </select>
            </Field>
            <Field label="Concurrencies (comma-separated)">
              <Input value={conc} onChange={(e) => setConc(e.target.value)} placeholder="1,2,4" />
            </Field>
            <Btn onClick={runPerf} disabled={!healthy || busy}>
              Start perf job
            </Btn>
            <p className="text-[12px] text-lab-muted">
              Results land under lab runs. Prefer concurrency + prefill/decode, not single-user tok/s only.
            </p>
          </div>
        </Panel>
      </div>

      {(jobStatus || logs) && (
        <Panel title="Job log">
          <div className="space-y-2 p-4">
            <div className="flex flex-wrap items-center gap-2 text-[13px]">
              <Badge
                tone={
                  jobStatus === "done"
                    ? "ok"
                    : jobStatus === "error" || jobStatus === "failed"
                      ? "danger"
                      : "accent"
                }
              >
                {jobStatus || "—"}
              </Badge>
              <span className="text-lab-muted">{jobMsg}</span>
            </div>
            <LogView text={logs} />
          </div>
        </Panel>
      )}

      <Panel title="Recent runs">
        <div className="overflow-x-auto p-2">
          <table className="lab-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Kind</th>
                <th>Intent</th>
                <th>Model</th>
                <th>When</th>
              </tr>
            </thead>
            <tbody>
              {runs.slice(0, 20).map((r) => (
                <tr key={r.run_id}>
                  <td className="font-mono text-[12px]">
                    {r.kind === "agentic_tool_eval" ? (
                      <Link
                        href={`/evals/tool/${r.run_id}`}
                        className="text-lab-accent-bright hover:underline"
                      >
                        {r.run_id}
                      </Link>
                    ) : (
                      r.run_id
                    )}
                  </td>
                  <td>
                    {r.kind === "agentic_tool_eval" ? (
                      <span>
                        tool-eval{" "}
                        {r.summary?.final_score != null ? (
                          <Badge tone="ok">{String(r.summary.final_score)}</Badge>
                        ) : null}
                      </span>
                    ) : (
                      r.kind
                    )}
                  </td>
                  <td>{r.intent || "—"}</td>
                  <td className="max-w-[220px] truncate">{r.model_id?.split("/").pop() || "—"}</td>
                  <td className="text-lab-muted">{r.created_at?.slice(0, 19) || "—"}</td>
                </tr>
              ))}
              {!runs.length && (
                <tr>
                  <td colSpan={5} className="text-lab-muted">
                    No runs yet
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
