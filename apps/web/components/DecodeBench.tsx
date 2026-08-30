"use client";

import { useMemo, useState } from "react";
import { api, watchJob, type RunRow } from "@/lib/api";
import {
  CONCURRENCY_LEVELS,
  WORKLOAD_KINDS,
  WORKLOAD_LABELS,
  decodeRunLabel,
  sortConcurrencies,
  type WorkloadKind,
} from "@/lib/decode-bench";
import { Badge, Btn, LogView, Panel, ProgressBar } from "@/components/ui";
import { cn } from "@/lib/utils";

function Nil({ word = "Awaiting" }: { word?: "Awaiting" | "None" }) {
  return (
    <span className="inline-flex items-center gap-1.5 align-middle">
      <span
        aria-hidden
        className="h-[5px] w-[5px] shrink-0 rotate-45 border border-[color:var(--animus-hairline)]"
      />
      <span className="animus-eyebrow">{word}</span>
    </span>
  );
}

function rate(v: unknown): string | null {
  if (typeof v !== "number" || !Number.isFinite(v)) return null;
  return v >= 100 ? String(Math.round(v)) : v.toFixed(1);
}

export function DecodeBench({
  healthy,
  runs,
  onSettled,
}: {
  healthy: boolean;
  runs: RunRow[];
  onSettled?: () => void;
}) {
  const [kind, setKind] = useState<WorkloadKind>("prose");
  const [selected, setSelected] = useState<Set<number>>(() => new Set([1]));
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [logs, setLogs] = useState("");
  const [jobMsg, setJobMsg] = useState("");
  const [jobStatus, setJobStatus] = useState("");
  const [jobProgress, setJobProgress] = useState(0);

  const levels = sortConcurrencies(selected);
  const jobRunning = jobStatus === "running" || jobStatus === "queued";
  const jobFailed = jobStatus === "error" || jobStatus === "failed";
  const jobDone = jobStatus === "done" || jobStatus === "completed";

  const latest = useMemo(
    () => runs.find((r) => decodeRunLabel(r.summary) != null) || null,
    [runs],
  );
  const decode = rate(latest?.summary?.decode_tok_per_s_median_c1);
  const prefill = rate(latest?.summary?.prefill_tok_per_s_median_c1);
  const lastKind = decodeRunLabel(latest?.summary);
  const lastConcs = Array.isArray(latest?.summary?.concurrencies)
    ? (latest.summary.concurrencies as number[]).join(" → ")
    : null;

  function toggleLevel(n: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(n)) {
        if (next.size === 1) return next;
        next.delete(n);
      } else {
        next.add(n);
      }
      return next;
    });
  }

  async function run() {
    setErr(null);
    setBusy(true);
    setLogs("");
    setJobMsg("starting…");
    setJobStatus("running");
    setJobProgress(0);
    try {
      const { job_id } = await api.benchPerf({
        runner: "decode",
        workload: kind,
        concurrencies: levels,
      });
      watchJob(
        job_id,
        (chunk) => setLogs((l) => (l + chunk).slice(-80_000)),
        (s) => {
          setJobStatus(s.status);
          setJobMsg(s.message);
          setJobProgress(s.progress ?? 0);
          if (
            s.status === "done" ||
            s.status === "completed" ||
            s.status === "error" ||
            s.status === "failed"
          ) {
            setBusy(false);
            onSettled?.();
          }
        },
        () => {
          setBusy(false);
          onSettled?.();
        },
      );
    } catch (e) {
      setErr(String((e as Error).message || e));
      setBusy(false);
      setJobStatus("error");
    }
  }

  return (
    <Panel
      title="Bench"
      action={
        <span className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.16em] text-lab-muted">
          {healthy ? "ready" : "start a model"}
        </span>
      }
    >
      <div className="grid gap-px bg-lab-border-subtle lg:grid-cols-[minmax(0,1fr)_16rem]">
        <div className="space-y-4 bg-lab-panel p-4">
          <div>
            <div className="animus-eyebrow mb-2">Workload</div>
            <div className="flex flex-wrap gap-1.5" role="group" aria-label="Workload">
              {WORKLOAD_KINDS.map((k) => {
                const on = k === kind;
                return (
                  <button
                    key={k}
                    type="button"
                    disabled={busy}
                    onClick={() => setKind(k)}
                    aria-pressed={on}
                    className={cn(
                      "animus-chamfer-sm h-8 px-3 font-[family-name:var(--font-display)] text-[11px] font-semibold uppercase leading-none tracking-[0.12em] transition-[background,color,border-color] duration-150",
                      "focus-visible:outline-none! focus-visible:shadow-[inset_0_0_0_2px_var(--color-lab-line)]!",
                      on
                        ? "border border-[color:var(--animus-accent-edge)] bg-[color:color-mix(in_srgb,var(--color-lab-accent)_30%,#000)] bg-[image:var(--animus-selection-fade)] text-white"
                        : "border border-lab-border bg-transparent text-lab-text-dim hover:border-lab-line hover:text-lab-text",
                    )}
                  >
                    {WORKLOAD_LABELS[k]}
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <div className="mb-2 flex items-center justify-between gap-2">
              <div className="animus-eyebrow">Concurrency</div>
              <span className="font-mono text-[10px] tabular-nums text-lab-muted">
                {levels.join(" → ")}
              </span>
            </div>
            <div
              className="grid grid-cols-8 gap-1"
              role="group"
              aria-label="Concurrency 1 to 32"
            >
              {CONCURRENCY_LEVELS.map((n) => {
                const on = selected.has(n);
                return (
                  <button
                    key={n}
                    type="button"
                    disabled={busy}
                    onClick={() => toggleLevel(n)}
                    aria-pressed={on}
                    className={cn(
                      "animus-chamfer-sm h-7 min-w-0 font-[family-name:var(--font-display)] text-[10px] font-semibold leading-none tabular-nums transition-[background,color,border-color] duration-100",
                      "focus-visible:outline-none! focus-visible:shadow-[inset_0_0_0_2px_var(--color-lab-line)]!",
                      on
                        ? "border border-[color:var(--animus-accent-edge)] bg-[color:color-mix(in_srgb,var(--color-lab-accent)_30%,#000)] text-white"
                        : "border border-lab-border bg-transparent text-lab-muted hover:border-lab-line hover:text-lab-text",
                    )}
                  >
                    {n}
                  </button>
                );
              })}
            </div>
            <p className="mt-2 text-[11px] leading-snug text-lab-muted">
              Pick a kind. How many at once. Run.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Btn
              onClick={() => void run()}
              disabled={!healthy || busy || levels.length === 0}
              loading={busy && jobRunning}
              title={
                !healthy ? "Start a model on Serve first" : busy ? "Job in progress" : undefined
              }
            >
              Run
            </Btn>
            {err ? (
              <span className="text-[12px] text-lab-danger" role="alert">
                {err}
              </span>
            ) : null}
          </div>
        </div>

        <div className="bg-lab-panel">
          <div className="grid grid-cols-2 gap-px bg-lab-border-subtle">
            <div className="bg-lab-panel px-4 py-3">
              <div className="animus-eyebrow">Decode tok/s</div>
              <div className="mt-1.5 font-[family-name:var(--font-display)] text-[22px] font-semibold leading-none tabular-nums text-lab-text">
                {decode ?? <Nil />}
              </div>
            </div>
            <div className="bg-lab-panel px-4 py-3">
              <div className="animus-eyebrow">Prefill tok/s</div>
              <div className="mt-1.5 font-[family-name:var(--font-display)] text-[22px] font-semibold leading-none tabular-nums text-lab-text">
                {prefill ?? <Nil />}
              </div>
            </div>
            <div className="col-span-2 bg-lab-panel px-4 py-3">
              <div className="animus-eyebrow">Last run</div>
              <div className="mt-1.5 font-mono text-[11px] text-lab-text-dim">
                {latest && lastKind ? (
                  <>
                    {lastKind}
                    {lastConcs ? ` · ${lastConcs}` : ""}
                  </>
                ) : (
                  <Nil word="None" />
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {(jobStatus || logs) && (
        <div className="space-y-3 border-t border-lab-border-subtle p-4">
          <div className="flex items-center justify-between gap-2">
            <span className="animus-eyebrow">Run</span>
            <Badge
              tone={jobDone ? "ok" : jobFailed ? "danger" : "accent"}
              dot={jobRunning}
            >
              {jobDone ? "done" : jobFailed ? "failed" : jobStatus || "idle"}
            </Badge>
          </div>
          <ProgressBar
            value={Math.round((jobProgress || 0) * 100)}
            indeterminate={jobRunning && !(jobProgress > 0)}
            label={jobMsg || "Running…"}
          />
          <LogView text={logs} live={jobRunning} className="max-h-40" />
        </div>
      )}
    </Panel>
  );
}
