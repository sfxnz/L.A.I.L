"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { Badge, Btn, EmptyState, Panel, btnClass } from "@/components/ui";
import { cn } from "@/lib/utils";

type Scenario = {
  scenario_id?: string;
  status?: string;
  points?: number;
  summary?: string;
  note?: string | null;
  title?: string;
  category?: string;
  duration_seconds?: number;
  ttft_ms?: number;
  expected_behavior?: string;
  tool_calls_made?: string[];
};

function statusTone(s?: string): "ok" | "warn" | "danger" | "muted" {
  const v = (s || "").toLowerCase();
  if (v === "pass") return "ok";
  if (v === "partial") return "warn";
  if (v === "fail" || v === "failed") return "danger";
  return "muted";
}

export default function ToolEvalRunDetailPage() {
  const params = useParams();
  const runId = String(params?.runId || "");
  const [err, setErr] = useState<string | null>(null);
  const [envelope, setEnvelope] = useState<Record<string, unknown> | null>(null);
  const [raw, setRaw] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    if (!runId) return;
    api
      .run(runId)
      .then((r) => {
        setEnvelope((r.envelope as Record<string, unknown>) || null);
        setRaw((r.tool_eval_raw as Record<string, unknown>) || null);
      })
      .catch((e) => setErr(String(e.message || e)));
  }, [runId]);

  const ag = (envelope?.agentic as Record<string, unknown>) || {};
  const model = (envelope?.model as Record<string, unknown>) || {};
  const engine = (envelope?.engine as Record<string, unknown>) || {};
  const workload = (envelope?.workload as Record<string, unknown>) || {};
  const scores = (ag.scores as Record<string, unknown>) || {};
  const categories = (scores.category_scores as Array<Record<string, unknown>>) || [];

  const scenarios: Scenario[] = useMemo(() => {
    const fromScores = (scores.scenario_results as Scenario[]) || [];
    if (fromScores.length) return fromScores;
    const fromRaw = (raw?.scores as Record<string, unknown>)?.scenario_results as Scenario[];
    if (Array.isArray(fromRaw)) return fromRaw;
    return [];
  }, [scores, raw]);

  const modelId = String(model.id || "unknown");
  const score = ag.final_score as number | undefined;

  return (
    <div className="space-y-6">
      <div className="page-header">
        <div>
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <Link
              href="/evals/tool"
              className="text-[12px] font-medium text-lab-muted hover:text-lab-text"
            >
              ← Tool Eval
            </Link>
            <Badge tone="muted">{String(workload.preset || "run")}</Badge>
            {(ag.safety_gate as { passed?: boolean } | undefined)?.passed === false ? (
              <Badge tone="danger">safety warn</Badge>
            ) : (
              <Badge tone="ok">safety ok</Badge>
            )}
          </div>
          <h1 className="page-title">{String(modelId).split("/").pop()}</h1>
          <p className="page-sub font-mono text-[12px]">
            {modelId}
            {engine.version ? ` · vLLM ${engine.version}` : ""}
            {runId ? ` · ${runId}` : ""}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Link href="/evals/tool" className={btnClass("secondary", "sm")}>
            Leaderboard
          </Link>
          <Link href="/server" className={btnClass("ghost", "sm")}>
            Serve
          </Link>
        </div>
      </div>

      {err && (
        <div className="rounded-[12px] border border-[rgba(255,69,58,0.28)] bg-[rgba(255,69,58,0.1)] px-3.5 py-2.5 text-[13px] text-lab-danger">
          {err}
        </div>
      )}

      {!envelope && !err && <div className="text-[13px] text-lab-muted">Loading run…</div>}

      {envelope && (
        <>
          <div className="bento">
            <div className="bento-span-3">
              <Panel className="p-5">
                <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-lab-muted">
                  Final score
                </div>
                <div
                  className={cn(
                    "mt-2 text-4xl font-semibold tabular-nums tracking-[-0.04em]",
                    (score ?? 0) >= 90
                      ? "text-lab-ok"
                      : (score ?? 0) >= 75
                        ? "text-lab-accent-bright"
                        : "text-lab-text",
                  )}
                >
                  {score ?? "—"}
                </div>
                <div className="mt-1 text-[13px] text-lab-muted">{String(ag.rating || "")}</div>
                <div className="mt-3 text-[12px] text-lab-muted">
                  {String(scores.total_points ?? "—")}/{String(scores.max_points ?? "—")} pts ·{" "}
                  {String(ag.total_scenarios ?? scenarios.length)} scenarios
                </div>
              </Panel>
            </div>
            <div className="bento-span-3">
              <Panel className="p-5">
                <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-lab-muted">
                  Deployability
                </div>
                <div className="mt-2 text-3xl font-semibold tabular-nums tracking-[-0.03em]">
                  {String(ag.deployability ?? "—")}
                </div>
                <div className="mt-1 text-[12px] text-lab-muted">α-blend quality × safety</div>
              </Panel>
            </div>
            <div className="bento-span-3">
              <Panel className="p-5">
                <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-lab-muted">
                  Responsiveness
                </div>
                <div className="mt-2 text-3xl font-semibold tabular-nums tracking-[-0.03em]">
                  {String(ag.responsiveness ?? "—")}
                </div>
                <div className="mt-1 text-[12px] text-lab-muted">latency subscore (not quality)</div>
              </Panel>
            </div>
            <div className="bento-span-3">
              <Panel className="p-5">
                <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-lab-muted">
                  Stack
                </div>
                <div className="mt-2 text-[13px] font-medium tracking-[-0.01em]">
                  {String(engine.name || "vllm")} {String(engine.version || "")}
                </div>
                <div className="mt-1 truncate font-mono text-[11px] text-lab-muted">
                  {String(engine.image || "—")}
                </div>
              </Panel>
            </div>
          </div>

          {!!categories.length && (
            <Panel title="Categories">
              <div className="grid gap-2 p-3 sm:grid-cols-2 lg:grid-cols-3">
                {categories.map((c) => {
                  const pct = Number(c.percent ?? 0);
                  return (
                    <div
                      key={String(c.category)}
                      className="rounded-[12px] border border-lab-border-subtle px-3 py-3"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="text-[12px] font-medium text-lab-text">
                          {String(c.label || c.category)}
                        </div>
                        <span className="font-mono text-[12px] tabular-nums text-lab-muted">
                          {pct}%
                        </span>
                      </div>
                      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-lab-hover">
                        <div
                          className={cn(
                            "h-full rounded-full",
                            pct >= 90
                              ? "bg-lab-ok"
                              : pct >= 75
                                ? "bg-lab-accent"
                                : pct >= 50
                                  ? "bg-lab-warn"
                                  : "bg-lab-danger",
                          )}
                          style={{ width: `${Math.min(100, pct)}%` }}
                        />
                      </div>
                      <div className="mt-1.5 text-[10px] text-lab-muted">
                        {String(c.pass_count ?? 0)}p / {String(c.partial_count ?? 0)}Δ /{" "}
                        {String(c.fail_count ?? 0)}f
                      </div>
                    </div>
                  );
                })}
              </div>
            </Panel>
          )}

          <Panel
            title="Scenarios"
            action={
              <span className="text-[11px] text-lab-muted">
                {scenarios.filter((s) => s.status === "pass").length} pass ·{" "}
                {scenarios.filter((s) => s.status === "partial").length} partial ·{" "}
                {scenarios.filter((s) => s.status === "fail").length} fail
              </span>
            }
          >
            {!scenarios.length ? (
              <EmptyState title="No scenario breakdown">
                Raw TEB JSON missing — score still saved on the envelope.
              </EmptyState>
            ) : (
              <div className="divide-y divide-[var(--color-lab-border-subtle)]">
                {scenarios.map((s) => (
                  <div
                    key={s.scenario_id || s.summary}
                    className="flex flex-col gap-1 px-4 py-3 sm:flex-row sm:items-start sm:gap-4"
                  >
                    <div className="flex w-full shrink-0 items-center gap-2 sm:w-[200px]">
                      <Badge tone={statusTone(s.status)}>{s.status || "?"}</Badge>
                      <span className="font-mono text-[12px] text-lab-text-dim">
                        {s.scenario_id}
                      </span>
                      <span className="ml-auto font-mono text-[11px] tabular-nums text-lab-muted sm:ml-0">
                        {s.points ?? "—"}/2
                      </span>
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="text-[13px] text-lab-text">{s.summary || "—"}</div>
                      {s.note ? (
                        <div className="mt-0.5 text-[12px] text-lab-warn">{s.note}</div>
                      ) : null}
                      {!!s.tool_calls_made?.length && (
                        <div className="mt-1 font-mono text-[10px] text-lab-muted">
                          {s.tool_calls_made.slice(0, 3).join(" · ")}
                          {s.tool_calls_made.length > 3 ? " …" : ""}
                        </div>
                      )}
                    </div>
                    <div className="shrink-0 text-right font-mono text-[11px] tabular-nums text-lab-muted">
                      {s.duration_seconds != null ? `${s.duration_seconds.toFixed(1)}s` : ""}
                      {s.ttft_ms != null ? (
                        <div className="text-[10px]">ttft {Math.round(s.ttft_ms)}ms</div>
                      ) : null}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Panel>
        </>
      )}
    </div>
  );
}
