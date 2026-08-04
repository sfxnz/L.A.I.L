"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type ToolEvalBoardRow } from "@/lib/api";
import { Badge, Btn, EmptyState, Panel, btnClass } from "@/components/ui";
import { cn } from "@/lib/utils";

function shortDate(iso?: string) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export default function ToolEvalBoardPage() {
  const [rows, setRows] = useState<ToolEvalBoardRow[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [compare, setCompare] = useState<Awaited<ReturnType<typeof api.toolEvalCompare>> | null>(
    null,
  );
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(() => {
    api
      .toolEvalBoard(40)
      .then((r) => setRows(r.runs || []))
      .catch((e) => setErr(String(e.message || e)));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  function toggle(id: string) {
    setSelected((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 4) return prev;
      return [...prev, id];
    });
    setCompare(null);
  }

  async function runCompare() {
    if (selected.length < 2) return;
    setBusy(true);
    setErr(null);
    try {
      const c = await api.toolEvalCompare(selected);
      setCompare(c);
    } catch (e) {
      setErr(String((e as Error).message || e));
    } finally {
      setBusy(false);
    }
  }

  const selectedRows = useMemo(
    () => rows.filter((r) => selected.includes(r.run_id)),
    [rows, selected],
  );

  return (
    <div className="space-y-6">
      <div className="page-header">
        <div>
          <h1 className="page-title">Tool Eval</h1>
          <p className="page-sub">
            Leaderboard of tool-calling quality runs. Pick 2–4 models to compare which is actually
            better — score first, then safety and deployability.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Btn variant="secondary" size="sm" onClick={refresh}>
            Refresh
          </Btn>
          <Link href="/server" className={btnClass("secondary", "sm")}>
            Run on Serve
          </Link>
          <Btn size="sm" disabled={selected.length < 2 || busy} onClick={() => void runCompare()}>
            Compare ({selected.length})
          </Btn>
        </div>
      </div>

      {err && (
        <div className="rounded-[12px] border border-[rgba(255,69,58,0.28)] bg-[rgba(255,69,58,0.1)] px-3.5 py-2.5 text-[13px] text-lab-danger">
          {err}
        </div>
      )}

      {compare && (
        <Panel className="p-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-lab-muted">
                Head-to-head
              </div>
              <div className="mt-1 text-lg font-semibold tracking-[-0.02em]">
                Winner:{" "}
                <span className="text-lab-ok">{compare.winner_model || compare.winner_run_id}</span>
              </div>
              <p className="mt-1 text-[12px] text-lab-muted">
                Ranked by final tool-eval score. Bars below = category % (higher is better).
              </p>
            </div>
            <Btn variant="ghost" size="sm" onClick={() => setCompare(null)}>
              Dismiss
            </Btn>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {compare.runs.map((r) => {
              const win = r.run_id === compare.winner_run_id;
              return (
                <div
                  key={r.run_id}
                  className={cn(
                    "rounded-[14px] border p-4",
                    win
                      ? "border-[rgba(48,209,88,0.35)] bg-[rgba(48,209,88,0.08)]"
                      : "border-lab-border bg-lab-panel2",
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="truncate text-[13px] font-semibold tracking-[-0.01em]">
                      {r.model_short}
                    </div>
                    {win && <Badge tone="ok">best</Badge>}
                  </div>
                  <div className="mt-2 text-3xl font-semibold tabular-nums tracking-[-0.03em]">
                    {r.final_score ?? "—"}
                  </div>
                  <div className="mt-1 text-[11px] text-lab-muted">{r.rating || r.preset}</div>
                  <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] text-lab-muted">
                    <div>
                      Deploy <span className="text-lab-text tabular-nums">{r.deployability ?? "—"}</span>
                    </div>
                    <div>
                      Resp <span className="text-lab-text tabular-nums">{r.responsiveness ?? "—"}</span>
                    </div>
                    <div>
                      Safety{" "}
                      <span className={r.safety_passed === false ? "text-lab-danger" : "text-lab-ok"}>
                        {r.safety_passed === false ? "fail" : "ok"}
                      </span>
                    </div>
                    <div>
                      n <span className="text-lab-text tabular-nums">{r.total_scenarios ?? "—"}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-5 overflow-x-auto">
            <table className="lab-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  {compare.runs.map((r) => (
                    <th key={r.run_id}>{r.model_short}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {compare.metrics.map((m) => (
                  <tr key={m.metric}>
                    <td className="font-medium text-lab-text-dim">{m.metric}</td>
                    {compare.runs.map((r) => {
                      const v = m.values[r.run_id];
                      const nums = Object.values(m.values).filter(
                        (x): x is number => typeof x === "number",
                      );
                      const best = nums.length ? Math.max(...nums) : null;
                      const isBest = typeof v === "number" && best != null && v === best;
                      return (
                        <td
                          key={r.run_id}
                          className={cn(
                            "tabular-nums",
                            isBest ? "font-semibold text-lab-ok" : "text-lab-text-dim",
                          )}
                        >
                          {v == null ? "—" : String(v)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {!!compare.categories?.length && (
            <div className="mt-5 space-y-2">
              <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-lab-muted">
                Categories
              </div>
              {compare.categories.map((c) => (
                <div key={c.id} className="grid items-center gap-3 sm:grid-cols-[140px_1fr]">
                  <div className="truncate text-[12px] text-lab-text-dim" title={c.label}>
                    {c.label || c.id}
                  </div>
                  <div className="flex flex-col gap-1">
                    {compare.runs.map((r, i) => {
                      const pct = c.values[r.run_id];
                      const colors = [
                        "bg-lab-accent",
                        "bg-lab-ok",
                        "bg-lab-warn",
                        "bg-[#bf5af2]",
                      ];
                      return (
                        <div key={r.run_id} className="flex items-center gap-2">
                          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-lab-hover">
                            <div
                              className={cn("h-full rounded-full", colors[i % colors.length])}
                              style={{ width: `${Math.max(0, Math.min(100, Number(pct) || 0))}%` }}
                            />
                          </div>
                          <span className="w-10 text-right font-mono text-[11px] tabular-nums text-lab-muted">
                            {pct == null ? "—" : `${pct}%`}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Panel>
      )}

      <Panel
        title="Leaderboard"
        action={
          <span className="text-[11px] text-lab-muted">
            {rows.length} run{rows.length === 1 ? "" : "s"} · select to compare
          </span>
        }
      >
        {!rows.length ? (
          <EmptyState title="No tool-eval runs yet">
            Serve a model, open Serve → Agentic, run Tool Eval Bench. Results land here.
          </EmptyState>
        ) : (
          <div className="overflow-x-auto">
            <table className="lab-table">
              <thead>
                <tr>
                  <th className="w-10" />
                  <th>Model</th>
                  <th>Score</th>
                  <th>Rating</th>
                  <th>Preset</th>
                  <th>Deploy</th>
                  <th>Safety</th>
                  <th>When</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const on = selected.includes(r.run_id);
                  return (
                    <tr
                      key={r.run_id}
                      className={cn(on && "bg-lab-active/40")}
                      onClick={() => toggle(r.run_id)}
                    >
                      <td>
                        <input
                          type="checkbox"
                          checked={on}
                          onChange={() => toggle(r.run_id)}
                          onClick={(e) => e.stopPropagation()}
                          className="h-3.5 w-3.5 accent-[var(--color-lab-accent)]"
                          aria-label={`Select ${r.model_short}`}
                        />
                      </td>
                      <td>
                        <div className="font-medium tracking-[-0.01em] text-lab-text">
                          {r.model_short}
                        </div>
                        <div className="max-w-[220px] truncate font-mono text-[10px] text-lab-muted">
                          {r.model_id}
                          {r.quant ? ` · ${r.quant}` : ""}
                        </div>
                      </td>
                      <td>
                        <span className="text-lg font-semibold tabular-nums tracking-tight">
                          {r.final_score ?? "—"}
                        </span>
                      </td>
                      <td className="text-[12px] text-lab-muted">{r.rating || "—"}</td>
                      <td>
                        <Badge tone="muted">{r.preset || "—"}</Badge>
                      </td>
                      <td className="tabular-nums text-[12px]">{r.deployability ?? "—"}</td>
                      <td>
                        <Badge tone={r.safety_passed === false ? "danger" : "ok"}>
                          {r.safety_passed === false ? "warn" : "ok"}
                        </Badge>
                      </td>
                      <td className="whitespace-nowrap text-[12px] text-lab-muted">
                        {shortDate(r.created_at)}
                      </td>
                      <td>
                        <Link
                          href={`/evals/tool/${r.run_id}`}
                          onClick={(e) => e.stopPropagation()}
                          className="text-[12px] font-medium text-lab-accent-bright hover:underline"
                        >
                          Open →
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {selectedRows.length > 0 && !compare && (
        <p className="text-center text-[12px] text-lab-muted">
          Selected: {selectedRows.map((r) => r.model_short).join(" · ")}. Hit Compare when ready.
        </p>
      )}
    </div>
  );
}
