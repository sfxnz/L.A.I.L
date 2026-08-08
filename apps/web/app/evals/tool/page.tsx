"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, type ToolEvalBoardRow } from "@/lib/api";
import { Badge, Btn, Callout, EmptyState, Panel, Skeleton, btnClass } from "@/components/ui";
import { cn } from "@/lib/utils";

/* ═══════════════════════════════════════════════════════════════════════════
   Tool Eval board — Animus HUD readout.
   Score first, then the compare rig, then the leaderboard. Crimson is the only
   chromatic accent; ok/warn/danger stay reserved for pass/partial/fail state.
   ═══════════════════════════════════════════════════════════════════════════ */

function shortDate(iso?: string) {
  if (!iso) return null;
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

/** 0–100 gauge with a hairline graticule. Crimson fill; dimmed for non-leaders. */
function Gauge({
  pct,
  label,
  lead = true,
  divisions = 10,
  className,
}: {
  pct: number | null | undefined;
  label: string;
  lead?: boolean;
  divisions?: number;
  className?: string;
}) {
  const v = Math.max(0, Math.min(100, Number(pct) || 0));
  const step = 100 / divisions;
  return (
    <div
      className={cn("relative h-[6px] w-full overflow-hidden bg-lab-hover", className)}
      role="img"
      aria-label={pct == null ? `${label}: no reading` : `${label}: ${Math.round(v)} of 100`}
    >
      {pct != null && (
        <div
          className={cn(
            "h-full transition-[width] duration-700 ease-out",
            lead
              ? "bg-lab-accent"
              : "bg-[color:color-mix(in_srgb,var(--color-lab-accent)_45%,transparent)]",
          )}
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

export default function ToolEvalBoardPage() {
  const [rows, setRows] = useState<ToolEvalBoardRow[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<string[]>([]);
  const [compare, setCompare] = useState<Awaited<ReturnType<typeof api.toolEvalCompare>> | null>(
    null,
  );
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(() => {
    setLoading(true);
    api
      .toolEvalBoard(40)
      .then((r) => {
        setRows(r.runs || []);
        setErr(null);
      })
      .catch((e) => setErr(String(e.message || e)))
      .finally(() => setLoading(false));
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

  /* Derived from rows already in state — no extra fetch, no new state shape. */
  const champion = useMemo(() => {
    let best: ToolEvalBoardRow | null = null;
    for (const r of rows) {
      if (typeof r.final_score !== "number") continue;
      if (!best || (best.final_score ?? -1) < r.final_score) best = r;
    }
    return best;
  }, [rows]);

  const scored = useMemo(
    () => rows.filter((r) => typeof r.final_score === "number").length,
    [rows],
  );

  return (
    <div className="lab-fade-in space-y-6">
      <div className="page-header">
        <div className="min-w-0">
          <div className="animus-eyebrow mb-1.5 flex items-center gap-2">
            <span aria-hidden className="h-3 w-px bg-lab-accent" />
            Agentic benchmark
          </div>
          <h1 className="page-title">Tool Eval</h1>
          <p className="page-sub">
            Leaderboard of tool-calling quality runs. Pick 2–4 models to compare which is actually
            better — score first, then safety and deployability.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Btn variant="secondary" size="sm" onClick={refresh} loading={loading && rows.length > 0}>
            Refresh
          </Btn>
          <Link href="/server" className={btnClass("secondary", "sm")}>
            Run on Serve
          </Link>
          <Btn
            size="sm"
            disabled={selected.length < 2 || busy}
            loading={busy}
            title={selected.length < 2 ? "Select at least 2 runs" : undefined}
            onClick={() => void runCompare()}
          >
            Compare ({selected.length})
          </Btn>
        </div>
      </div>

      {err && (
        <Callout tone="danger" title="Board error" onDismiss={() => setErr(null)}>
          {err}
        </Callout>
      )}

      {/* ── Champion readout ─────────────────────────────────────────────── */}
      {(loading || rows.length > 0) && (
        <Section className="lab-rise lab-rise-1 space-y-3"
            label="Top score"
            meta={
              <span className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.18em] tabular-nums text-lab-muted">
                {loading ? "reading" : `${scored} scored / ${rows.length} runs`}
              </span>
            }>

          <Panel>
            <div className="grid gap-px bg-lab-border-subtle md:grid-cols-[minmax(0,19rem)_minmax(0,1fr)]">
              <div className="relative bg-[color:var(--animus-accent-wash)] px-5 py-4">
                <span aria-hidden className="absolute inset-y-0 left-0 w-[2px] bg-lab-accent" />
                <CornerTicks />
                <div className="animus-eyebrow">Best final score</div>

                {loading && !champion ? (
                  <div className="mt-3 space-y-2.5" aria-busy="true" aria-label="Loading top score">
                    <Skeleton className="h-12 w-32" />
                    <Skeleton className="h-[6px] w-full" />
                    <Skeleton className="h-3 w-40" />
                  </div>
                ) : champion?.final_score != null ? (
                  <>
                    <div className="mt-1.5 flex items-end gap-1.5">
                      <span className="font-[family-name:var(--font-display)] text-[68px] font-semibold leading-[0.76] tracking-[0.01em] tabular-nums text-lab-text">
                        {champion.final_score}
                      </span>
                      <span className="pb-2 font-[family-name:var(--font-display)] text-[14px] font-semibold uppercase leading-none tracking-[0.16em] text-lab-muted">
                        /100
                      </span>
                    </div>
                    <div className="mt-3">
                      <Gauge pct={champion.final_score} label="Best final score" />
                    </div>
                    <div
                      className="mt-3 truncate font-[family-name:var(--font-display)] text-[15px] font-semibold uppercase tracking-[0.1em] text-lab-text"
                      title={champion.model_id}
                    >
                      {champion.model_short}
                    </div>
                    <div className="mt-2 flex flex-wrap items-center gap-2">
                      <Badge tone={scoreTone(champion.final_score)}>
                        {champion.rating || "scored"}
                      </Badge>
                      <Link
                        href={`/evals/tool/${champion.run_id}`}
                        className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.16em] text-lab-accent-bright underline-offset-4 hover:underline"
                      >
                        Open run →
                      </Link>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="mt-3 font-[family-name:var(--font-display)] text-[28px] font-semibold uppercase leading-[0.9] tracking-[0.14em] text-lab-muted">
                      No scores
                      <br />
                      yet
                    </div>
                    <div className="mt-3">
                      <Gauge pct={null} label="Best final score" />
                    </div>
                  </>
                )}
              </div>

              <div className="grid grid-cols-2 gap-px bg-lab-border-subtle">
                <Cell label="Deployability" className="bg-lab-panel">
                  {loading && !champion ? (
                    <Skeleton className="h-3.5 w-12" />
                  ) : champion?.deployability != null ? (
                    <span className="font-[family-name:var(--font-display)] text-[20px] font-semibold leading-none tabular-nums">
                      {champion.deployability}
                    </span>
                  ) : (
                    <Absent>awaiting</Absent>
                  )}
                </Cell>
                <Cell label="Responsiveness" className="bg-lab-panel">
                  {loading && !champion ? (
                    <Skeleton className="h-3.5 w-12" />
                  ) : champion?.responsiveness != null ? (
                    <span className="font-[family-name:var(--font-display)] text-[20px] font-semibold leading-none tabular-nums">
                      {champion.responsiveness}
                    </span>
                  ) : (
                    <Absent>awaiting</Absent>
                  )}
                </Cell>
                <Cell label="Safety gate" className="bg-lab-panel">
                  {loading && !champion ? (
                    <Skeleton className="h-3.5 w-16" />
                  ) : champion ? (
                    <span className="inline-flex items-center gap-2">
                      <span
                        className={cn(
                          "lab-dot",
                          champion.safety_passed === false ? "bg-lab-danger" : "bg-lab-ok",
                        )}
                        aria-hidden
                      />
                      <span className="font-[family-name:var(--font-display)] text-[12px] font-semibold uppercase tracking-[0.14em]">
                        {champion.safety_passed === false ? "warn" : "clear"}
                      </span>
                    </span>
                  ) : (
                    <Absent>awaiting</Absent>
                  )}
                </Cell>
                <Cell label="Scenarios" className="bg-lab-panel">
                  {loading && !champion ? (
                    <Skeleton className="h-3.5 w-12" />
                  ) : champion?.total_scenarios != null ? (
                    <span className="font-[family-name:var(--font-display)] text-[20px] font-semibold leading-none tabular-nums">
                      {champion.total_scenarios}
                    </span>
                  ) : (
                    <Absent>awaiting</Absent>
                  )}
                </Cell>
                <Cell label="Preset" className="col-span-2 bg-lab-panel">
                  {loading && !champion ? (
                    <Skeleton className="h-3.5 w-40" />
                  ) : champion ? (
                    <span className="flex flex-wrap items-center gap-2">
                      <Badge tone="muted">{champion.preset || "default"}</Badge>
                      <span className="truncate font-mono text-[11px] text-lab-muted">
                        {champion.quant ? `${champion.quant} · ` : ""}
                        {shortDate(champion.created_at) || "no stamp"}
                      </span>
                    </span>
                  ) : (
                    <Absent>awaiting</Absent>
                  )}
                </Cell>
              </div>
            </div>
          </Panel>
        </Section>
      )}

      {loading && !rows.length && (
        <Section className="space-y-3" label="Leaderboard">
          <Panel title="Leaderboard" padded>
            <div className="space-y-3" aria-busy="true" aria-label="Loading tool-eval board">
              {[0, 1, 2, 3].map((i) => (
                <div key={i} className="flex items-center gap-3">
                  <Skeleton className="h-4 w-4" />
                  <Skeleton className="h-4 w-32" />
                  <Skeleton className="h-4 flex-1" />
                  <Skeleton className="h-6 w-14" />
                </div>
              ))}
            </div>
          </Panel>
        </Section>
      )}

      {!loading && !rows.length && !err && (
        <Section className="space-y-3" label="Leaderboard" meta={<Absent>no runs yet</Absent>}>
          <Panel padded>
            <EmptyState
              title="No tool-eval runs yet"
              icon={
                <span
                  aria-hidden
                  className="block h-6 w-6 rotate-45 border border-lab-line/60 bg-[color:var(--animus-accent-wash)]"
                />
              }
              action={
                <Link href="/server" className={btnClass("primary", "sm")}>
                  Open Serve → Agentic
                </Link>
              }
            >
              Run tool-eval-bench from Serve when a model is healthy. Results land here score-first.
            </EmptyState>
          </Panel>
        </Section>
      )}

      {/* ── Head-to-head ─────────────────────────────────────────────────── */}
      {compare && (
        <Section className="lab-fade-in space-y-3"
            label="Head to head"
            meta={
              <span className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.18em] tabular-nums text-lab-muted">
                {compare.runs.length} models
              </span>
            }>

          <Panel>
            <div className="flex flex-wrap items-start justify-between gap-3 border-b border-lab-border-subtle px-5 py-4">
              <div className="min-w-0">
                <div className="animus-eyebrow">Winner</div>
                <div className="mt-1 truncate font-[family-name:var(--font-display)] text-[22px] font-semibold uppercase leading-none tracking-[0.06em] text-lab-accent-bright">
                  {compare.winner_model || compare.winner_run_id}
                </div>
                <p className="mt-1.5 text-[12px] leading-snug text-lab-muted">
                  Ranked by final tool-eval score. Bars below = category % (higher is better).
                </p>
              </div>
              <Btn variant="ghost" size="sm" onClick={() => setCompare(null)}>
                Dismiss
              </Btn>
            </div>

            <div className="grid gap-px bg-lab-border-subtle sm:grid-cols-2 lg:grid-cols-4">
              {compare.runs.map((r) => {
                const win = r.run_id === compare.winner_run_id;
                return (
                  <div
                    key={r.run_id}
                    className={cn(
                      "relative min-w-0 px-4 py-4",
                      win ? "bg-[color:var(--animus-accent-wash)]" : "bg-lab-panel",
                    )}
                  >
                    {win && (
                      <span aria-hidden className="absolute inset-y-0 left-0 w-[2px] bg-lab-accent" />
                    )}
                    <div className="flex items-center justify-between gap-2">
                      <div
                        className="truncate font-[family-name:var(--font-display)] text-[13px] font-semibold uppercase tracking-[0.1em] text-lab-text"
                        title={r.model_id}
                      >
                        {r.model_short}
                      </div>
                      {win && <Badge tone="accent">best</Badge>}
                    </div>

                    <div className="mt-2.5 flex items-end gap-1">
                      {r.final_score != null ? (
                        <>
                          <span className="font-[family-name:var(--font-display)] text-[40px] font-semibold leading-[0.8] tabular-nums text-lab-text">
                            {r.final_score}
                          </span>
                          <span className="pb-1 font-[family-name:var(--font-display)] text-[11px] font-semibold uppercase leading-none tracking-[0.14em] text-lab-muted">
                            /100
                          </span>
                        </>
                      ) : (
                        <Absent>unscored</Absent>
                      )}
                    </div>

                    <div className="mt-2.5">
                      <Gauge
                        pct={r.final_score}
                        label={`${r.model_short} final score`}
                        lead={win}
                      />
                    </div>

                    <div className="mt-2 truncate font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase tracking-[0.16em] text-lab-muted">
                      {r.rating || r.preset || "unrated"}
                    </div>

                    <dl className="mt-3 grid grid-cols-2 gap-x-3 gap-y-2 border-t border-lab-border-subtle pt-3 text-[10px]">
                      <div className="min-w-0">
                        <dt className="animus-eyebrow text-[9px]">Deploy</dt>
                        <dd className="mt-0.5 font-mono text-[12px] tabular-nums text-lab-text">
                          {r.deployability ?? <Absent>n/a</Absent>}
                        </dd>
                      </div>
                      <div className="min-w-0">
                        <dt className="animus-eyebrow text-[9px]">Resp</dt>
                        <dd className="mt-0.5 font-mono text-[12px] tabular-nums text-lab-text">
                          {r.responsiveness ?? <Absent>n/a</Absent>}
                        </dd>
                      </div>
                      <div className="min-w-0">
                        <dt className="animus-eyebrow text-[9px]">Safety</dt>
                        <dd className="mt-0.5 flex items-center gap-1.5">
                          <span
                            className={cn(
                              "lab-dot",
                              r.safety_passed === false ? "bg-lab-danger" : "bg-lab-ok",
                            )}
                            aria-hidden
                          />
                          <span
                            className={cn(
                              "font-[family-name:var(--font-display)] text-[11px] font-semibold uppercase tracking-[0.12em]",
                              r.safety_passed === false ? "text-lab-danger" : "text-lab-ok",
                            )}
                          >
                            {r.safety_passed === false ? "fail" : "ok"}
                          </span>
                        </dd>
                      </div>
                      <div className="min-w-0">
                        <dt className="animus-eyebrow text-[9px]">Scenarios</dt>
                        <dd className="mt-0.5 font-mono text-[12px] tabular-nums text-lab-text">
                          {r.total_scenarios ?? <Absent>n/a</Absent>}
                        </dd>
                      </div>
                    </dl>
                  </div>
                );
              })}
            </div>

            <div className="border-t border-lab-border-subtle">
              <div className="flex items-center gap-3 px-5 pb-1 pt-4">
                <span aria-hidden className="h-3 w-px shrink-0 bg-lab-accent" />
                <h3 className="animus-eyebrow">Metric matrix</h3>
                <span aria-hidden className="animus-rule min-w-8 flex-1" />
              </div>
              <div className="overflow-x-auto px-2 pb-2">
                <table className="lab-table">
                  <thead>
                    <tr>
                      <th scope="col">Metric</th>
                      {compare.runs.map((r) => (
                        <th key={r.run_id} scope="col" className="whitespace-nowrap">
                          {r.model_short}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {compare.metrics.map((m) => {
                      const nums = Object.values(m.values).filter(
                        (x): x is number => typeof x === "number",
                      );
                      const best = nums.length ? Math.max(...nums) : null;
                      return (
                        <tr key={m.metric}>
                          <td className="whitespace-nowrap font-[family-name:var(--font-display)] text-[12px] font-semibold uppercase tracking-[0.1em] text-lab-text-dim">
                            {m.metric}
                          </td>
                          {compare.runs.map((r) => {
                            const v = m.values[r.run_id];
                            const isBest = typeof v === "number" && best != null && v === best;
                            return (
                              <td key={r.run_id} className="tabular-nums">
                                {v == null ? (
                                  <Absent>n/a</Absent>
                                ) : (
                                  <span
                                    className={cn(
                                      "inline-flex items-center gap-1.5 font-mono text-[12px]",
                                      isBest
                                        ? "font-semibold text-lab-accent-bright"
                                        : "text-lab-text-dim",
                                    )}
                                  >
                                    {isBest && (
                                      <span
                                        aria-hidden
                                        className="h-1.5 w-1.5 rotate-45 bg-lab-accent"
                                      />
                                    )}
                                    {String(v)}
                                  </span>
                                )}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {!!compare.categories?.length && (
              <div className="border-t border-lab-border-subtle">
                <div className="flex items-center gap-3 px-5 pb-3 pt-4">
                  <span aria-hidden className="h-3 w-px shrink-0 bg-lab-accent" />
                  <h3 className="animus-eyebrow">Category bands</h3>
                  <span aria-hidden className="animus-rule min-w-8 flex-1" />
                </div>
                <div className="divide-y divide-[color:var(--color-lab-border-subtle)]">
                  {compare.categories.map((c) => {
                    const vals = compare.runs
                      .map((r) => c.values[r.run_id])
                      .filter((x): x is number => typeof x === "number");
                    const lead = vals.length ? Math.max(...vals) : null;
                    return (
                      <div
                        key={c.id}
                        className="grid items-start gap-x-4 gap-y-2 px-5 py-3 sm:grid-cols-[150px_1fr]"
                      >
                        <div
                          className="truncate font-[family-name:var(--font-display)] text-[12px] font-semibold uppercase tracking-[0.1em] text-lab-text-dim"
                          title={c.label}
                        >
                          {c.label || c.id}
                        </div>
                        <div className="flex flex-col gap-1.5">
                          {compare.runs.map((r) => {
                            const pct = c.values[r.run_id];
                            const isLead =
                              typeof pct === "number" && lead != null && pct === lead;
                            return (
                              <div key={r.run_id} className="flex items-center gap-2.5">
                                <span
                                  className="w-24 shrink-0 truncate font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase tracking-[0.12em] text-lab-muted"
                                  title={r.model_short}
                                >
                                  {r.model_short}
                                </span>
                                <Gauge
                                  pct={pct ?? null}
                                  label={`${r.model_short} · ${c.label || c.id}`}
                                  lead={isLead}
                                  divisions={5}
                                  className="h-[5px] flex-1"
                                />
                                <span className="w-11 shrink-0 text-right font-mono text-[11px] tabular-nums text-lab-text-dim">
                                  {pct == null ? "··" : `${pct}%`}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </Panel>
        </Section>
      )}

      {/* ── Leaderboard ──────────────────────────────────────────────────── */}
      {rows.length > 0 && (
        <Section className="lab-rise lab-rise-2 space-y-3"
            label="Leaderboard"
            meta={
              <span className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.18em] tabular-nums text-lab-muted">
                {rows.length} run{rows.length === 1 ? "" : "s"} · select 2–4
              </span>
            }>

          <Panel
            title="Leaderboard"
            action={
              <span className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.16em] tabular-nums text-lab-muted">
                {selected.length}/4 selected
              </span>
            }
          >
            <div className="overflow-x-auto">
              <table className="lab-table">
                <thead>
                  <tr>
                    <th scope="col" className="w-10">
                      <span className="sr-only">Select for compare</span>
                    </th>
                    <th scope="col" className="w-10 text-right">
                      #
                    </th>
                    <th scope="col">Model</th>
                    <th scope="col">Score</th>
                    <th scope="col">Rating</th>
                    <th scope="col">Preset</th>
                    <th scope="col">Deploy</th>
                    <th scope="col">Safety</th>
                    <th scope="col">When</th>
                    <th scope="col">
                      <span className="sr-only">Open run</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => {
                    const on = selected.includes(r.run_id);
                    const atCap = !on && selected.length >= 4;
                    return (
                      <tr
                        key={r.run_id}
                        className={cn(
                          "cursor-pointer",
                          on && "bg-[color:var(--animus-accent-wash)]",
                        )}
                        onClick={() => toggle(r.run_id)}
                      >
                        <td className={cn(on && "shadow-[inset_2px_0_0_var(--color-lab-accent)]")}>
                          <input
                            type="checkbox"
                            checked={on}
                            onChange={() => toggle(r.run_id)}
                            onClick={(e) => e.stopPropagation()}
                            className="h-3.5 w-3.5 accent-[var(--color-lab-accent)]"
                            aria-label={`Select ${r.model_short} for compare`}
                            title={
                              atCap ? "Compare holds 4 runs — deselect one first" : undefined
                            }
                          />
                        </td>
                        <td className="text-right font-[family-name:var(--font-display)] text-[12px] font-semibold tabular-nums text-lab-muted">
                          {String(i + 1).padStart(2, "0")}
                        </td>
                        <td>
                          <div className="font-[family-name:var(--font-display)] text-[13px] font-semibold uppercase tracking-[0.06em] text-lab-text">
                            {r.model_short}
                          </div>
                          <div
                            className="max-w-[220px] truncate font-mono text-[10px] text-lab-muted"
                            title={r.model_id}
                          >
                            {r.model_id}
                            {r.quant ? ` · ${r.quant}` : ""}
                          </div>
                        </td>
                        <td className="min-w-[7.5rem]">
                          {r.final_score != null ? (
                            <>
                              <span className="font-[family-name:var(--font-display)] text-[22px] font-semibold leading-none tabular-nums text-lab-text">
                                {r.final_score}
                              </span>
                              <Gauge
                                pct={r.final_score}
                                label={`${r.model_short} score`}
                                lead={r.run_id === champion?.run_id}
                                divisions={5}
                                className="mt-1.5 h-[4px] w-[88px]"
                              />
                            </>
                          ) : (
                            <Absent>unscored</Absent>
                          )}
                        </td>
                        <td className="text-[12px] text-lab-muted">
                          {r.rating || <Absent>unrated</Absent>}
                        </td>
                        <td>
                          {r.preset ? (
                            <Badge tone="muted">{r.preset}</Badge>
                          ) : (
                            <Absent>default</Absent>
                          )}
                        </td>
                        <td className="font-mono text-[12px] tabular-nums">
                          {r.deployability ?? <Absent>n/a</Absent>}
                        </td>
                        <td>
                          <Badge tone={r.safety_passed === false ? "danger" : "ok"} dot>
                            {r.safety_passed === false ? "warn" : "ok"}
                          </Badge>
                        </td>
                        <td className="whitespace-nowrap font-mono text-[11px] tabular-nums text-lab-muted">
                          {shortDate(r.created_at) || <Absent>no stamp</Absent>}
                        </td>
                        <td>
                          <Link
                            href={`/evals/tool/${r.run_id}`}
                            onClick={(e) => e.stopPropagation()}
                            className="whitespace-nowrap font-[family-name:var(--font-display)] text-[11px] font-semibold uppercase tracking-[0.14em] text-lab-accent-bright underline-offset-4 hover:underline"
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
          </Panel>
        </Section>
      )}

      {selectedRows.length > 0 && !compare && (
        <div
          className="animus-chamfer-sm flex flex-wrap items-center gap-x-3 gap-y-2 border border-lab-border-subtle border-l-2 border-l-lab-accent bg-lab-panel2 px-4 py-3"
          role="status"
        >
          <span className="animus-eyebrow">Compare queue</span>
          <span aria-hidden className="h-3 w-px bg-[color:var(--animus-hairline)]" />
          <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-lab-text-dim">
            {selectedRows.map((r) => r.model_short).join("  ·  ")}
          </span>
          <span className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.16em] tabular-nums text-lab-muted">
            {selectedRows.length < 2
              ? "select 1 more"
              : `${selectedRows.length} armed · hit compare`}
          </span>
        </div>
      )}
    </div>
  );
}
