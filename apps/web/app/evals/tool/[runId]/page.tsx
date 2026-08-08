"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "@/lib/api";
import { Badge, Callout, EmptyState, Panel, PageSkeleton, btnClass } from "@/components/ui";
import { cn } from "@/lib/utils";

/* ═══════════════════════════════════════════════════════════════════════════
   Tool Eval run detail — Animus HUD readout, at board parity.
   Score hero → subscores → category bands → scenario board.
   Crimson is the only chromatic accent; ok/warn/danger stay reserved for
   pass/partial/fail state marks.
   ═══════════════════════════════════════════════════════════════════════════ */

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

/** Geometric state mark for a scenario verdict — same vocabulary everywhere. */
function statusMark(s?: string) {
  const tone = statusTone(s);
  return tone === "ok"
    ? "bg-lab-ok"
    : tone === "warn"
      ? "bg-lab-warn"
      : tone === "danger"
        ? "bg-lab-danger"
        : "bg-transparent shadow-[inset_0_0_0_1px_var(--color-lab-muted)]";
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

/** 0–100 gauge with a hairline graticule. Crimson fill; dimmed off-lead. */
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

/** Unknown → number | null, without changing any fetched shape. */
function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
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

  const passCount = scenarios.filter((s) => s.status === "pass").length;
  const partialCount = scenarios.filter((s) => s.status === "partial").length;
  const failCount = scenarios.filter((s) => s.status === "fail").length;
  const safetyOk = (ag.safety_gate as { passed?: boolean } | undefined)?.passed !== false;

  const totalPoints = num(scores.total_points);
  const maxPoints = num(scores.max_points);
  const totalScenarios = num(ag.total_scenarios) ?? scenarios.length;
  const deployability = num(ag.deployability);
  const responsiveness = num(ag.responsiveness);

  return (
    <div className="lab-fade-in space-y-6">
      <div className="page-header">
        <div className="min-w-0">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <Link
              href="/evals/tool"
              className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.18em] text-lab-muted transition-colors hover:text-lab-text"
            >
              ← Tool Eval
            </Link>
            <span aria-hidden className="h-3 w-px bg-[color:var(--animus-hairline)]" />
            <Badge tone="muted">{String(workload.preset || "run")}</Badge>
            <Badge tone={safetyOk ? "ok" : "danger"} dot>
              {safetyOk ? "safety ok" : "safety warn"}
            </Badge>
          </div>
          <h1 className="page-title">{String(modelId).split("/").pop()}</h1>
          <p className="page-sub font-mono text-[12px]">
            {modelId}
            {engine.version ? ` · vLLM ${String(engine.version)}` : ""}
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
        <Callout tone="danger" title="Couldn’t load run">
          {err}
        </Callout>
      )}

      {!envelope && !err && <PageSkeleton rows={4} />}

      {envelope && (
        <>
          {/* ── Verdict ──────────────────────────────────────────────────── */}
          <Section className="lab-rise lab-rise-1 space-y-3"
              label="Verdict"
              meta={
                <span className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.18em] tabular-nums text-lab-muted">
                  {totalScenarios} scenarios
                </span>
              }>

            <Panel>
              <div className="grid gap-px bg-lab-border-subtle md:grid-cols-[minmax(0,19rem)_minmax(0,1fr)]">
                <div className="relative bg-[color:var(--animus-accent-wash)] px-5 py-4">
                  <span aria-hidden className="absolute inset-y-0 left-0 w-[2px] bg-lab-accent" />
                  <CornerTicks />
                  <div className="animus-eyebrow">Final score</div>

                  {score != null ? (
                    <>
                      <div className="mt-1.5 flex items-end gap-1.5">
                        <span className="font-[family-name:var(--font-display)] text-[68px] font-semibold leading-[0.76] tracking-[0.01em] tabular-nums text-lab-text">
                          {score}
                        </span>
                        <span className="pb-2 font-[family-name:var(--font-display)] text-[14px] font-semibold uppercase leading-none tracking-[0.16em] text-lab-muted">
                          /100
                        </span>
                      </div>
                      <div className="mt-3">
                        <Gauge pct={score} label="Final score" />
                      </div>
                      <div className="mt-3 flex flex-wrap items-center gap-2">
                        <Badge tone={scoreTone(score)}>{String(ag.rating || "scored")}</Badge>
                        <span className="font-mono text-[11px] tabular-nums text-lab-muted">
                          {totalPoints != null && maxPoints != null
                            ? `${totalPoints}/${maxPoints} pts`
                            : "points unrecorded"}
                        </span>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="mt-3 font-[family-name:var(--font-display)] text-[28px] font-semibold uppercase leading-[0.9] tracking-[0.14em] text-lab-muted">
                        Not
                        <br />
                        scored
                      </div>
                      <div className="mt-3">
                        <Gauge pct={null} label="Final score" />
                      </div>
                    </>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-px bg-lab-border-subtle">
                  <Cell label="Deployability" className="bg-lab-panel">
                    {deployability != null ? (
                      <>
                        <span className="font-[family-name:var(--font-display)] text-[22px] font-semibold leading-none tabular-nums">
                          {deployability}
                        </span>
                        <Gauge
                          pct={deployability}
                          label="Deployability"
                          lead={false}
                          divisions={5}
                          className="mt-2 h-[4px]"
                        />
                        <div className="mt-1.5 text-[11px] text-lab-muted">
                          α-blend quality × safety
                        </div>
                      </>
                    ) : (
                      <Absent>awaiting</Absent>
                    )}
                  </Cell>
                  <Cell label="Responsiveness" className="bg-lab-panel">
                    {responsiveness != null ? (
                      <>
                        <span className="font-[family-name:var(--font-display)] text-[22px] font-semibold leading-none tabular-nums">
                          {responsiveness}
                        </span>
                        <Gauge
                          pct={responsiveness}
                          label="Responsiveness"
                          lead={false}
                          divisions={5}
                          className="mt-2 h-[4px]"
                        />
                        <div className="mt-1.5 text-[11px] text-lab-muted">
                          latency subscore (not quality)
                        </div>
                      </>
                    ) : (
                      <Absent>awaiting</Absent>
                    )}
                  </Cell>

                  <Cell label="Verdict split" className="col-span-2 bg-lab-panel">
                    {scenarios.length ? (
                      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                        {(
                          [
                            ["pass", passCount, "bg-lab-ok"],
                            ["partial", partialCount, "bg-lab-warn"],
                            ["fail", failCount, "bg-lab-danger"],
                          ] as const
                        ).map(([label, n, mark]) => (
                          <span key={label} className="inline-flex items-center gap-2">
                            <span className={cn("lab-dot", mark)} aria-hidden />
                            <span className="font-[family-name:var(--font-display)] text-[18px] font-semibold leading-none tabular-nums text-lab-text">
                              {n}
                            </span>
                            <span className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.16em] text-lab-muted">
                              {label}
                            </span>
                          </span>
                        ))}
                        <span
                          aria-hidden
                          className="ml-auto hidden h-3 w-px bg-[color:var(--animus-hairline)] sm:block"
                        />
                        <span className="font-mono text-[11px] tabular-nums text-lab-muted">
                          {Math.round((passCount / Math.max(1, scenarios.length)) * 100)}% clean
                        </span>
                      </div>
                    ) : (
                      <Absent>no breakdown</Absent>
                    )}
                  </Cell>

                  <Cell label="Stack" className="bg-lab-panel">
                    <span className="font-[family-name:var(--font-display)] text-[13px] font-semibold uppercase tracking-[0.08em]">
                      {String(engine.name || "vllm")} {String(engine.version || "")}
                    </span>
                  </Cell>
                  <Cell label="Image" className="bg-lab-panel">
                    {engine.image ? (
                      <span className="font-mono text-[11px] text-lab-muted" title={String(engine.image)}>
                        {String(engine.image)}
                      </span>
                    ) : (
                      <Absent>unrecorded</Absent>
                    )}
                  </Cell>
                </div>
              </div>
            </Panel>
          </Section>

          {/* ── Category bands ───────────────────────────────────────────── */}
          {!!categories.length && (
            <Section className="lab-rise lab-rise-2 space-y-3"
                label="Category bands"
                meta={
                  <span className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.18em] tabular-nums text-lab-muted">
                    {categories.length} bands
                  </span>
                }>

              <Panel>
                <div className="divide-y divide-[color:var(--color-lab-border-subtle)]">
                  {categories.map((c) => {
                    const pct = Number(c.percent ?? 0);
                    const p = num(c.pass_count) ?? 0;
                    const pa = num(c.partial_count) ?? 0;
                    const f = num(c.fail_count) ?? 0;
                    return (
                      <div
                        key={String(c.category)}
                        className="grid items-center gap-x-4 gap-y-2 px-5 py-3 sm:grid-cols-[minmax(0,11rem)_minmax(0,1fr)_auto]"
                      >
                        <div
                          className="truncate font-[family-name:var(--font-display)] text-[12px] font-semibold uppercase tracking-[0.1em] text-lab-text-dim"
                          title={String(c.label || c.category)}
                        >
                          {String(c.label || c.category)}
                        </div>

                        <div className="flex min-w-0 items-center gap-3">
                          <Gauge
                            pct={pct}
                            label={`${String(c.label || c.category)} band`}
                            lead={pct >= 90}
                            className="flex-1"
                          />
                          <span className="w-12 shrink-0 text-right font-[family-name:var(--font-display)] text-[16px] font-semibold leading-none tabular-nums text-lab-text">
                            {pct}%
                          </span>
                        </div>

                        <div className="flex shrink-0 items-center gap-3">
                          {(
                            [
                              [p, "bg-lab-ok", "pass"],
                              [pa, "bg-lab-warn", "partial"],
                              [f, "bg-lab-danger", "fail"],
                            ] as const
                          ).map(([n, mark, name]) => (
                            <span
                              key={name}
                              className="inline-flex items-center gap-1.5"
                              title={`${n} ${name}`}
                            >
                              <span
                                className={cn("lab-dot", mark, !n && "opacity-25")}
                                aria-hidden
                              />
                              <span className="font-mono text-[11px] tabular-nums text-lab-muted">
                                {n}
                              </span>
                            </span>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </Panel>
            </Section>
          )}

          {/* ── Scenario board ───────────────────────────────────────────── */}
          <Section className="lab-rise lab-rise-3 space-y-3"
              label="Scenario board"
              meta={
                <span className="flex items-center gap-3">
                  {(
                    [
                      [passCount, "bg-lab-ok", "pass"],
                      [partialCount, "bg-lab-warn", "partial"],
                      [failCount, "bg-lab-danger", "fail"],
                    ] as const
                  ).map(([n, mark, name]) => (
                    <span key={name} className="inline-flex items-center gap-1.5">
                      <span className={cn("lab-dot", mark)} aria-hidden />
                      <span className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.16em] tabular-nums text-lab-muted">
                        {n} {name}
                      </span>
                    </span>
                  ))}
                </span>
              }>

            <Panel
              title="Scenarios"
              action={
                <span className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.16em] tabular-nums text-lab-muted">
                  {scenarios.length || 0} recorded
                </span>
              }
            >
              {!scenarios.length ? (
                <EmptyState
                  title="No scenario breakdown"
                  icon={
                    <span
                      aria-hidden
                      className="block h-6 w-6 rotate-45 border border-lab-line/60 bg-[color:var(--animus-accent-wash)]"
                    />
                  }
                >
                  Raw TEB JSON missing — score still saved on the envelope.
                </EmptyState>
              ) : (
                <div className="divide-y divide-[color:var(--color-lab-border-subtle)]">
                  {scenarios.map((s) => {
                    const tone = statusTone(s.status);
                    return (
                      <div
                        key={s.scenario_id || s.summary}
                        className={cn(
                          "group relative flex flex-col gap-2 px-5 py-3.5 transition-colors sm:flex-row sm:items-start sm:gap-4",
                          "hover:bg-[color:var(--animus-accent-wash)]",
                        )}
                      >
                        <span
                          aria-hidden
                          className="absolute inset-y-0 left-0 w-[2px] bg-lab-accent opacity-0 transition-opacity group-hover:opacity-100"
                        />

                        <div className="flex w-full shrink-0 items-center gap-2.5 sm:w-[236px]">
                          <span className={cn("lab-dot", statusMark(s.status))} aria-hidden />
                          <Badge tone={tone}>{s.status || "unknown"}</Badge>
                          <span
                            className="min-w-0 truncate font-mono text-[12px] text-lab-text-dim"
                            title={s.scenario_id}
                          >
                            {s.scenario_id || <Absent>unnamed</Absent>}
                          </span>
                          <span className="ml-auto shrink-0 font-[family-name:var(--font-display)] text-[12px] font-semibold tabular-nums text-lab-muted">
                            {s.points ?? 0}
                            <span className="text-lab-muted/60">/2</span>
                          </span>
                        </div>

                        <div className="min-w-0 flex-1">
                          <div className="text-[13px] leading-snug text-lab-text">
                            {s.summary || s.title || <Absent>no summary</Absent>}
                          </div>
                          {s.note ? (
                            <div className="mt-1 flex items-start gap-1.5 text-[12px] leading-snug text-lab-warn">
                              <span
                                aria-hidden
                                className="mt-[5px] h-1.5 w-1.5 shrink-0 rotate-45 bg-lab-warn"
                              />
                              {s.note}
                            </div>
                          ) : null}
                          {!!s.tool_calls_made?.length && (
                            <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-[10px] text-lab-muted">
                              {s.tool_calls_made.slice(0, 3).map((t, i) => (
                                <span
                                  key={`${t}-${i}`}
                                  className="animus-chamfer-sm border border-lab-border-subtle px-1.5 py-[2px]"
                                >
                                  {t}
                                </span>
                              ))}
                              {s.tool_calls_made.length > 3 ? (
                                <span className="text-lab-muted/70">
                                  +{s.tool_calls_made.length - 3}
                                </span>
                              ) : null}
                            </div>
                          )}
                        </div>

                        <div className="flex shrink-0 items-center gap-4 sm:flex-col sm:items-end sm:gap-1">
                          <span className="font-mono text-[12px] tabular-nums text-lab-text-dim">
                            {s.duration_seconds != null ? (
                              `${s.duration_seconds.toFixed(1)}s`
                            ) : (
                              <Absent>no time</Absent>
                            )}
                          </span>
                          <span className="font-mono text-[10px] tabular-nums text-lab-muted">
                            {s.ttft_ms != null ? `ttft ${Math.round(s.ttft_ms)}ms` : ""}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </Panel>
          </Section>
        </>
      )}
    </div>
  );
}
