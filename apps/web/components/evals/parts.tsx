import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

/*
  Readout atoms shared by the Evals surfaces (overview, tool-eval board, run
  detail). Every colour is a lab-* token so both worlds resolve; crimson is the
  only chromatic accent, ok/warn/danger stay reserved for verdicts.
*/

/** An absent reading is a deliberate HUD state — never a bare em-dash. */
export function Absent({ children = "none" }: { children?: ReactNode }) {
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
export function CornerTicks() {
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
 * A page section hung off the vertical spine: a hairline rail down the left edge,
 * each section branching off it with a crimson node + eyebrow + horizontal rule.
 */
export function Section({
  label,
  meta,
  children,
  className,
  id,
}: {
  label: string;
  meta?: ReactNode;
  children: ReactNode;
  className?: string;
  id?: string;
}) {
  return (
    <section id={id} className={cn("relative space-y-3 pl-4", className)}>
      <span aria-hidden className="absolute bottom-1 left-0 top-2 w-px bg-[color:var(--animus-hairline)]" />
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

/** 0–100 score bar with a hairline graticule. Crimson fill; dimmed when not the leader. */
export function ScoreGauge({
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
            lead ? "bg-lab-accent" : "bg-[color:color-mix(in_srgb,var(--color-lab-accent)_45%,transparent)]",
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
export function Cell({ label, children, className }: { label: string; children: ReactNode; className?: string }) {
  return (
    <div className={cn("min-w-0 px-4 py-3", className)}>
      <div className="animus-eyebrow truncate text-[10px]">{label}</div>
      <div className="mt-1.5 truncate text-[13px] text-lab-text">{children}</div>
    </div>
  );
}

export function scoreTone(score: number | null | undefined) {
  if (score == null) return "muted" as const;
  if (score >= 90) return "ok" as const;
  if (score >= 75) return "accent" as const;
  if (score >= 50) return "warn" as const;
  return "danger" as const;
}
