"use client";

import type { ReactNode } from "react";
import { CONCURRENCY_LEVELS, LEVEL_PRESETS } from "@/lib/bench/levels";
import { cn } from "@/lib/utils";

/**
 * Configuration chip — the AC selection tell (crimson leading block fading
 * right) when on, hairline when off. Every multi-select on the bench (packs,
 * tokens, sizes, the 1–32 grid) is one of these, so they all lock the same way.
 */
export function Chip({
  on,
  onClick,
  disabled,
  children,
  title,
  className,
  compact,
}: {
  on: boolean;
  onClick: () => void;
  disabled?: boolean;
  children: ReactNode;
  title?: string;
  className?: string;
  /** the 1–32 grid: tighter, numeric */
  compact?: boolean;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      aria-pressed={on}
      title={title}
      className={cn(
        "animus-chamfer-sm lab-num min-w-0 font-[family-name:var(--font-display)] font-semibold uppercase leading-none transition-[background,color,border-color,opacity] duration-[var(--dur-tap)] disabled:cursor-not-allowed disabled:opacity-50",
        "focus-visible:outline-none! focus-visible:shadow-[inset_0_0_0_2px_var(--color-lab-line)]!",
        compact ? "h-7 px-0 text-[10px]" : "h-8 px-3 text-[11px] tracking-[0.12em]",
        on
          ? "border border-[color:var(--animus-accent-edge)] bg-[color:color-mix(in_srgb,var(--color-lab-accent)_30%,#000)] bg-[image:var(--animus-selection-fade)] text-white"
          : "border border-lab-border bg-transparent text-lab-text-dim hover:border-lab-line hover:text-lab-text",
        className,
      )}
    >
      {children}
    </button>
  );
}

/** The 1–32 concurrency grid. Presets (1 2 4 8 16 32) carry their key hint. */
export function LevelGrid({
  selected,
  onToggle,
  disabled,
}: {
  selected: ReadonlySet<number>;
  onToggle: (n: number) => void;
  disabled?: boolean;
}) {
  return (
    <div className="grid grid-cols-8 gap-1" role="group" aria-label="Concurrency 1 to 32">
      {CONCURRENCY_LEVELS.map((n) => {
        const preset = LEVEL_PRESETS.indexOf(n);
        return (
          <Chip
            key={n}
            compact
            on={selected.has(n)}
            disabled={disabled}
            onClick={() => onToggle(n)}
            title={preset >= 0 ? `×${n} — key ${preset + 1}` : `×${n}`}
            className={cn(preset >= 0 && !selected.has(n) && "border-[color:var(--color-lab-border-strong)]")}
          >
            {n}
          </Chip>
        );
      })}
    </div>
  );
}
