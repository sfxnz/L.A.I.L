import type {
  ButtonHTMLAttributes,
  InputHTMLAttributes,
  ReactNode,
} from "react";
import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

/*
  L.A.I.L primitives — Animus HUD.

  Rules this file obeys (see app/globals.css for the token contract):
    · corners are CUT (.animus-chamfer*) or 2px, never pills
    · labels/actions ride the condensed display face, uppercase, wide tracking
    · crimson (lab-accent) is the ONLY chromatic accent; lab-line is structure
    · the selection tell is a crimson leading block fading right
      (--animus-selection-fade) over a darkened accent base so white text stays
      legible in BOTH worlds
    · every colour comes from a lab-* token or a color-mix of one, so the light
      "reconstruction" plate and the dark "in simulation" void both resolve

  Two mechanical gotchas encoded below, don't undo them:
    1. globals.css is UNLAYERED, so its rules outrank Tailwind utilities.
       Overriding one (e.g. .animus-bracketed::before offsets) needs a
       trailing `!`.
    2. clip-path clips outlines and box-shadows, so any chamfered control must
       carry its own INSET focus ring. Focusable things that can't afford that
       (inputs, tabs) get radius-2 instead of a chamfer.
*/

export function Panel({
  children,
  className,
  title,
  action,
  padded = false,
}: {
  children: ReactNode;
  className?: string;
  title?: string;
  action?: ReactNode;
  padded?: boolean;
}) {
  return (
    <div
      className={cn(
        // Brackets are pulled inside the padding box so `overflow-hidden`
        // (which pages rely on) can't eat them.
        "lab-card animus-bracketed overflow-hidden",
        "before:top-[3px]! before:left-[3px]! after:right-[3px]! after:bottom-[3px]!",
        className,
      )}
    >
      {title && (
        <div className="flex shrink-0 items-center justify-between gap-3 border-b border-lab-border-subtle px-4 py-2.5">
          <div className="flex min-w-0 items-center gap-2">
            <span aria-hidden className="h-3 w-px shrink-0 bg-lab-accent" />
            <div className="animus-eyebrow truncate">{title}</div>
          </div>
          {action}
        </div>
      )}
      <div className={cn("min-h-0 flex-1", padded && "p-4")}>{children}</div>
    </div>
  );
}

/**
 * Primary is the AC menu selection: a solid crimson leading block fading right.
 *
 * The base under the gradient is a *deep ink* mix of lab-accent, not the accent
 * itself — if the base is crimson too, the fade decays crimson-into-crimson and
 * the button reads as a flat block (verified in-browser). Mixing toward #000
 * keeps it dark in BOTH worlds, so the white label stays legible on the light
 * reconstruction plate as well as the dark void.
 */
export const btnVariants = {
  primary:
    "border border-[color:var(--animus-accent-edge)] bg-[color:color-mix(in_srgb,var(--color-lab-accent)_30%,#000)] bg-[image:var(--animus-selection-fade)] text-white hover:border-[color:var(--color-lab-accent-bright)] hover:bg-[color:color-mix(in_srgb,var(--color-lab-accent)_55%,#000)]",
  secondary:
    // A bare outlined rectangle reads as a stock web button. The leading
    // structure rule (thick left edge, hairline elsewhere) is the HUD tell —
    // the structural sibling of primary's crimson leading block.
    "border border-lab-border border-l-2 border-l-[color:var(--animus-hairline)] bg-transparent text-lab-text-dim hover:border-lab-line hover:border-l-[color:var(--color-lab-line)] hover:bg-lab-hover hover:text-lab-text",
  danger:
    "border border-[color:color-mix(in_srgb,var(--color-lab-danger)_40%,transparent)] bg-[color:color-mix(in_srgb,var(--color-lab-danger)_10%,transparent)] text-lab-danger hover:border-[color:var(--color-lab-danger)] hover:bg-[color:color-mix(in_srgb,var(--color-lab-danger)_17%,transparent)]",
  ghost:
    "border border-transparent bg-transparent text-lab-muted hover:bg-lab-hover hover:text-lab-text",
} as const;

export const btnSizes = {
  sm: "h-8 px-3.5 text-[11px]",
  md: "h-9 px-4 text-[12px]",
} as const;

export function btnClass(
  variant: keyof typeof btnVariants = "primary",
  size: keyof typeof btnSizes = "md",
  className?: string,
) {
  return cn(
    "animus-chamfer-sm inline-flex items-center justify-center gap-1.5 whitespace-nowrap font-[family-name:var(--font-display)] font-semibold uppercase leading-none tracking-[0.12em] transition-[background,color,border-color,transform,opacity] duration-150 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-40",
    // The chamfer clips the global focus outline, so carry an inset one.
    "focus-visible:outline-none! focus-visible:shadow-[inset_0_0_0_2px_var(--color-lab-line)]!",
    btnVariants[variant],
    btnSizes[size],
    className,
  );
}

export function Spinner({
  className,
  size = "md",
  label = "Loading",
}: {
  className?: string;
  size?: "sm" | "md" | "lg";
  label?: string;
}) {
  const dim = size === "sm" ? "h-3.5 w-3.5" : size === "lg" ? "h-5 w-5" : "h-4 w-4";
  return (
    <span
      role="status"
      aria-live="polite"
      className={cn("inline-flex items-center justify-center", className)}
    >
      <span className="sr-only">{label}</span>
      <svg
        className={cn("lab-spin text-current", dim)}
        viewBox="0 0 16 16"
        fill="none"
        aria-hidden
      >
        <circle
          cx="8"
          cy="8"
          r="6"
          stroke="currentColor"
          strokeOpacity="0.2"
          strokeWidth="2"
        />
        <path
          d="M14 8a6 6 0 0 0-6-6"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="square"
        />
      </svg>
    </span>
  );
}

export function Btn({
  variant = "primary",
  size = "md",
  className,
  type = "button",
  loading = false,
  children,
  disabled,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: keyof typeof btnVariants;
  size?: keyof typeof btnSizes;
  loading?: boolean;
}) {
  return (
    <button
      type={type}
      className={btnClass(
        variant,
        size,
        cn(loading && "min-w-[7.5rem]", className),
      )}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...props}
    >
      {loading ? <Spinner size="sm" label="Working" /> : null}
      {children}
    </button>
  );
}

export function Badge({
  children,
  tone = "muted",
  dot,
}: {
  children: ReactNode;
  tone?: "ok" | "warn" | "danger" | "muted" | "accent";
  dot?: boolean;
}) {
  const tones = {
    ok: "border-[color:color-mix(in_srgb,var(--color-lab-ok)_38%,transparent)] bg-[color:color-mix(in_srgb,var(--color-lab-ok)_12%,transparent)] text-lab-ok",
    warn: "border-[color:color-mix(in_srgb,var(--color-lab-warn)_38%,transparent)] bg-[color:color-mix(in_srgb,var(--color-lab-warn)_12%,transparent)] text-lab-warn",
    danger:
      "border-[color:color-mix(in_srgb,var(--color-lab-danger)_38%,transparent)] bg-[color:color-mix(in_srgb,var(--color-lab-danger)_12%,transparent)] text-lab-danger",
    muted: "border-lab-border bg-lab-hover text-lab-text-dim",
    accent:
      "border-[color:var(--animus-accent-edge)] bg-[color:var(--animus-accent-wash)] text-lab-accent-bright",
  };
  const dotColor = {
    ok: "bg-lab-ok",
    warn: "bg-lab-warn",
    danger: "bg-lab-danger",
    muted: "bg-lab-muted",
    accent: "bg-lab-accent",
  };
  return (
    <span
      className={cn(
        "animus-chamfer-sm inline-flex items-center gap-1.5 border px-2 py-[3px] font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.14em]",
        tones[tone],
      )}
    >
      {dot && (
        <span className={cn("h-1.5 w-1.5 rotate-45", dotColor[tone])} aria-hidden />
      )}
      {children}
    </span>
  );
}

export function Skeleton({
  className,
  pulse = true,
}: {
  className?: string;
  pulse?: boolean;
}) {
  return (
    <div
      className={cn(
        "rounded-[2px] bg-lab-hover",
        pulse && "lab-skeleton",
        className,
      )}
      aria-hidden
    />
  );
}

export function Metric({
  label,
  value,
  sub,
  accent,
  tone,
  large,
  loading,
  progress,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
  tone?: "ok" | "danger" | "warn" | "muted";
  large?: boolean;
  loading?: boolean;
  /** 0–100: renders an animated meter under the value. */
  progress?: number | null;
}) {
  const valueTone =
    tone === "ok"
      ? "text-lab-ok"
      : tone === "danger"
        ? "text-lab-danger"
        : tone === "warn"
          ? "text-lab-warn"
          : accent
            ? "text-lab-accent-bright"
            : "text-lab-text";

  return (
    <Panel className="h-full p-4">
      <div className="animus-eyebrow truncate">{label}</div>
      {loading ? (
        <div className="mt-2 space-y-2" aria-busy="true" aria-label={`Loading ${label}`}>
          <Skeleton className={cn("h-8", large ? "w-28" : "w-24")} />
          <Skeleton className="h-3 w-36" />
        </div>
      ) : (
        <>
          <div
            className={cn(
              "mt-1.5 truncate font-[family-name:var(--font-display)] font-semibold leading-[1.05] tracking-[0.005em] tabular-nums",
              large ? "text-[30px]" : "text-[26px]",
              valueTone,
            )}
          >
            {value}
          </div>
          {sub ? (
            <div className="mt-1 truncate font-mono text-[11px] tabular-nums text-lab-muted" title={sub}>
              {sub}
            </div>
          ) : null}
          {progress != null && (
            <div
              className="mt-3 h-[3px] overflow-hidden bg-lab-hover"
              role="img"
              aria-label={`${label}: ${Math.round(progress)}%`}
            >
              <div
                className={cn(
                  "h-full transition-[width] duration-700 ease-out",
                  progress < 15
                    ? "bg-lab-warn"
                    : tone === "danger"
                      ? "bg-lab-danger"
                      : "bg-lab-accent",
                )}
                style={{ width: `${Math.max(0, Math.min(100, progress))}%` }}
              />
            </div>
          )}
        </>
      )}
    </Panel>
  );
}

export function Field({
  label,
  children,
  hint,
  error,
  htmlFor,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
  error?: string;
  htmlFor?: string;
}) {
  const hintId = htmlFor ? `${htmlFor}-hint` : undefined;
  const errId = htmlFor ? `${htmlFor}-error` : undefined;
  return (
    <div className="block space-y-1.5">
      <label
        className="block font-[family-name:var(--font-display)] text-[11px] font-semibold uppercase tracking-[0.12em] text-lab-text-dim"
        htmlFor={htmlFor}
      >
        {label}
      </label>
      {children}
      {error ? (
        <span className="block text-[11px] text-lab-danger" role="alert" id={errId}>
          {error}
        </span>
      ) : hint ? (
        <span className="block text-[12px] leading-snug text-lab-text-dim/80" id={hintId}>
          {hint}
        </span>
      ) : null}
    </div>
  );
}

/**
 * Not chamfered on purpose: clip-path would swallow the focus ring, and inputs
 * are the one control that can't afford that. Radius stays at the 2px token and
 * focus grows a crimson leading edge instead.
 */
export const inputCls =
  "w-full rounded-[2px] border border-lab-border bg-lab-input px-3 py-2 text-[13px] text-lab-text outline-none placeholder:text-lab-muted/70 transition-[border-color,box-shadow] focus:border-lab-line focus:shadow-[inset_2px_0_0_var(--color-lab-accent)] disabled:cursor-not-allowed disabled:opacity-50";

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn(inputCls, props.className)} {...props} />;
}

export function LogView({
  text,
  empty = "Waiting for output…",
  live,
  className,
}: {
  text: string;
  empty?: string;
  live?: boolean;
  className?: string;
}) {
  const ref = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (!live || !ref.current) return;
    ref.current.scrollTop = ref.current.scrollHeight;
  }, [text, live]);

  return (
    <pre
      ref={ref}
      className={cn(
        "max-h-72 overflow-auto rounded-[2px] border border-lab-border bg-lab-editor p-3.5 font-mono text-[11px] leading-relaxed text-lab-text-dim whitespace-pre-wrap",
        live && "border-l-2 border-l-lab-accent",
        className,
      )}
      aria-live={live ? "polite" : undefined}
    >
      {text || empty}
    </pre>
  );
}

export function StatusDot({
  live,
  label,
}: {
  live: boolean | null;
  /** Accessible name; defaults from live state */
  label?: string;
}) {
  const resolved =
    label ??
    (live === true ? "Online" : live === false ? "Offline" : "Unknown");
  return (
    <span
      className={cn(
        "lab-dot",
        live === true ? "lab-dot-live" : live === false ? "lab-dot-down" : "lab-dot-idle",
      )}
      role="img"
      aria-label={resolved}
      title={resolved}
    />
  );
}

export function EmptyState({
  children,
  title,
  action,
  icon,
}: {
  children: ReactNode;
  title?: string;
  action?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-8 text-center">
      {icon ? <div className="mb-2.5 text-lab-line opacity-60">{icon}</div> : null}
      {title ? (
        <div className="font-[family-name:var(--font-display)] text-[13px] font-semibold uppercase tracking-[0.14em] text-lab-text-dim">
          {title}
        </div>
      ) : null}
      <div className={cn("max-w-xs text-[12px] leading-relaxed text-lab-muted", title && "mt-1.5")}>
        {children}
      </div>
      {action ? <div className="mt-3.5">{action}</div> : null}
    </div>
  );
}

/** Inline alert for recoverable errors / warnings / success — not jargon-only. */
export function Callout({
  tone = "muted",
  title,
  children,
  action,
  onDismiss,
  className,
}: {
  tone?: "ok" | "warn" | "danger" | "muted" | "accent";
  title?: string;
  children: ReactNode;
  action?: ReactNode;
  onDismiss?: () => void;
  className?: string;
}) {
  const tones = {
    ok: "border-[color:color-mix(in_srgb,var(--color-lab-ok)_30%,transparent)] border-l-[color:var(--color-lab-ok)] bg-[color:color-mix(in_srgb,var(--color-lab-ok)_9%,transparent)] text-lab-text-dim",
    warn: "border-[color:color-mix(in_srgb,var(--color-lab-warn)_30%,transparent)] border-l-[color:var(--color-lab-warn)] bg-[color:color-mix(in_srgb,var(--color-lab-warn)_9%,transparent)] text-lab-text-dim",
    danger:
      "border-[color:color-mix(in_srgb,var(--color-lab-danger)_30%,transparent)] border-l-[color:var(--color-lab-danger)] bg-[color:color-mix(in_srgb,var(--color-lab-danger)_10%,transparent)] text-lab-text-dim",
    muted: "border-lab-border border-l-lab-line bg-lab-panel2 text-lab-text-dim",
    accent:
      "border-[color:var(--animus-accent-edge)] border-l-[color:var(--color-lab-accent)] bg-[color:var(--animus-accent-wash)] text-lab-text-dim",
  };
  const titleTone = {
    ok: "text-lab-ok",
    warn: "text-lab-warn",
    danger: "text-lab-danger",
    muted: "text-lab-text",
    accent: "text-lab-accent-bright",
  };
  const role = tone === "danger" || tone === "warn" ? "alert" : "status";

  return (
    <div
      role={role}
      className={cn(
        "flex flex-wrap items-start gap-3 rounded-[2px] border border-l-2 px-3.5 py-3 text-[13px] leading-snug",
        tones[tone],
        className,
      )}
    >
      <div className="min-w-0 flex-1">
        {title ? (
          <div
            className={cn(
              "font-[family-name:var(--font-display)] font-semibold uppercase tracking-[0.1em]",
              titleTone[tone],
            )}
          >
            {title}
          </div>
        ) : null}
        <div className={cn(title && "mt-1")}>{children}</div>
      </div>
      {(action || onDismiss) && (
        <div className="flex shrink-0 items-center gap-2">
          {action}
          {onDismiss ? (
            <button
              type="button"
              onClick={onDismiss}
              className="animus-chamfer-sm border border-transparent px-2 py-1 font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.14em] text-lab-muted transition-colors hover:border-lab-border hover:text-lab-text focus-visible:outline-none! focus-visible:shadow-[inset_0_0_0_2px_var(--color-lab-line)]!"
              aria-label="Dismiss"
            >
              Dismiss
            </button>
          ) : null}
        </div>
      )}
    </div>
  );
}

export function ProgressBar({
  value,
  indeterminate,
  label,
  className,
}: {
  value?: number;
  indeterminate?: boolean;
  label?: string;
  className?: string;
}) {
  const pct = Math.max(0, Math.min(100, value ?? 0));
  return (
    <div className={cn("space-y-1.5", className)}>
      {label ? (
        <div className="flex items-center justify-between gap-2 text-[11px] text-lab-muted">
          <span className="truncate font-[family-name:var(--font-display)] uppercase tracking-[0.1em]">
            {label}
          </span>
          {!indeterminate && (
            <span className="shrink-0 font-mono tabular-nums text-lab-text-dim">
              {Math.round(pct)}%
            </span>
          )}
        </div>
      ) : null}
      <div
        className="h-[3px] overflow-hidden bg-lab-hover"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={indeterminate ? undefined : Math.round(pct)}
        aria-label={label || "Progress"}
        aria-busy={indeterminate || undefined}
      >
        <div
          className={cn(
            "h-full bg-lab-accent transition-[width] duration-300 ease-out",
            indeterminate && "lab-progress-indeterminate w-1/3",
          )}
          style={indeterminate ? undefined : { width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function SegmentedControl<T extends string>({
  value,
  onChange,
  options,
  ariaLabel,
  size = "md",
}: {
  value: T;
  onChange: (v: T) => void;
  options: ReadonlyArray<{ id: T; label: string; disabled?: boolean }>;
  ariaLabel: string;
  size?: "sm" | "md";
}) {
  const enabled = options.filter((o) => !o.disabled);

  function move(delta: number) {
    const idx = enabled.findIndex((o) => o.id === value);
    if (idx < 0) return;
    const next = enabled[(idx + delta + enabled.length) % enabled.length];
    if (next) onChange(next.id);
  }

  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className="animus-chamfer-sm inline-flex flex-wrap gap-0.5 border border-lab-border bg-lab-panel p-1"
      onKeyDown={(e) => {
        if (e.key === "ArrowRight" || e.key === "ArrowDown") {
          e.preventDefault();
          move(1);
        } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
          e.preventDefault();
          move(-1);
        } else if (e.key === "Home") {
          e.preventDefault();
          if (enabled[0]) onChange(enabled[0].id);
        } else if (e.key === "End") {
          e.preventDefault();
          if (enabled[enabled.length - 1]) onChange(enabled[enabled.length - 1].id);
        }
      }}
    >
      {options.map((opt) => {
        const selected = value === opt.id;
        return (
          <button
            key={opt.id}
            type="button"
            role="tab"
            aria-selected={selected}
            tabIndex={selected ? 0 : -1}
            disabled={opt.disabled}
            onClick={() => onChange(opt.id)}
            className={cn(
              // Radius, not chamfer: these keep the global focus ring for
              // arrow-key navigation.
              "rounded-[2px] font-[family-name:var(--font-display)] font-semibold uppercase tracking-[0.12em] transition-colors disabled:opacity-40",
              size === "sm" ? "px-3 py-1 text-[11px]" : "px-3.5 py-1.5 text-[12px]",
              selected
                ? "bg-[color:color-mix(in_srgb,var(--color-lab-accent)_30%,#000)] bg-[image:var(--animus-selection-fade)] text-white"
                : "text-lab-muted hover:text-lab-text",
            )}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

export function PageSkeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-6" aria-busy="true" aria-label="Loading page">
      <div className="page-header">
        <div className="space-y-2">
          <Skeleton className="h-3 w-24" />
          <Skeleton className="h-8 w-40" />
          <Skeleton className="h-3.5 w-72 max-w-full" />
        </div>
        <div className="flex gap-2">
          <Skeleton className="h-8 w-20" />
          <Skeleton className="h-8 w-16" />
        </div>
      </div>
      <div className="bento">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="bento-span-3">
            <Panel className="h-full space-y-3 p-4">
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-8 w-28" />
              <Skeleton className="h-3 w-36" />
            </Panel>
          </div>
        ))}
      </div>
    </div>
  );
}

export function CheckboxRow({
  checked,
  onChange,
  children,
  disabled,
  id,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  children: ReactNode;
  disabled?: boolean;
  id?: string;
}) {
  return (
    <label
      htmlFor={id}
      className={cn(
        "flex cursor-pointer items-start gap-2.5 rounded-[2px] border border-transparent px-1.5 py-1.5 text-[12px] text-lab-text-dim transition-colors hover:border-lab-border-subtle hover:bg-lab-hover/60",
        disabled && "cursor-not-allowed opacity-50",
      )}
    >
      <input
        id={id}
        type="checkbox"
        className="mt-0.5 h-3.5 w-3.5 shrink-0 border-lab-border accent-lab-accent"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="leading-snug">{children}</span>
    </label>
  );
}
