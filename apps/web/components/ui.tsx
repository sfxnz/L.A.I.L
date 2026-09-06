import type {
  ButtonHTMLAttributes,
  CSSProperties,
  InputHTMLAttributes,
  ReactNode,
} from "react";
import { useEffect, useRef, useState } from "react";
import { copyText } from "@/lib/streams/clipboard";
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

/**
 * The 10px eyebrow — condensed, uppercase, wide-tracked, muted. Every small
 * label/meta/state word on every page is one of these. `eyebrowClass` is for
 * elements that must stay a Link/button; `Eyebrow` for plain text.
 */
export function eyebrowClass(className?: string) {
  return cn(
    "font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.16em] text-lab-muted",
    className,
  );
}

export function Eyebrow({
  children,
  className,
  title,
}: {
  children: ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <span className={eyebrowClass(className)} title={title}>
      {children}
    </span>
  );
}

/**
 * The one null treatment: an absent value is a condensed STATE WORD behind a
 * hollow diamond, never a bare em-dash (which reads as broken data).
 * "Awaiting" = a live readout that hasn't reported yet; "None" = settled and
 * genuinely empty.
 */
export function Nil({ word = "Awaiting" }: { word?: "Awaiting" | "None" }) {
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

/** Hairline divider between readout cells. */
export function Tick({ className }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={cn("h-3 w-px shrink-0 bg-[color:var(--animus-hairline)]", className)}
    />
  );
}

/**
 * Sync ring — the ONE status glyph, with fixed semantics:
 *   serving  solid       (serving AND serving_worker — a headless TP worker is serving)
 *   idle     hollow
 *   loading  dashed
 *   offline  notched, danger
 *   token    hourglass   (LAIL_TOKEN required — never "offline")
 *   null     faint hollow (not probed yet)
 * Still by design. It blooms once, 700ms, on the transition TO serving.
 */
export type SyncState = "serving" | "idle" | "loading" | "offline" | "token";

export function syncStateFromNode(state?: string | null): SyncState | null {
  switch (state) {
    case "serving":
    case "serving_worker":
      return "serving";
    case "idle":
    case "loading":
    case "offline":
      return state;
    default:
      return null;
  }
}

const SYNC_LABEL: Record<SyncState, string> = {
  serving: "Serving",
  idle: "Idle",
  loading: "Loading",
  offline: "Offline",
  token: "Token required",
};

export function SyncRing({
  state,
  label,
  size = 16,
  className,
}: {
  state: SyncState | null;
  label?: string;
  size?: number;
  className?: string;
}) {
  const prev = useRef(state);
  const [bloom, setBloom] = useState(false);

  useEffect(() => {
    const was = prev.current;
    prev.current = state;
    if (state !== "serving" || was === "serving") return;
    setBloom(true);
    const t = window.setTimeout(() => setBloom(false), 700);
    return () => window.clearTimeout(t);
  }, [state]);

  const resolved = label ?? (state ? SYNC_LABEL[state] : "Unknown");
  const D = "M8 1.5 14.5 8 8 14.5 1.5 8Z";
  return (
    <span
      className={cn("sync-ring", bloom && "sync-ring-bloom", className)}
      role="img"
      aria-label={resolved}
      title={resolved}
      style={{ width: size, height: size }}
    >
      <svg viewBox="0 0 16 16" width={size} height={size} aria-hidden fill="none">
        {state === "serving" && <path d={D} className="fill-lab-ok stroke-lab-ok" strokeWidth="1" />}
        {state === "idle" && <path d={D} className="stroke-lab-muted" strokeWidth="1.25" />}
        {state === "loading" && (
          <path d={D} className="stroke-lab-warn" strokeWidth="1.25" strokeDasharray="2.6 1.8" />
        )}
        {state === "offline" && (
          <path
            d="M12.6 6.1 14.5 8 8 14.5 1.5 8 8 1.5 10.4 3.9"
            className="stroke-lab-danger"
            strokeWidth="1.25"
          />
        )}
        {state === "token" && (
          <>
            <path d={D} className="stroke-lab-muted" strokeWidth="0.9" opacity="0.45" />
            <path
              d="M5.3 4.6h5.4L8 8l2.7 3.4H5.3L8 8Z"
              className="stroke-lab-muted"
              strokeWidth="1"
              strokeLinejoin="round"
            />
          </>
        )}
        {state === null && <path d={D} className="stroke-lab-muted" strokeWidth="1" opacity="0.45" />}
      </svg>
    </span>
  );
}

const EASE_OUT_EXPO = (t: number) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t));

function reducedMotion() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Hero numeral — counts up over --dur-hero (700ms, easeOutExpo) into the value,
 * snaps under reduced motion. Writes the DOM directly (one rAF loop, no
 * re-renders). Renders `from` first so server and client agree, then animates.
 */
export function HeroNumber({
  value,
  from,
  format = (n) => String(Math.round(n)),
  className,
  label,
}: {
  value: number;
  /** start of the count-up on mount; defaults to `value` (no mount animation) */
  from?: number;
  format?: (n: number) => string;
  className?: string;
  label?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const shown = useRef(from ?? value);
  const fmt = useRef(format);
  fmt.current = format;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const start = shown.current;
    if (start === value || reducedMotion()) {
      shown.current = value;
      el.textContent = fmt.current(value);
      return;
    }
    const t0 = performance.now();
    let raf = 0;
    const step = (now: number) => {
      const t = Math.min(1, (now - t0) / 700);
      const v = start + (value - start) * EASE_OUT_EXPO(t);
      shown.current = v;
      el.textContent = fmt.current(v);
      if (t < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [value]);

  return (
    <span ref={ref} className={cn("animus-hero", className)} aria-label={label}>
      {format(from ?? value)}
    </span>
  );
}

/**
 * Sync bar — segmented progress with a % and a unit-bearing label. Anything
 * that waits longer than a second shows one of these (or `indeterminate`).
 */
export function SyncBar({
  value,
  label,
  unit,
  segments = 20,
  indeterminate,
  tone = "line",
  className,
}: {
  /** 0–100 */
  value?: number | null;
  label: string;
  /** unit-bearing readout, e.g. "512 tok" or "3 / 8 levels" */
  unit?: string;
  segments?: number;
  indeterminate?: boolean;
  tone?: "line" | "accent" | "warn" | "danger";
  className?: string;
}) {
  const pct = Math.max(0, Math.min(100, value ?? 0));
  const filled = Math.round((pct / 100) * segments);
  const fill = {
    line: "bg-lab-line-2",
    accent: "bg-lab-accent",
    warn: "bg-lab-warn",
    danger: "bg-lab-danger",
  }[tone];
  return (
    <div className={cn("space-y-1.5", className)}>
      <div className="flex items-center justify-between gap-2">
        <Eyebrow className="truncate">{label}</Eyebrow>
        <span className="lab-num flex shrink-0 items-baseline gap-2 font-mono text-[11px] text-lab-text-dim">
          {unit ? <span className="text-lab-muted">{unit}</span> : null}
          {!indeterminate && <span>{Math.round(pct)}%</span>}
        </span>
      </div>
      <div
        className="relative grid h-[4px] gap-px overflow-hidden"
        style={{ gridTemplateColumns: `repeat(${segments}, minmax(0, 1fr))` }}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={indeterminate ? undefined : Math.round(pct)}
        aria-label={label}
        aria-busy={indeterminate || undefined}
      >
        {Array.from({ length: segments }).map((_, i) => (
          <span
            key={i}
            className={cn(
              "h-full transition-colors duration-[var(--dur-sync)] ease-[var(--ease-animus-out)]",
              !indeterminate && i < filled ? fill : "bg-lab-hover",
            )}
          />
        ))}
        {indeterminate && (
          <span
            aria-hidden
            className={cn("lab-progress-indeterminate absolute inset-y-0 left-0 w-1/3 opacity-70", fill)}
          />
        )}
      </div>
    </div>
  );
}

/**
 * Sparkline — one series, ≤120 points, SVG. Colour comes from `currentColor`,
 * so pass a `text-lab-*` token class; never a literal.
 */
export function Sparkline({
  points,
  width = 120,
  height = 28,
  min,
  max,
  area = true,
  label,
  className,
}: {
  points: readonly number[];
  width?: number;
  height?: number;
  min?: number;
  max?: number;
  area?: boolean;
  label?: string;
  className?: string;
}) {
  const pts = points.slice(-120);
  const lo = min ?? Math.min(0, ...pts);
  const hi = Math.max(max ?? -Infinity, ...pts, lo + 1e-9);
  const n = pts.length;
  const x = (i: number) => (n > 1 ? (i / (n - 1)) * width : width);
  const y = (v: number) => height - 1 - ((v - lo) / (hi - lo)) * (height - 2);
  const line = pts.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      preserveAspectRatio="none"
      className={cn("block text-lab-line", className)}
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      {n > 1 && area && (
        <path
          d={`${line} L${width} ${height} L0 ${height} Z`}
          fill="currentColor"
          fillOpacity="0.12"
        />
      )}
      {n > 0 && (
        <path
          d={n > 1 ? line : `M0 ${y(pts[0]).toFixed(1)} L${width} ${y(pts[0]).toFixed(1)}`}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.25"
          vectorEffect="non-scaling-stroke"
        />
      )}
    </svg>
  );
}

/**
 * Stat — the eyebrow-over-value cell every instrument readout is built from
 * (cluster node cards, the Streams instrument bar, the hardware strip, Status
 * cards). `mono` wraps the value in the tabular mono telemetry style; without
 * it the caller owns the value's type (hero digits, badges, stacked lines).
 */
export function Stat({
  label,
  children,
  title,
  mono,
  className,
}: {
  label: string;
  children: ReactNode;
  title?: string;
  mono?: boolean;
  className?: string;
}) {
  return (
    <div className={cn("flex min-w-0 flex-col gap-1", className)} title={title}>
      <Eyebrow className="truncate">{label}</Eyebrow>
      {mono ? (
        <span className="lab-num truncate font-mono text-[13px] text-lab-text">{children}</span>
      ) : (
        children
      )}
    </div>
  );
}

/**
 * SparkStat — a Stat with its 60 s history beside it: label, last value + unit,
 * sparkline. Node cards (tok/s, power) and the bench hardware strip share it.
 */
export function SparkStat({
  label,
  unit = "",
  values,
  max,
  tone = "text-lab-line",
  format = (n) => String(Math.round(n)),
  title,
  width = 72,
  height = 22,
  className,
}: {
  label: string;
  unit?: string;
  values: readonly number[];
  max?: number;
  /** a text-lab-* token class; drives the sparkline colour via currentColor */
  tone?: string;
  format?: (n: number) => string;
  title?: string;
  width?: number;
  height?: number;
  className?: string;
}) {
  const last = values.length ? values[values.length - 1] : null;
  return (
    <div className={cn("flex min-w-0 items-center gap-2", className)} title={title}>
      <div className="min-w-0">
        <Eyebrow className="block text-[9px]">{label}</Eyebrow>
        <div className="lab-num mt-0.5 font-mono text-[12px] text-lab-text">
          {last !== null ? (
            <>
              {format(last)}
              <span className="text-lab-muted">{unit}</span>
            </>
          ) : (
            <Nil />
          )}
        </div>
      </div>
      <Sparkline
        points={values}
        width={width}
        height={height}
        min={0}
        max={max}
        className={cn("shrink-0", tone)}
        label={`${label} over the last 60 s`}
      />
    </div>
  );
}

/**
 * Copy feedback: `copy(label, text)` writes to the clipboard (with the insecure-
 * origin fallback) and flashes "<label> copied" for 1.6 s.
 */
export function useCopy(): [string | null, (label: string, text: string) => void] {
  const [flash, setFlash] = useState<string | null>(null);
  useEffect(() => {
    if (!flash) return;
    const t = setTimeout(() => setFlash(null), 1600);
    return () => clearTimeout(t);
  }, [flash]);
  const copy = (label: string, text: string) => {
    void copyText(text).then((ok) => setFlash(ok ? `${label} copied` : "Clipboard blocked — select the text instead"));
  };
  return [flash, copy];
}

/** Inline copy-to-clipboard control for ids, URLs and fingerprints. Flashes "copied" in place. */
export function CopyButton({
  text,
  label = "Copy",
  className,
}: {
  text: string;
  /** what was copied, e.g. "Endpoint" → "Endpoint copied" */
  label?: string;
  className?: string;
}) {
  const [flash, copy] = useCopy();
  return (
    <button
      type="button"
      onClick={() => copy(label, text)}
      aria-label={`Copy ${label.toLowerCase()}`}
      title={flash ?? `Copy ${label.toLowerCase()}`}
      className={cn(
        "animus-chamfer-sm inline-flex h-6 shrink-0 items-center gap-1 border border-lab-border px-1.5 font-[family-name:var(--font-display)] text-[9px] font-semibold uppercase leading-none tracking-[0.14em] text-lab-muted transition-colors duration-[var(--dur-tap)] hover:border-lab-line hover:text-lab-text",
        "focus-visible:outline-none! focus-visible:shadow-[inset_0_0_0_2px_var(--color-lab-line)]!",
        flash && "border-lab-ok text-lab-ok",
        className,
      )}
    >
      <svg viewBox="0 0 12 12" width="10" height="10" aria-hidden fill="none" stroke="currentColor" strokeWidth="1.1">
        <rect x="3.5" y="3.5" width="7" height="7" />
        <path d="M1.5 8.5v-7h7" />
      </svg>
      {flash ? "copied" : "copy"}
    </button>
  );
}

/**
 * Corridor — the empty state when nothing is loaded: a horizon grid, six
 * drifting fragments, and the instruction. "No memory loaded. Serve a model to
 * begin synchronization."
 */
const FRAGMENTS = [
  [12, 18],
  [28, 62],
  [46, 30],
  [61, 70],
  [77, 22],
  [90, 56],
] as const;

export function Corridor({
  title = "No memory loaded",
  children = "Serve a model to begin synchronization.",
  action,
  className,
}: {
  title?: string;
  children?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("animus-corridor flex items-center justify-center px-6 py-10", className)}>
      {FRAGMENTS.map(([l, t], i) => (
        <span
          key={i}
          aria-hidden
          className="animus-fragment"
          style={{ left: `${l}%`, top: `${t}%`, "--i": i } as CSSProperties}
        />
      ))}
      <div className="relative z-10 max-w-sm text-center">
        <div className="font-[family-name:var(--font-display)] text-[15px] font-semibold uppercase tracking-[0.18em] text-lab-text">
          {title}
        </div>
        <p className="mt-2 text-[13px] leading-relaxed text-lab-muted">{children}</p>
        {action ? <div className="mt-4 flex justify-center gap-2">{action}</div> : null}
      </div>
    </div>
  );
}

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
              "lab-num mt-1.5 truncate font-[family-name:var(--font-display)] font-semibold leading-[1.05] tracking-[0.005em]",
              large ? "text-[30px]" : "text-[26px]",
              valueTone,
            )}
          >
            {value}
          </div>
          {sub ? (
            <div className="lab-num mt-1 truncate font-mono text-[11px] text-lab-muted" title={sub}>
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
