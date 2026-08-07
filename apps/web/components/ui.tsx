import type {
  ButtonHTMLAttributes,
  InputHTMLAttributes,
  ReactNode,
} from "react";
import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

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
    <div className={cn("lab-card overflow-hidden", className)}>
      {title && (
        <div className="flex shrink-0 items-center justify-between gap-3 border-b border-lab-border-subtle px-4 py-3">
          <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-lab-muted">
            {title}
          </div>
          {action}
        </div>
      )}
      <div className={cn("min-h-0 flex-1", padded && "p-4")}>{children}</div>
    </div>
  );
}

export const btnVariants = {
  primary:
    "bg-lab-accent text-white hover:bg-lab-accent-bright border border-transparent shadow-[0_1px_2px_rgba(0,0,0,0.35)]",
  secondary:
    "bg-lab-panel2 text-lab-text-dim hover:bg-lab-hover hover:text-lab-text border border-lab-border",
  danger:
    "bg-[rgba(255,69,58,0.12)] text-lab-danger hover:bg-[rgba(255,69,58,0.18)] border border-[rgba(255,69,58,0.28)]",
  ghost:
    "bg-transparent text-lab-muted hover:text-lab-text hover:bg-lab-hover border border-transparent",
} as const;

export const btnSizes = {
  sm: "h-8 px-3 text-[12px] rounded-[8px]",
  md: "h-9 px-3.5 text-[13px] rounded-[10px]",
} as const;

export function btnClass(
  variant: keyof typeof btnVariants = "primary",
  size: keyof typeof btnSizes = "md",
  className?: string,
) {
  return cn(
    "inline-flex items-center justify-center gap-1.5 font-medium tracking-[-0.01em] transition-[background,color,border-color,transform,opacity] duration-150 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-40",
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
          strokeLinecap="round"
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
    ok: "text-lab-ok bg-[rgba(48,209,88,0.12)] border-[rgba(48,209,88,0.22)]",
    warn: "text-lab-warn bg-[rgba(255,214,10,0.1)] border-[rgba(255,214,10,0.22)]",
    danger: "text-lab-danger bg-[rgba(255,69,58,0.12)] border-[rgba(255,69,58,0.22)]",
    muted: "text-lab-muted bg-lab-hover border-lab-border-subtle",
    accent: "text-lab-accent-bright bg-[rgba(10,132,255,0.12)] border-[rgba(10,132,255,0.25)]",
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
        "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-medium tracking-[0.02em]",
        tones[tone],
      )}
    >
      {dot && <span className={cn("h-1.5 w-1.5 rounded-full", dotColor[tone])} aria-hidden />}
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
        "rounded-md bg-lab-hover",
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
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
  tone?: "ok" | "danger" | "warn" | "muted";
  large?: boolean;
  loading?: boolean;
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
    <Panel className="h-full p-4 transition-colors hover:border-lab-border-strong">
      <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-lab-muted">
        {label}
      </div>
      {loading ? (
        <div className="mt-2 space-y-2" aria-busy="true" aria-label={`Loading ${label}`}>
          <Skeleton className={cn("h-7", large ? "w-28" : "w-24")} />
          <Skeleton className="h-3 w-36" />
        </div>
      ) : (
        <>
          <div
            className={cn(
              "mt-2 truncate font-semibold tracking-[-0.03em] tabular-nums",
              large ? "text-2xl" : "text-xl",
              valueTone,
            )}
          >
            {value}
          </div>
          {sub ? (
            <div className="mt-1.5 truncate font-mono text-[11px] text-lab-muted" title={sub}>
              {sub}
            </div>
          ) : null}
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
        className="block text-[12px] font-medium text-lab-text-dim"
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

export const inputCls =
  "w-full rounded-[10px] border border-lab-border bg-lab-input px-3 py-2 text-[13px] text-lab-text outline-none placeholder:text-lab-muted/70 transition-[border-color,box-shadow] focus:border-lab-accent focus:ring-2 focus:ring-[rgba(10,132,255,0.2)] disabled:cursor-not-allowed disabled:opacity-50";

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn(inputCls, props.className)} {...props} />;
}

export function LogView({
  text,
  empty = "Waiting for output…",
  live,
}: {
  text: string;
  empty?: string;
  live?: boolean;
}) {
  const ref = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (!live || !ref.current) return;
    ref.current.scrollTop = ref.current.scrollHeight;
  }, [text, live]);

  return (
    <pre
      ref={ref}
      className="max-h-72 overflow-auto rounded-[12px] border border-lab-border bg-lab-editor p-3.5 font-mono text-[11px] leading-relaxed text-lab-text-dim whitespace-pre-wrap"
      aria-live={live ? "polite" : undefined}
    >
      {text || empty}
    </pre>
  );
}

export function ModeBanner({ mode }: { mode: "lab_safe" | "workflow_max" }) {
  const safe = mode === "lab_safe";
  return (
    <div
      className={cn(
        "rounded-[12px] border px-3.5 py-3 text-[13px] leading-snug",
        safe
          ? "border-[rgba(10,132,255,0.22)] bg-[rgba(10,132,255,0.08)] text-lab-text-dim"
          : "border-lab-border-strong bg-lab-panel2 text-lab-text-dim",
      )}
      role="status"
    >
      <span className={cn("font-semibold", safe ? "text-lab-accent-bright" : "text-lab-text")}>
        {safe ? "Lab Safe" : "Workflow Max"}
      </span>
      <span className="text-lab-muted"> — </span>
      {safe
        ? "util ≤ 0.4, leave headroom for OS / Hermes. Best for long-lived agent endpoints."
        : "util ~0.7–0.85 for large weights / long context. Watch free RAM; not set-and-forget."}
    </div>
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
      {icon ? <div className="mb-2.5 text-lab-muted opacity-70">{icon}</div> : null}
      {title ? (
        <div className="text-[13px] font-medium tracking-[-0.01em] text-lab-text-dim">{title}</div>
      ) : null}
      <div className={cn("max-w-xs text-[12px] leading-relaxed text-lab-muted", title && "mt-1")}>
        {children}
      </div>
      {action ? <div className="mt-3">{action}</div> : null}
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
    ok: "border-[rgba(48,209,88,0.28)] bg-[rgba(48,209,88,0.08)] text-lab-text-dim",
    warn: "border-[rgba(255,214,10,0.28)] bg-[rgba(255,214,10,0.08)] text-lab-text-dim",
    danger: "border-[rgba(255,69,58,0.28)] bg-[rgba(255,69,58,0.1)] text-lab-text-dim",
    muted: "border-lab-border bg-lab-panel2 text-lab-text-dim",
    accent: "border-[rgba(10,132,255,0.28)] bg-[rgba(10,132,255,0.08)] text-lab-text-dim",
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
        "flex flex-wrap items-start gap-3 rounded-[12px] border px-3.5 py-3 text-[13px] leading-snug",
        tones[tone],
        className,
      )}
    >
      <div className="min-w-0 flex-1">
        {title ? (
          <div className={cn("font-semibold tracking-[-0.01em]", titleTone[tone])}>{title}</div>
        ) : null}
        <div className={cn(title && "mt-0.5")}>{children}</div>
      </div>
      {(action || onDismiss) && (
        <div className="flex shrink-0 items-center gap-2">
          {action}
          {onDismiss ? (
            <button
              type="button"
              onClick={onDismiss}
              className="rounded-md px-1.5 py-0.5 text-[11px] font-medium text-lab-muted transition-colors hover:bg-lab-hover hover:text-lab-text"
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
          <span className="truncate">{label}</span>
          {!indeterminate && <span className="tabular-nums">{Math.round(pct)}%</span>}
        </div>
      ) : null}
      <div
        className="h-1.5 overflow-hidden rounded-full bg-lab-hover"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={indeterminate ? undefined : Math.round(pct)}
        aria-label={label || "Progress"}
        aria-busy={indeterminate || undefined}
      >
        <div
          className={cn(
            "h-full rounded-full bg-lab-accent transition-[width] duration-300 ease-out",
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
      className="inline-flex flex-wrap gap-0.5 rounded-full border border-lab-border bg-lab-panel p-1"
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
              "rounded-full font-medium tracking-[-0.01em] transition-colors disabled:opacity-40",
              size === "sm" ? "px-3 py-1 text-[12px]" : "px-3.5 py-1.5 text-[12px]",
              selected
                ? "bg-lab-active text-lab-text shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]"
                : "text-lab-muted hover:text-lab-text-dim",
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
      <div className="page-header !border-b-lab-border-subtle">
        <div className="space-y-2">
          <Skeleton className="h-3 w-24" />
          <Skeleton className="h-8 w-40" />
          <Skeleton className="h-3.5 w-72 max-w-full" />
        </div>
        <div className="flex gap-2">
          <Skeleton className="h-8 w-20 rounded-[8px]" />
          <Skeleton className="h-8 w-16 rounded-[8px]" />
        </div>
      </div>
      <div className="bento">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="bento-span-3">
            <Panel className="h-full space-y-3 p-4">
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-7 w-28" />
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
        "flex cursor-pointer items-start gap-2.5 rounded-[10px] border border-transparent px-1 py-1.5 text-[12px] text-lab-text-dim transition-colors hover:bg-lab-hover/60",
        disabled && "cursor-not-allowed opacity-50",
      )}
    >
      <input
        id={id}
        type="checkbox"
        className="mt-0.5 h-3.5 w-3.5 shrink-0 rounded border-lab-border accent-lab-accent"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="leading-snug">{children}</span>
    </label>
  );
}
