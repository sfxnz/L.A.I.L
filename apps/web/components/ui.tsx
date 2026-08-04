import { cn } from "@/lib/utils";
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode } from "react";

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
    "inline-flex items-center justify-center gap-1.5 font-medium tracking-[-0.01em] transition-[background,color,border-color,transform] duration-150 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-40",
    btnVariants[variant],
    btnSizes[size],
    className,
  );
}

export function Btn({
  variant = "primary",
  size = "md",
  className,
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: keyof typeof btnVariants;
  size?: keyof typeof btnSizes;
}) {
  return <button type={type} className={btnClass(variant, size, className)} {...props} />;
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
      {dot && <span className={cn("h-1.5 w-1.5 rounded-full", dotColor[tone])} />}
      {children}
    </span>
  );
}

export function Metric({
  label,
  value,
  sub,
  accent,
  tone,
  large,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
  tone?: "ok" | "danger" | "warn" | "muted";
  large?: boolean;
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
      <div
        className={cn(
          "mt-2 truncate text-xl font-semibold tracking-[-0.03em] tabular-nums",
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
    </Panel>
  );
}

export function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  return (
    <label className="block space-y-1.5">
      <span className="text-[12px] font-medium text-lab-text-dim">{label}</span>
      {children}
      {hint ? <span className="block text-[11px] text-lab-muted">{hint}</span> : null}
    </label>
  );
}

export const inputCls =
  "w-full rounded-[10px] border border-lab-border bg-lab-input px-3 py-2 text-[13px] text-lab-text outline-none placeholder:text-lab-muted/70 transition-[border-color,box-shadow] focus:border-lab-accent focus:ring-2 focus:ring-[rgba(10,132,255,0.2)]";

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn(inputCls, props.className)} {...props} />;
}

export function LogView({ text }: { text: string }) {
  return (
    <pre className="max-h-72 overflow-auto rounded-[12px] border border-lab-border bg-lab-editor p-3.5 font-mono text-[11px] leading-relaxed text-lab-text-dim whitespace-pre-wrap">
      {text || "—"}
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

export function StatusDot({ live }: { live: boolean | null }) {
  return (
    <span
      className={cn(
        "lab-dot",
        live === true ? "lab-dot-live" : live === false ? "lab-dot-down" : "lab-dot-idle",
      )}
      aria-hidden
    />
  );
}

export function EmptyState({
  children,
  title,
}: {
  children: ReactNode;
  title?: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-12 text-center">
      {title ? (
        <div className="text-[13px] font-medium tracking-[-0.01em] text-lab-text-dim">{title}</div>
      ) : null}
      <div className={cn("max-w-xs text-[12px] leading-relaxed text-lab-muted", title && "mt-1.5")}>
        {children}
      </div>
    </div>
  );
}
