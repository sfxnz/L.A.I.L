import { cn } from "@/lib/utils";
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode } from "react";

export function Panel({
  children,
  className,
  title,
}: {
  children: ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-md border border-lab-border bg-lab-panel2",
        className,
      )}
    >
      {title && (
        <div className="border-b border-lab-border-subtle px-3 py-2 text-[11px] font-medium uppercase tracking-wide text-lab-muted">
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

export function Btn({
  variant = "primary",
  size = "md",
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "danger" | "ghost";
  size?: "sm" | "md";
}) {
  const styles = {
    primary:
      "bg-[#3c3c3c] text-lab-text hover:bg-[#4a4a4a] border border-[#4e4e4e]",
    secondary:
      "bg-transparent text-lab-text-dim hover:bg-lab-hover border border-lab-border",
    danger:
      "bg-lab-danger/15 text-lab-danger hover:bg-lab-danger/25 border border-lab-danger/30",
    ghost: "bg-transparent text-lab-muted hover:text-lab-text hover:bg-lab-hover border border-transparent",
  };
  const sizes = {
    sm: "h-7 px-2 text-[11px] rounded",
    md: "h-8 px-2.5 text-xs rounded-md",
  };
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-1.5 font-medium transition-colors disabled:pointer-events-none disabled:opacity-40",
        styles[variant],
        sizes[size],
        className,
      )}
      {...props}
    />
  );
}

export function Badge({
  children,
  tone = "muted",
}: {
  children: ReactNode;
  tone?: "ok" | "warn" | "danger" | "muted" | "accent";
}) {
  const tones = {
    ok: "text-lab-ok bg-lab-ok/10",
    warn: "text-lab-warn bg-lab-warn/10",
    danger: "text-lab-danger bg-lab-danger/10",
    muted: "text-lab-muted bg-white/5",
    accent: "text-lab-accent-bright bg-lab-accent/10",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium tracking-wide",
        tones[tone],
      )}
    >
      {children}
    </span>
  );
}

export function Metric({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
}) {
  return (
    <Panel className="p-3">
      <div className="text-[10px] font-medium uppercase tracking-wider text-lab-muted">
        {label}
      </div>
      <div
        className={cn(
          "mt-1 truncate text-lg font-semibold tracking-tight tabular-nums",
          accent ? "text-lab-accent-bright" : "text-lab-text",
        )}
      >
        {value}
      </div>
      {sub && <div className="mt-0.5 truncate text-[11px] text-lab-muted">{sub}</div>}
    </Panel>
  );
}

export function Field({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <label className="block space-y-1">
      <span className="text-[11px] font-medium text-lab-muted">{label}</span>
      {children}
    </label>
  );
}

export const inputCls =
  "w-full rounded-md border border-lab-border bg-lab-input px-2.5 py-1.5 text-xs text-lab-text outline-none placeholder:text-lab-muted/70 focus:border-lab-accent/50 focus:ring-1 focus:ring-lab-accent/30";

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn(inputCls, props.className)} {...props} />;
}

export function LogView({ text }: { text: string }) {
  return (
    <pre className="max-h-64 overflow-auto rounded-md border border-lab-border bg-lab-bg p-2.5 font-mono text-[11px] leading-relaxed text-lab-text-dim whitespace-pre-wrap">
      {text || "—"}
    </pre>
  );
}

export function ModeBanner({ mode }: { mode: "lab_safe" | "workflow_max" }) {
  const safe = mode === "lab_safe";
  return (
    <div
      className={cn(
        "rounded-md border px-3 py-2 text-xs",
        safe
          ? "border-lab-accent/25 bg-lab-accent/5 text-lab-accent-bright"
          : "border-lab-accent2/25 bg-lab-accent2/5 text-[#d8b4e2]",
      )}
    >
      {safe
        ? "Lab Safe — util ≤ 0.4, comparable benches, leave >60 GiB free when possible."
        : "Workflow Max — agent / long context. util ~0.7–0.85; keep ≥15–20 GiB free to avoid OOM."}
    </div>
  );
}
