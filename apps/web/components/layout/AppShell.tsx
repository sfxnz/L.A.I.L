"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  Boxes,
  Cable,
  FlaskConical,
  LayoutDashboard,
  Server,
  Settings2,
} from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useLabStore } from "@/lib/store";
import { WORKSPACE_NAV } from "@/lib/ide-chrome";
import { cn } from "@/lib/utils";
import { Badge, StatusDot } from "@/components/ui";

const NAV_ICONS: Record<string, React.ComponentType<{ className?: string; strokeWidth?: number }>> = {
  Status: LayoutDashboard,
  Serve: Server,
  Evals: FlaskConical,
  Connect: Cable,
  Models: Boxes,
  Configure: Settings2,
};

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { modelLabel, setModelLabel } = useLabStore();
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [serveOk, setServeOk] = useState<boolean | null>(null);
  const [avail, setAvail] = useState<string | null>(null);

  useEffect(() => {
    const tick = () => {
      api
        .labStatus()
        .then((s) => {
          setHealthy(true);
          const ok = !!(s.serve && !s.serve.unreachable && s.serve.healthy);
          setServeOk(ok);
          const id = s.serve?.model_id || s.defaultModel;
          if (id && id !== "auto" && id !== "default") setModelLabel(id);
          else if (!ok) setModelLabel("");
          if (s.serve?.hardware?.available_gib != null) {
            setAvail(`${s.serve.hardware.available_gib} GiB free`);
          }
        })
        .catch(() => {
          setHealthy(false);
          setServeOk(false);
        });
    };
    tick();
    const t = setInterval(tick, 6000);
    return () => clearInterval(t);
  }, [setModelLabel]);

  return (
    <div className="flex h-full min-h-0 flex-col bg-lab-bg text-lab-text">
      <header className="sticky top-0 z-20 border-b border-lab-border-subtle bg-[rgba(10,10,11,0.78)] backdrop-blur-xl backdrop-saturate-150">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-5 px-4 md:px-6">
          <Link href="/status" className="group flex shrink-0 items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-[10px] bg-lab-text text-[13px] font-semibold tracking-tight text-lab-bg shadow-[0_1px_2px_rgba(0,0,0,0.4)] transition-transform group-hover:scale-[1.03]">
              L
            </div>
            <div className="leading-tight">
              <div className="text-[14px] font-semibold tracking-[-0.02em]">L.A.I.L</div>
              <div className="text-[10px] font-medium tracking-[0.06em] text-lab-muted uppercase">
                Local AI Lab
              </div>
            </div>
          </Link>

          <nav
            className="flex min-w-0 flex-1 items-center gap-0.5 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
            aria-label="Main"
          >
            {WORKSPACE_NAV.map(({ href, label }) => {
              const Icon = NAV_ICONS[label] || Activity;
              const active = pathname === href || pathname.startsWith(href + "/");
              return (
                <Link
                  key={href}
                  href={href}
                  className={cn(
                    "flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[13px] font-medium tracking-[-0.01em] transition-colors",
                    active
                      ? "bg-lab-active text-lab-text shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]"
                      : "text-lab-muted hover:bg-lab-hover hover:text-lab-text-dim",
                  )}
                >
                  <Icon className="h-3.5 w-3.5 shrink-0 opacity-75" strokeWidth={1.75} />
                  {label}
                </Link>
              );
            })}
          </nav>

          <div className="hidden shrink-0 items-center gap-2 sm:flex">
            <div className="flex items-center gap-1.5 rounded-full border border-lab-border-subtle bg-lab-panel/80 px-2 py-1">
              <StatusDot live={healthy} />
              <span className="text-[11px] text-lab-muted">
                {healthy ? "controller" : "offline"}
              </span>
            </div>
            <Badge tone={serveOk ? "ok" : "muted"} dot>
              {serveOk ? "vLLM" : "no serve"}
            </Badge>
            {modelLabel && (
              <span
                className="max-w-[148px] truncate font-mono text-[11px] text-lab-muted"
                title={modelLabel}
              >
                {modelLabel.split("/").pop()}
              </span>
            )}
            {avail && (
              <span className="hidden text-[11px] tabular-nums text-lab-muted lg:inline">
                {avail}
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-6xl px-4 py-7 md:px-6 md:py-9">{children}</div>
      </main>

      <footer className="border-t border-lab-border-subtle bg-lab-bg/90">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2 px-4 py-3 text-[11px] text-lab-muted md:px-6">
          <span className="tracking-[-0.01em]">Serve &amp; eval on Spark · agent work → Hermes</span>
          <Link
            href="/connect"
            className="font-medium text-lab-accent-bright transition-colors hover:text-lab-accent"
          >
            Wire Hermes →
          </Link>
        </div>
      </footer>
    </div>
  );
}
