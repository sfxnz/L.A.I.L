"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
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
  Configure: Settings2,
};

const PAGE_TITLES: Array<{ match: (p: string) => boolean; title: string }> = [
  { match: (p) => p.startsWith("/evals/tool"), title: "Tool Eval" },
  { match: (p) => p.startsWith("/evals"), title: "Evals" },
  { match: (p) => p.startsWith("/server"), title: "Serve" },
  { match: (p) => p.startsWith("/configure"), title: "Configure" },
  { match: (p) => p.startsWith("/status") || p === "/", title: "Status" },
  { match: (p) => p.startsWith("/connect"), title: "Connect" },
  { match: (p) => p.startsWith("/lab"), title: "Lab" },
  { match: (p) => p.startsWith("/workbench"), title: "Workbench" },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { modelLabel, setModelLabel } = useLabStore();
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [serveOk, setServeOk] = useState<boolean | null>(null);
  const [avail, setAvail] = useState<string | null>(null);
  const [probeNote, setProbeNote] = useState("Checking lab…");

  useEffect(() => {
    const hit = PAGE_TITLES.find((t) => t.match(pathname || "/"));
    document.title = hit ? `${hit.title} · L.A.I.L` : "L.A.I.L — Serve & Evals";
  }, [pathname]);

  useEffect(() => {
    const tick = () => {
      api
        .labStatus()
        .then((s) => {
          setHealthy(true);
          const ok = !!(s.serve && !s.serve.unreachable && s.serve.healthy);
          setServeOk(ok);
          const servedId = s.serve?.model_id;
          if (ok && servedId && servedId !== "auto" && servedId !== "default") {
            setModelLabel(servedId);
          } else {
            setModelLabel("");
          }
          if (s.serve?.hardware?.available_gib != null) {
            setAvail(`${s.serve.hardware.available_gib} GiB free`);
          } else {
            setAvail(null);
          }
          setProbeNote(
            ok
              ? `Controller up · vLLM serving${servedId ? ` ${String(servedId).split("/").pop()}` : ""}`
              : "Controller up · no vLLM serve",
          );
        })
        .catch(() => {
          setHealthy(false);
          setServeOk(false);
          setAvail(null);
          setProbeNote("Controller offline — start bun run dev");
        });
    };
    tick();
    const t = setInterval(tick, 6000);
    return () => clearInterval(t);
  }, [setModelLabel]);

  return (
    <div className="flex h-full min-h-0 flex-col bg-lab-bg text-lab-text">
      <a href="#main" className="lab-skip-link">
        Skip to content
      </a>

      <header className="sticky top-0 z-20 border-b border-lab-border-subtle bg-[rgba(10,10,11,0.78)] backdrop-blur-xl backdrop-saturate-150">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-5 px-4 md:px-6">
          <Link
            href="/status"
            className="group flex shrink-0 items-center gap-2.5 rounded-lg focus-visible:outline-offset-4"
          >
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
                  aria-current={active ? "page" : undefined}
                  className={cn(
                    "flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[13px] font-medium tracking-[-0.01em] transition-colors",
                    active
                      ? "bg-lab-active text-lab-text shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]"
                      : "text-lab-muted hover:bg-lab-hover hover:text-lab-text-dim",
                  )}
                >
                  <Icon className="h-3.5 w-3.5 shrink-0 opacity-75" strokeWidth={1.75} aria-hidden />
                  {label}
                </Link>
              );
            })}
          </nav>

          <div
            className="hidden shrink-0 items-center gap-2 sm:flex"
            aria-live="polite"
            aria-atomic="true"
          >
            <div
              className="flex items-center gap-1.5 rounded-full border border-lab-border-subtle bg-lab-panel/80 px-2 py-1"
              title={probeNote}
            >
              <StatusDot
                live={healthy}
                label={
                  healthy === null
                    ? "Controller status unknown"
                    : healthy
                      ? "Controller online"
                      : "Controller offline"
                }
              />
              <span className="text-[11px] text-lab-muted">
                {healthy === null ? "…" : healthy ? "controller" : "offline"}
              </span>
            </div>
            {serveOk === null ? (
              <Badge tone="muted">checking…</Badge>
            ) : (
              <Badge tone={serveOk ? "ok" : "muted"} dot={serveOk}>
                {serveOk ? "vLLM serving" : "idle"}
              </Badge>
            )}
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

      <main id="main" className="min-h-0 flex-1 overflow-y-auto" tabIndex={-1}>
        <div className="mx-auto max-w-6xl px-4 py-5 md:px-6 md:py-6">{children}</div>
      </main>

      <footer className="border-t border-lab-border-subtle bg-lab-bg/90">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2 px-4 py-3 text-[11px] text-lab-muted md:px-6">
          <span className="tracking-[-0.01em]">
            Serve · eval · Hermes builds · public on GitHub Pages
          </span>
          <a
            href="https://sfxnz.github.io/dgx-lab/"
            target="_blank"
            rel="noreferrer"
            className="font-medium text-lab-accent-bright transition-colors hover:text-lab-accent"
          >
            Public site →
          </a>
        </div>
      </footer>
    </div>
  );
}
