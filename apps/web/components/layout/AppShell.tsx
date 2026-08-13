"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  Boxes,
  FlaskConical,
  LayoutDashboard,
  Plug,
  Server,
  Settings2,
} from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { setClientToken } from "@/lib/auth-token";
import { useLabStore } from "@/lib/store";
import { WORKSPACE_NAV } from "@/lib/ide-chrome";
import { cn } from "@/lib/utils";
import { Badge, StatusDot } from "@/components/ui";
import { AnimusField } from "@/components/animus/AnimusField";
import { ThemeToggle } from "@/components/animus/ThemeToggle";

const NAV_ICONS: Record<string, React.ComponentType<{ className?: string; strokeWidth?: number }>> = {
  Status: LayoutDashboard,
  Serve: Server,
  Models: Boxes,
  Evals: FlaskConical,
  Connect: Plug,
  Configure: Settings2,
};

const PAGE_TITLES: Array<{ match: (p: string) => boolean; title: string }> = [
  { match: (p) => p.startsWith("/evals/tool"), title: "Tool Eval" },
  { match: (p) => p.startsWith("/evals"), title: "Evals" },
  { match: (p) => p.startsWith("/server"), title: "Serve" },
  { match: (p) => p.startsWith("/configure"), title: "Configure" },
  { match: (p) => p.startsWith("/status") || p === "/", title: "Status" },
  { match: (p) => p.startsWith("/connect"), title: "Connect" },
  { match: (p) => p.startsWith("/models"), title: "Models" },
  { match: (p) => p.startsWith("/lab"), title: "Lab" },
  { match: (p) => p.startsWith("/workbench"), title: "Workbench" },
];

/** Hairline divider between readout cells. */
function Tick({ className }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={cn(
        "h-3 w-px shrink-0 bg-[color:var(--animus-hairline)]",
        className,
      )}
    />
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { modelLabel, setModelLabel } = useLabStore();
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [serveOk, setServeOk] = useState<boolean | null>(null);
  const [avail, setAvail] = useState<string | null>(null);
  const [probeNote, setProbeNote] = useState("Checking lab…");
  const [needToken, setNeedToken] = useState(false);
  const [tokenDraft, setTokenDraft] = useState("");

  useEffect(() => {
    const hit = PAGE_TITLES.find((t) => t.match(pathname || "/"));
    document.title = hit ? `${hit.title} · L.A.I.L` : "L.A.I.L — Serve & Evals";
  }, [pathname]);

  useEffect(() => {
    const tick = () => {
      api
        .labStatus()
        .then((s) => {
          setNeedToken(false);
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
        .catch((e: unknown) => {
          const msg = e instanceof Error ? e.message : String(e);
          const unauthorized = /401|unauthorized|LAIL_TOKEN/i.test(msg);
          setHealthy(false);
          setServeOk(false);
          setAvail(null);
          setNeedToken(unauthorized);
          setProbeNote(
            unauthorized
              ? "LAIL_TOKEN required — enter it below"
              : "Controller offline — start bun run dev",
          );
        });
    };
    tick();
    const t = setInterval(tick, 6000);
    return () => clearInterval(t);
  }, [setModelLabel]);

  return (
    <div className="relative isolate flex h-full min-h-0 flex-col bg-lab-bg text-lab-text">
      {/* Reconstruction field — z-0, behind every layer of chrome. */}
      <AnimusField />

      <a href="#main" className="lab-skip-link">
        Skip to content
      </a>

      <header className="sticky top-0 z-20 shrink-0 border-b border-[color:var(--animus-hairline)] bg-[color:var(--animus-glass)] backdrop-blur-xl backdrop-saturate-150">
        <div className="animus-bracketed relative mx-auto flex h-14 max-w-6xl items-center gap-3 px-4 md:gap-5 md:px-6">
          <Link
            href="/status"
            className="group flex shrink-0 items-center gap-2.5 focus-visible:outline-offset-4"
          >
            <span className="animus-chamfer-sm flex h-7 w-7 items-center justify-center bg-lab-accent font-[family-name:var(--font-display)] text-[14px] font-semibold leading-none text-white transition-transform duration-200 group-hover:scale-[1.04]">
              L
            </span>
            <span className="leading-none">
              <span className="block font-[family-name:var(--font-display)] text-[15px] font-semibold uppercase leading-none tracking-[0.22em] text-lab-text md:tracking-[0.3em]">
                L.A.I.L
              </span>
              <span className="mt-1 hidden font-[family-name:var(--font-display)] text-[9px] font-medium uppercase leading-none tracking-[0.26em] text-lab-muted md:block">
                Local AI Lab
              </span>
            </span>
          </Link>

          <Tick className="hidden h-6 md:block" />

          <nav
            // Navigation is primary chrome: it must NEVER be the thing that
            // gives way. This was `min-w-0 flex-1 overflow-x-auto` against a
            // `shrink-0` status cluster, so once a model started serving the
            // cluster grew (controller + vLLM badge + model + memory) and the
            // nav absorbed the entire squeeze — 343px of links crushed into
            // 176px, silently scroll-clipping EVALS and CONFIGURE off-screen
            // with no scrollbar to hint they existed. Secondary telemetry
            // truncates instead; see the status cluster below.
            className="flex shrink-0 items-center gap-1"
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
                    "relative flex shrink-0 items-center gap-1.5 px-2.5 py-1.5 font-[family-name:var(--font-display)] text-[12px] font-semibold uppercase leading-none tracking-[0.16em] transition-colors duration-200 focus-visible:z-10 md:px-3",
                    active
                      ? "text-lab-text"
                      : "text-lab-muted hover:bg-[color:var(--animus-accent-wash)] hover:text-lab-text-dim",
                  )}
                >
                  {active && (
                    <>
                      <span
                        aria-hidden
                        className="animus-notch absolute inset-0 bg-[image:var(--animus-selection-fade)] opacity-50"
                      />
                      <span
                        aria-hidden
                        className="absolute inset-y-0 left-0 w-[2px] bg-lab-accent"
                      />
                    </>
                  )}
                  <Icon className="relative h-3.5 w-3.5 shrink-0" strokeWidth={1.75} aria-hidden />
                  <span className="relative hidden lg:inline">{label}</span>
                </Link>
              );
            })}
          </nav>

          <div
            // Secondary telemetry: this is the side that yields. `min-w-0`
            // + `flex-1` lets it soak up the remaining space and truncate
            // (its children already hide progressively at sm/lg/xl), so the
            // nav above always renders in full.
            className="flex min-w-0 flex-1 items-center justify-end gap-2 overflow-hidden sm:gap-2.5"
            aria-live="polite"
            aria-atomic="true"
          >
            <span className="sr-only">{probeNote}</span>

            <span className="flex items-center gap-1.5" title={probeNote}>
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
              <span className="hidden font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.18em] text-lab-muted lg:inline">
                {healthy === null ? "…" : healthy ? "controller" : "offline"}
              </span>
            </span>

            <Tick className="hidden sm:block" />

            <span className="hidden sm:inline-flex">
              {serveOk === null ? (
                <Badge tone="muted">checking…</Badge>
              ) : (
                <span
                  className={cn(
                    "inline-flex transition-shadow duration-300",
                    serveOk &&
                      "shadow-[0_0_14px_color-mix(in_srgb,var(--color-lab-ok)_32%,transparent)]",
                  )}
                >
                  <Badge tone={serveOk ? "ok" : "muted"} dot={serveOk}>
                    {serveOk ? "vLLM serving" : "idle"}
                  </Badge>
                </span>
              )}
            </span>

            {modelLabel && (
              <>
                <Tick className="hidden xl:block" />
                <span
                  className="hidden max-w-[128px] truncate font-mono text-[10px] tabular-nums text-lab-text-dim xl:inline"
                  title={modelLabel}
                >
                  {modelLabel.split("/").pop()}
                </span>
              </>
            )}

            {avail && (
              <>
                <Tick className="hidden lg:block" />
                <span className="hidden font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.12em] tabular-nums text-lab-muted lg:inline">
                  {avail}
                </span>
              </>
            )}
          </div>

          <Tick className="hidden sm:block" />

          <ThemeToggle />
        </div>
      </header>

      {needToken && (
        <form
          className="relative z-20 flex shrink-0 items-center gap-2 border-b border-[color:var(--animus-hairline)] bg-[color:var(--animus-glass)] px-4 py-2 md:px-6"
          onSubmit={(e) => {
            e.preventDefault();
            setClientToken(tokenDraft);
            setNeedToken(false);
            window.location.reload();
          }}
        >
          <label className="font-[family-name:var(--font-display)] text-[11px] font-semibold uppercase tracking-[0.14em] text-lab-muted">
            Token
          </label>
          <input
            type="password"
            autoComplete="off"
            value={tokenDraft}
            onChange={(e) => setTokenDraft(e.target.value)}
            placeholder="LAIL_TOKEN"
            className="min-w-0 flex-1 border border-[color:var(--animus-hairline)] bg-transparent px-2 py-1 font-mono text-[12px]"
          />
          <button
            type="submit"
            className="shrink-0 bg-lab-accent px-2 py-1 font-[family-name:var(--font-display)] text-[11px] font-semibold uppercase tracking-[0.14em] text-white"
          >
            Store
          </button>
        </form>
      )}

      <main id="main" className="relative z-10 min-h-0 flex-1 overflow-y-auto" tabIndex={-1}>
        <div className="mx-auto max-w-6xl px-4 py-5 md:px-6 md:py-6">{children}</div>
      </main>

      <footer className="relative z-10 shrink-0 border-t border-[color:var(--animus-hairline)] bg-[color:var(--animus-glass)] backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-4 gap-y-1 px-4 py-2 md:px-6">
          <span className="font-[family-name:var(--font-display)] text-[10px] font-medium uppercase leading-none tracking-[0.18em] text-lab-muted">
            Serve · eval · Hermes
          </span>
        </div>
      </footer>
    </div>
  );
}
