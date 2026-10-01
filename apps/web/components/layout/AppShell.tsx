"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  FlaskConical,
  Gauge,
  LayoutDashboard,
  Server,
  Settings2,
  Waves,
} from "lucide-react";
import { useEffect, useState } from "react";
import { setClientToken } from "@/lib/auth-token";
import { fmtKvPct } from "@/lib/status/forecast";
import { serveHealthy, startLabStatusPolling, useLabStatusStore } from "@/lib/lab-status-store";
import { WORKSPACE_NAV, isProseRoute } from "@/lib/ide-chrome";
import { cn } from "@/lib/utils";
import { Eyebrow, SyncRing, Tick, type SyncState } from "@/components/ui";
import { AnimusField } from "@/components/animus/AnimusField";
import { ThemeToggle } from "@/components/animus/ThemeToggle";
import { CommandPalette } from "@/components/command/CommandPalette";
import { useGlobalShortcuts } from "@/lib/shortcuts";

const NAV_ICONS: Record<string, React.ComponentType<{ className?: string; strokeWidth?: number }>> = {
  Status: LayoutDashboard,
  Serve: Server,
  Bench: Gauge,
  Streams: Waves,
  Evals: FlaskConical,
  Configure: Settings2,
};

const TOKEN_COPY = "LAIL_TOKEN required — paste it to synchronize.";

function fmtRate(v: number) {
  return v >= 100 ? String(Math.round(v)) : v.toFixed(1);
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { status, loading, needToken, unreachable, liveRun } = useLabStatusStore();
  const [tokenDraft, setTokenDraft] = useState("");

  // The ONE lab-status poll: 2s while visible, paused while hidden.
  useEffect(() => startLabStatusPolling(), []);
  useGlobalShortcuts();

  const serve = status?.serve;
  const healthy = serveHealthy(status);
  const servedId = serve?.model_id;
  const model =
    healthy && servedId && servedId !== "auto" && servedId !== "default" ? servedId : null;
  // Header truth: while a run streams, the header shows the run's own aggregate
  // (the number Streams/Bench show), labelled "run"; otherwise the endpoint's
  // per-stream decode rate (busy time, serve.metrics), labelled "endpoint". When
  // the endpoint is idle, the previous burst is shown dimmed and labelled "last" —
  // never as live. Never two unexplained numbers.
  const metrics = healthy ? serve?.metrics : null;
  const lastBurst = metrics?.last_burst?.decode_tok_per_s ?? null;
  const liveTokS = liveRun ? liveRun.tok_s : (metrics?.decode_tok_per_s ?? null);
  const tokS = liveTokS ?? lastBurst;
  const rateSource = liveRun ? "run" : liveTokS != null ? "endpoint" : "last";
  const engine = serve?.engine;
  const running = liveRun?.running ?? engine?.requests_running ?? serve?.metrics?.requests_running ?? null;
  const waiting = liveRun?.waiting ?? engine?.requests_waiting ?? serve?.metrics?.requests_waiting ?? null;
  const kvPct = engine?.kv_usage_pct ?? null;
  const freeGib = serve?.hardware?.available_gib;

  const ring: SyncState | null = needToken
    ? "token"
    : unreachable
      ? "offline"
      : loading
        ? null
        : healthy
          ? "serving"
          : "idle";
  const word = needToken
    ? "token"
    : unreachable
      ? "unreachable"
      : loading
        ? "…"
        : healthy
          ? "serving"
          : "idle";
  const probeNote = needToken
    ? TOKEN_COPY
    : unreachable
      ? "Controller unreachable on :8787 — bun run dev."
      : loading
        ? "Checking lab…"
        : healthy
          ? `Controller up · serving ${model ? model.split("/").pop() : "a model"}`
          : "Controller up · no model loaded";

  // Instrument pages get the 1440px bento; prose pages keep the 1152px measure.
  const measure = isProseRoute(pathname || "/") ? "max-w-6xl" : "max-w-[1440px]";

  return (
    <div className="relative isolate flex h-full min-h-0 flex-col bg-lab-bg text-lab-text">
      {/* Reconstruction field — z-0, behind every layer of chrome. */}
      <AnimusField />

      <a href="#main" className="lab-skip-link">
        Skip to content
      </a>

      <header className="sticky top-0 z-20 shrink-0 border-b border-[color:var(--animus-hairline)] bg-[color:var(--animus-glass)] backdrop-blur-xl backdrop-saturate-150">
        <div
          className={cn(
            "animus-bracketed relative mx-auto flex h-14 items-center gap-3 px-4 md:gap-5 md:px-6",
            measure,
          )}
        >
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
            // gives way. The instrument strip on the right is `min-w-0 flex-1`
            // and truncates instead; see below.
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
            // Instrument strip: SyncRing · model · live tok/s · strands · free GiB.
            // This is the side that yields — `min-w-0 flex-1` soaks up the
            // remaining space and its cells hide progressively at md/lg/xl, so
            // the nav above always renders in full.
            className="lab-num flex min-w-0 flex-1 items-center justify-end gap-2 overflow-hidden sm:gap-2.5"
            aria-live="polite"
            aria-atomic="true"
          >
            <span className="sr-only">{probeNote}</span>

            <span className="flex items-center gap-1.5" title={probeNote}>
              <SyncRing
                state={ring}
                label={
                  needToken
                    ? "Token required"
                    : unreachable
                      ? "Controller unreachable"
                      : loading
                        ? "Controller status unknown"
                        : healthy
                          ? "Serving"
                          : "Idle"
                }
              />
              <Eyebrow className="hidden tracking-[0.18em] lg:inline">{word}</Eyebrow>
            </span>

            {model && (
              <>
                <Tick className="hidden xl:block" />
                <span
                  className="hidden max-w-[160px] truncate font-mono text-[10px] text-lab-text-dim xl:inline"
                  title={model}
                >
                  {model.split("/").pop()}
                </span>
              </>
            )}

            {tokS != null && (
              <>
                <Tick className="hidden md:block" />
                <span
                  className="hidden shrink-0 items-baseline gap-1.5 font-mono text-[10px] text-lab-text md:inline-flex"
                  title={
                    liveRun
                      ? `Live ${liveRun.source} run aggregate (usage-calibrated) · peak ${fmtRate(liveRun.peak)}`
                      : liveTokS != null
                        ? `Per-stream decode rate over busy time (serve-engine, 1 s)${
                            metrics?.throughput_tok_per_s != null ? ` · all streams ${fmtRate(metrics.throughput_tok_per_s)} tok/s` : ""
                          }${metrics?.spec_accept_rate != null ? ` · spec accept ${Math.round(metrics.spec_accept_rate * 100)}%` : ""}`
                        : "Endpoint idle — decode rate of the last burst, not live"
                  }
                >
                  <Eyebrow className={cn("text-[8px]", liveRun ? "text-lab-target" : undefined)}>{rateSource}</Eyebrow>
                  <span className={rateSource === "last" ? "text-lab-muted" : undefined}>
                    {fmtRate(tokS)} <span className="text-lab-muted">tok/s</span>
                  </span>
                </span>
              </>
            )}

            {!loading && !needToken && !unreachable && (running != null || waiting != null) && (
              <>
                <Tick className="hidden lg:block" />
                <Eyebrow
                  className="hidden shrink-0 lg:inline"
                  title={liveRun ? "Strands in the live run" : "Requests running / waiting on the engine"}
                >
                  {running ?? 0} running / {waiting ?? 0} waiting
                </Eyebrow>
              </>
            )}

            {kvPct != null && (
              <>
                <Tick className="hidden xl:block" />
                <Eyebrow className="hidden shrink-0 xl:inline" title="KV cache in use">
                  KV {fmtKvPct(kvPct)}
                </Eyebrow>
              </>
            )}

            {freeGib != null && (
              <>
                <Tick className="hidden lg:block" />
                <Eyebrow className="hidden shrink-0 tracking-[0.12em] lg:inline" title="MemAvailable on this host (/proc/meminfo)">
                  {freeGib.toFixed(1)} GiB free
                </Eyebrow>
              </>
            )}
          </div>

          <Tick className="hidden sm:block" />

          <CommandPalette />
          <ThemeToggle />
        </div>
      </header>

      {needToken && (
        <form
          className="relative z-20 flex shrink-0 flex-wrap items-center gap-2 border-b border-[color:var(--animus-hairline)] bg-[color:var(--animus-glass)] px-4 py-2 md:px-6"
          onSubmit={(e) => {
            e.preventDefault();
            setClientToken(tokenDraft);
            window.location.reload();
          }}
        >
          <SyncRing state="token" />
          <span className="text-[12px] text-lab-text-dim">{TOKEN_COPY}</span>
          <input
            type="password"
            autoComplete="off"
            value={tokenDraft}
            onChange={(e) => setTokenDraft(e.target.value)}
            placeholder="LAIL_TOKEN"
            aria-label="LAIL_TOKEN"
            className="min-w-[10rem] flex-1 border border-[color:var(--animus-hairline)] bg-transparent px-2 py-1 font-mono text-[12px]"
          />
          <button
            type="submit"
            className="shrink-0 bg-lab-accent px-2 py-1 font-[family-name:var(--font-display)] text-[11px] font-semibold uppercase tracking-[0.14em] text-white"
          >
            Synchronize
          </button>
        </form>
      )}

      <main id="main" className="relative z-10 min-h-0 flex-1 overflow-y-auto" tabIndex={-1}>
        <div className={cn("mx-auto px-4 py-5 md:px-6 md:py-6", measure)}>{children}</div>
      </main>

      <footer className="relative z-10 shrink-0 border-t border-[color:var(--animus-hairline)] bg-[color:var(--animus-glass)] backdrop-blur-md">
        <div
          className={cn(
            "mx-auto flex flex-wrap items-center justify-between gap-x-4 gap-y-1 px-4 py-2 md:px-6",
            measure,
          )}
        >
          <span className="font-[family-name:var(--font-display)] text-[10px] font-medium uppercase leading-none tracking-[0.18em] text-lab-muted">
            Serve · bench · streams · Hermes
          </span>
        </div>
      </footer>
    </div>
  );
}
