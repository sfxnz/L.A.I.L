"use client";

import Link from "next/link";
import { memo, useEffect, useState } from "react";
import type { LabStatus } from "@/lib/api";
import type { EndpointSample } from "@/lib/lab-status-store";
import { fmtKvPct, fmtTokensK, forecastLine, kvForecast } from "@/lib/status/forecast";
import { engineLabel } from "@/lib/engines";
import { fmtAgo, fmtRate, fmtUptime } from "@/lib/status/format";
import { parseQuant } from "@/lib/status/quant";
import { Badge, CopyButton, Eyebrow, Nil, Panel, Sparkline, Stat, useCopy } from "@/components/ui";
import { cn } from "@/lib/utils";

/*
  The served model, once: what is loaded, how fast it is going right now, and how
  to point Hermes at it. Every live number on Status appears here and nowhere
  else on the page.

    decode       per-stream tok/s over busy time — the speed one Hermes request
                 feels. Idle: the last burst, dimmed and labelled with its age,
                 never shown as live.
    throughput   all streams together per wall-clock second, with its last 60 s
                 on a real time axis (idle reads 0, a gap is a missed sample).
    TTFT · prefill · requests · spec acceptance · KV — each with its own idle rule.
*/

const WINDOW_MS = 60_000;

type Serve = NonNullable<LabStatus["serve"]>;

function msOrS(s: number): string {
  return s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(2)} s`;
}

/** The OpenAI base URL Hermes should use: the port that actually answers, on loopback and on this page's host. */
export function hermesBases(baseUrl: string | null | undefined, pageHost: string): { local: string; remote: string | null } | null {
  const port = (baseUrl || "").match(/:(\d+)/)?.[1];
  if (!port) return null;
  const local = `http://127.0.0.1:${port}/v1`;
  const loopback = pageHost === "127.0.0.1" || pageHost === "localhost" || pageHost === "";
  return { local, remote: loopback ? null : `http://${pageHost}:${port}/v1` };
}

function Big({ children, dim, className }: { children: React.ReactNode; dim?: boolean; className?: string }) {
  return (
    <div
      className={cn(
        "lab-num flex items-baseline gap-1.5 font-[family-name:var(--font-display)] font-bold leading-none tabular-nums transition-colors duration-300",
        dim ? "text-lab-muted" : "text-lab-text",
        className,
      )}
    >
      {children}
    </div>
  );
}

const Unit = ({ children }: { children: React.ReactNode }) => (
  <span className="font-mono text-[10px] font-normal text-lab-muted">{children}</span>
);

export const EndpointHero = memo(function EndpointHero({
  serve,
  endpoint,
  serverNow,
  stale,
}: {
  serve: Serve;
  /** the endpoint's 60 s rate series from the store */
  endpoint: readonly EndpointSample[];
  serverNow: number | null;
  /** the snapshot stopped updating: show the numbers dimmed, never as live */
  stale?: boolean;
}) {
  const [pageHost, setPageHost] = useState("127.0.0.1");
  useEffect(() => setPageHost(window.location.hostname || "127.0.0.1"), []);
  const [flash, copy] = useCopy();

  const m = serve.metrics ?? {};
  const engine = serve.engine ?? {};
  const modelId = serve.model_id || null;
  const version = engine.version ?? serve.version?.version ?? null;
  // What is actually serving (owned_by / metrics prefix), not the configured default backend.
  const engineName = engineLabel(engine.name);
  const cluster = serve.cluster;
  const tp = cluster?.summary?.multi?.tensor_parallel_hint ?? cluster?.nodes?.find((n) => n.tensor_parallel_size != null)?.tensor_parallel_size ?? null;
  const quant = parseQuant(modelId);
  const running = engine.requests_running ?? m.requests_running ?? null;
  const waiting = engine.requests_waiting ?? m.requests_waiting ?? null;
  const kvPct = engine.kv_usage_pct ?? null;
  const forecast = kvForecast(engine.kv_capacity_tokens);

  const decode = m.decode_tok_per_s ?? null;
  const burst = m.last_burst;
  const burstAge = burst?.ended_at != null && serverNow != null ? (serverNow - burst.ended_at) / 1000 : null;
  const prefill = m.prefill_tok_per_s ?? null;
  const lastPrefill = prefill == null ? m.last_prefill : null;
  const lastPrefillAge = lastPrefill && serverNow != null ? (serverNow - lastPrefill.at) / 1000 : null;
  // No request started this second (mid-decode, or idle): the last TTFT, dimmed, with its age.
  const lastTtft = m.ttft_s == null ? m.last_ttft : null;
  const lastTtftAge = lastTtft && serverNow != null ? (serverNow - lastTtft.at) / 1000 : null;
  const busy = (running ?? 0) > 0;
  const domain = serverNow != null ? ([serverNow - WINDOW_MS, serverNow] as const) : undefined;
  const throughputPts = endpoint.map((s) => ({ t: s.t, v: s.throughput }));
  const bases = hermesBases(serve.base_url, pageHost);
  const env = bases && modelId ? `OPENAI_BASE_URL=${bases.local}\nOPENAI_API_KEY=local\nOPENAI_MODEL=${modelId}` : null;

  return (
    <Panel
      title="Served model"
      className="flex h-full flex-col"
      action={
        <span className="flex items-center gap-2">
          {flash && <Eyebrow className="text-lab-ok">{flash}</Eyebrow>}
          {engineName ? <Badge tone="ok">{engineName}</Badge> : <Nil word="Awaiting" />}
          {version && <span className="lab-num font-mono text-[10px] text-lab-muted">v{version}</span>}
        </span>
      }
    >
      <div className={cn("flex flex-col gap-4 p-4 transition-opacity duration-300", stale && "opacity-60")}>
        {/* identity */}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <span className="lab-num min-w-0 truncate font-mono text-[15px] font-medium tracking-[-0.01em] text-lab-text" title={modelId || undefined}>
            {modelId || <Nil word="None" />}
          </span>
          {modelId && <CopyButton text={modelId} label="Model id" />}
          <div className="flex flex-wrap items-center gap-2">
            {quant && <Badge tone="accent">{quant}</Badge>}
            {tp != null && <Badge tone="muted">TP={tp}</Badge>}
            {engine.max_model_len != null && (
              <span className="lab-num font-mono text-[10px] text-lab-muted" title="Context window the engine was started with">
                ctx {fmtTokensK(engine.max_model_len)}
              </span>
            )}
            {engine.uptime_s != null && (
              <span className="lab-num font-mono text-[10px] text-lab-muted" title="Engine process uptime">
                up {fmtUptime(engine.uptime_s)}
              </span>
            )}
          </div>
        </div>

        {/* live numbers */}
        <div className="grid gap-x-6 gap-y-4 border-t border-[color:var(--animus-hairline)] pt-4 sm:grid-cols-2 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1.4fr)_minmax(0,1fr)]">
          <Stat
            label="Decode · per stream"
            title="Tokens per second one request sees while it decodes (busy time, 1 s window, serve-engine /metrics)"
          >
            {decode != null ? (
              <Big className="text-[44px]">
                {fmtRate(decode)}
                <Unit>tok/s</Unit>
              </Big>
            ) : burst?.decode_tok_per_s != null ? (
              <div title="The endpoint is idle: this is the last burst's rate, not a live reading">
                <Big dim className="text-[44px]">
                  {fmtRate(burst.decode_tok_per_s)}
                  <Unit>tok/s</Unit>
                </Big>
                <Eyebrow className="mt-1.5 block text-[9px]">
                  last burst{burst.tokens ? ` · ${burst.tokens} tok` : ""}
                  {burstAge != null ? ` · ${fmtAgo(burstAge)}` : ""}
                </Eyebrow>
              </div>
            ) : (
              <Big dim className="text-[44px]">
                <Nil word="Idle" />
              </Big>
            )}
          </Stat>

          <div className="min-w-0">
            <Stat label="Throughput · all streams" title="Generated tokens per wall-clock second across every request (1 s window)">
              <Big className={cn("text-[26px]", !busy && (m.throughput_tok_per_s ?? 0) === 0 && "text-lab-muted")}>
                {m.throughput_tok_per_s != null ? fmtRate(m.throughput_tok_per_s) : <Nil word="Awaiting" />}
                <Unit>tok/s</Unit>
              </Big>
            </Stat>
            <Sparkline
              points={throughputPts}
              domain={domain}
              width={320}
              height={40}
              min={0}
              className="mt-2 w-full text-lab-line"
              label="Throughput over the last 60 s"
            />
            <div className="mt-0.5 flex justify-between font-mono text-[9px] text-lab-muted" aria-hidden>
              <span>−60 s</span>
              <span>now</span>
            </div>
          </div>

          <dl className="lab-num grid grid-cols-2 gap-x-4 gap-y-3 font-mono text-[12px] text-lab-text sm:col-span-2 lg:col-span-1 lg:grid-cols-1">
            <Row label="Requests" title="Running / queued on the engine">
              {running != null || waiting != null ? (
                <>
                  {running ?? 0} <span className="text-lab-muted">running</span> · {waiting ?? 0}{" "}
                  <span className={cn("text-lab-muted", (waiting ?? 0) > 0 && "text-lab-warn")}>queued</span>
                </>
              ) : (
                <Nil />
              )}
            </Row>
            <Row label="TTFT" title="Mean time to first token of the requests that started in the last second">
              {m.ttft_s != null ? (
                msOrS(m.ttft_s)
              ) : lastTtft ? (
                <span className="text-lab-muted" title="No request started this second: the last one's TTFT — not live">
                  last {msOrS(lastTtft.s)}
                  {lastTtftAge != null ? ` · ${fmtAgo(lastTtftAge)}` : ""}
                </span>
              ) : (
                <Nil word={busy ? "None" : "Idle"} />
              )}
            </Row>
            <Row label="Prefill" title="Prompt tok/s of requests that finished in the last second (cache hits excluded)">
              {prefill != null ? (
                `${fmtRate(prefill)} tok/s`
              ) : lastPrefill ? (
                <span className="text-lab-muted" title="Last finished request's prefill — not live">
                  last {fmtRate(lastPrefill.tok_per_s)}
                  {lastPrefillAge != null ? ` · ${fmtAgo(lastPrefillAge)}` : ""}
                </span>
              ) : (
                <Nil word={busy ? "None" : "Idle"} />
              )}
            </Row>
            <Row label="Spec accept" title="Speculative decoding: accepted draft tokens / proposed, and tokens per step">
              {m.spec_accept_rate != null ? (
                <>
                  {Math.round(m.spec_accept_rate * 100)}%
                  {m.spec_tokens_per_step != null && <span className="text-lab-muted"> · {m.spec_tokens_per_step.toFixed(2)}/step</span>}
                </>
              ) : m.spec_accept_rate_lifetime != null ? (
                <span className="text-lab-muted" title="No drafts this second: acceptance since the engine started">
                  {Math.round(m.spec_accept_rate_lifetime * 100)}% lifetime
                </span>
              ) : (
                <Nil word="None" />
              )}
            </Row>
            <Row
              label="KV cache"
              title={`Share of the KV block pool in use (blocks are reserved whole, so this is not a token count)${forecast ? ` · ${forecastLine(forecast)}` : ""}`}
            >
              {kvPct != null ? (
                <span className={cn(kvPct >= 90 && "text-lab-warn")}>{fmtKvPct(kvPct)}</span>
              ) : (
                <Nil />
              )}
            </Row>
          </dl>
        </div>

        {/* wiring */}
        <div className="flex flex-col gap-2 border-t border-[color:var(--animus-hairline)] pt-3">
          <div className="flex flex-wrap items-center gap-2">
            <Eyebrow>Hermes / OpenAI client</Eyebrow>
            {bases ? (
              <>
                <code className="lab-num min-w-0 truncate font-mono text-[12px] text-lab-text" title="OPENAI_BASE_URL on this Spark">
                  {bases.local}
                </code>
                <CopyButton text={bases.local} label="Base URL" />
                {env && (
                  <button
                    type="button"
                    onClick={() => copy("Env block", env)}
                    className="animus-chamfer-sm inline-flex h-6 items-center border border-lab-border px-1.5 font-[family-name:var(--font-display)] text-[9px] font-semibold uppercase leading-none tracking-[0.14em] text-lab-muted transition-colors hover:border-lab-line hover:text-lab-text"
                    title={env}
                  >
                    copy env
                  </button>
                )}
              </>
            ) : (
              <Nil word="None" />
            )}
            <Link
              href="/connect"
              className="ml-auto font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.14em] text-lab-accent-bright transition-colors hover:text-lab-accent"
              title="Tailscale URL, curl probes"
            >
              More ways to connect →
            </Link>
          </div>
          {bases?.remote && (
            <div className="flex flex-wrap items-center gap-2">
              <Eyebrow>From this browser&apos;s host</Eyebrow>
              <code className="lab-num min-w-0 truncate font-mono text-[11px] text-lab-text-dim">{bases.remote}</code>
              <CopyButton text={bases.remote} label="Remote base URL" />
            </div>
          )}
          <Flags flags={engine.flags ?? null} fingerprint={engine.flags_fingerprint ?? null} />
        </div>
      </div>
    </Panel>
  );
});

function Row({ label, title, children }: { label: string; title?: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0" title={title}>
      <dt className="animus-eyebrow text-[9px]">{label}</dt>
      <dd className="mt-0.5 truncate">{children}</dd>
    </div>
  );
}

function Flags({ flags, fingerprint }: { flags: string[] | null; fingerprint: string | null }) {
  const [open, setOpen] = useState(false);
  if (!fingerprint && !flags?.length) return null;
  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        <Eyebrow>Serve flags</Eyebrow>
        {fingerprint && (
          <span className="lab-num font-mono text-[11px] text-lab-text-dim" title="Short hash of the engine's launch flags">
            {fingerprint}
          </span>
        )}
        {flags?.length ? (
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.14em] text-lab-accent-bright transition-colors hover:text-lab-accent"
          >
            {open ? "hide" : `show (${flags.length})`}
          </button>
        ) : null}
      </div>
      {open && flags?.length ? (
        <pre className="mt-2 max-h-40 overflow-auto rounded-[2px] border border-lab-border bg-lab-editor p-2.5 font-mono text-[10px] leading-relaxed text-lab-text-dim whitespace-pre-wrap">
          {flags.join("\n")}
        </pre>
      ) : null}
    </div>
  );
}
