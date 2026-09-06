"use client";

import Link from "next/link";
import { useState } from "react";
import type { LabStatus } from "@/lib/api";
import { fmtTokensK, kvFraction } from "@/lib/status/forecast";
import { fmtUptime } from "@/lib/status/format";
import { parseQuant } from "@/lib/status/quant";
import { Badge, CopyButton, Eyebrow, Nil, Panel, Stat, SyncRing, Tick } from "@/components/ui";
import { cn } from "@/lib/utils";

/**
 * Now serving — the one card that says what is loaded and how to reach it:
 * model (full id, copy), engine + version, TP, quant, max-model-len, uptime,
 * KV % now, running/waiting, flag fingerprint (view flags), endpoint (copy),
 * Hermes wiring hint. Every engine field is C2's additive contract and may be
 * absent — <Nil/> then, never a guess.
 */
export function NowServing({
  status,
  healthy,
  loading,
  className,
}: {
  status: LabStatus | null;
  healthy: boolean;
  loading: boolean;
  className?: string;
}) {
  const [flagsOpen, setFlagsOpen] = useState(false);
  const serve = status?.serve;
  const engine = serve?.engine;
  const metrics = serve?.metrics;
  const cluster = status?.cluster || serve?.cluster;
  const modelId = healthy ? serve?.model_id || null : null;
  const version = engine?.version ?? serve?.version?.version ?? null;
  const backendKey = status?.defaultBackend || "vllm";
  const engineLabel = backendKey.toLowerCase() === "vllm" ? "vLLM" : backendKey.toLowerCase() === "llamacpp" ? "llama.cpp" : backendKey;
  const tp = cluster?.summary?.multi?.tensor_parallel_hint ?? cluster?.nodes?.find((n) => n.tensor_parallel_size != null)?.tensor_parallel_size ?? null;
  const quant = parseQuant(modelId);
  const kvUsage = engine?.kv_usage_pct ?? metrics?.gpu_kv_cache_usage ?? null;
  const kvPct = kvUsage == null ? null : Math.round(kvFraction(kvUsage) * 100);
  const running = engine?.requests_running ?? metrics?.requests_running ?? null;
  const waiting = engine?.requests_waiting ?? metrics?.requests_waiting ?? null;
  const endpoint = status?.openAiBase || (serve?.base_url ? `${serve.base_url}/v1` : null);
  const flags = engine?.flags ?? null;
  const fingerprint = engine?.flags_fingerprint ?? null;

  const nil = (word: "Awaiting" | "None" = healthy ? "Awaiting" : "None") => <Nil word={word} />;
  const mono = "lab-num font-mono text-[13px] text-lab-text";

  return (
    <Panel
      title="Now serving"
      className={cn("flex h-full flex-col", className)}
      action={
        <span className="flex items-center gap-2">
          <SyncRing state={loading ? null : healthy ? "serving" : "idle"} label={healthy ? "Endpoint live" : "No model serving"} />
          <Eyebrow className={healthy ? "text-lab-ok" : undefined}>{loading ? "probing" : healthy ? "live" : "idle"}</Eyebrow>
        </span>
      }
    >
      <div className="flex flex-1 flex-col gap-4 p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <Eyebrow>Model</Eyebrow>
            <div className="mt-1 flex min-w-0 items-center gap-2">
              <span
                className="lab-num min-w-0 truncate font-mono text-[15px] font-medium tracking-[-0.01em] text-lab-text"
                title={modelId || undefined}
              >
                {modelId || nil("None")}
              </span>
              {modelId && <CopyButton text={modelId} label="Model id" />}
            </div>
            <div className="mt-1.5 flex flex-wrap items-center gap-2">
              <Badge tone={healthy ? "ok" : "muted"}>{engineLabel}</Badge>
              {quant && <Badge tone="accent">{quant}</Badge>}
              {tp != null && <Badge tone="muted">TP={tp}</Badge>}
              <span className="lab-num font-mono text-[10px] text-lab-muted" title="Engine version">
                {version ? `v${version}` : nil()}
              </span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-x-4 gap-y-3 border-t border-[color:var(--animus-hairline)] pt-3 sm:grid-cols-4">
          <Stat label="max-model-len" title="Context window the engine was started with">
            <span className={mono}>{engine?.max_model_len != null ? fmtTokensK(engine.max_model_len) : nil()}</span>
          </Stat>
          <Stat label="Uptime" title="Engine process uptime">
            <span className={mono}>{engine?.uptime_s != null ? fmtUptime(engine.uptime_s) : nil()}</span>
          </Stat>
          <Stat label="KV now" title="KV cache in use">
            <span className={cn(mono, kvPct != null && kvPct >= 90 && "text-lab-warn")}>
              {kvPct != null ? `${kvPct}%` : nil()}
            </span>
          </Stat>
          <Stat label="Requests" title="Running / waiting on the engine">
            <span className={mono}>
              {running != null || waiting != null ? (
                <>
                  {running ?? 0} <span className="text-lab-muted">running</span> · {waiting ?? 0} <span className="text-lab-muted">waiting</span>
                </>
              ) : (
                nil()
              )}
            </span>
          </Stat>
        </div>

        <div className="border-t border-[color:var(--animus-hairline)] pt-3">
          <div className="flex flex-wrap items-center gap-2">
            <Eyebrow>Flags</Eyebrow>
            <span className="lab-num font-mono text-[11px] text-lab-text-dim" title="Short hash of the placement-engine output">
              {fingerprint || nil()}
            </span>
            {fingerprint && <CopyButton text={fingerprint} label="Fingerprint" />}
            {flags?.length ? (
              <button
                type="button"
                onClick={() => setFlagsOpen((v) => !v)}
                aria-expanded={flagsOpen}
                className="ml-auto font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.14em] text-lab-accent-bright transition-colors hover:text-lab-accent"
              >
                {flagsOpen ? "hide flags" : `view flags (${flags.length})`}
              </button>
            ) : null}
          </div>
          {flagsOpen && flags?.length ? (
            <pre className="mt-2 max-h-40 overflow-auto rounded-[2px] border border-lab-border bg-lab-editor p-2.5 font-mono text-[10px] leading-relaxed text-lab-text-dim whitespace-pre-wrap">
              {flags.join("\n")}
            </pre>
          ) : null}
        </div>

        <div className="mt-auto flex flex-wrap items-center gap-2 border-t border-[color:var(--animus-hairline)] pt-3">
          <Eyebrow>Endpoint</Eyebrow>
          <span className="lab-num min-w-0 truncate font-mono text-[11px] text-lab-text" title={endpoint || undefined}>
            {endpoint || nil("None")}
          </span>
          {endpoint && <CopyButton text={endpoint} label="Endpoint" />}
          <Tick className="hidden sm:block" />
          <Link
            href="/connect"
            className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.14em] text-lab-accent-bright transition-colors hover:text-lab-accent"
            title="Wire Hermes or any OpenAI client to this endpoint"
          >
            Wire Hermes →
          </Link>
        </div>
      </div>
    </Panel>
  );
}
