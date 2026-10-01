"use client";

import { memo, type CSSProperties } from "react";
import type { ClusterNode, ClusterStatus } from "@/lib/api";
import { staleAfterS, useLabStatusStore, type NodeSample } from "@/lib/lab-status-store";
import { fmtAge, fmtBytesRate, fmtPct, fmtTemp, fmtWatts } from "@/lib/status/format";
import {
  Badge,
  EmptyState,
  Eyebrow,
  Nil,
  Panel,
  Skeleton,
  SparkStat,
  SyncRing,
  syncStateFromNode,
  type SparkPoint,
} from "@/components/ui";
import { MemoryBar } from "@/components/status/MemoryBar";
import { cn } from "@/lib/utils";

/*
  The Sparks — one card per node with its own hardware truth (GPU / SoC / NIC /
  NVMe temperatures, power, GPU util, CPU, memory with a 60 s trend), and the
  fabric between them: one line per RoCE rail, moving only while the RDMA byte
  counters say traffic is crossing it. The served model's rates live once, in the
  endpoint panel above — never copied onto TP nodes, which would read as 2×.

  Honesty rules: every sparkline sits on a real 60 s time axis ending at the
  server's "now" (gaps stay gaps); a node whose telemetry is older than
  staleAfterS (longer while polling) is dimmed with its age; a down node says Offline / SSH failed and
  shows no numbers; nothing animates while the data is stale.
*/

export const WINDOW_MS = 60_000;
const TEMP_WARN_C = 80;
/** Below this the rail is idle (control traffic only): no flow drawn. */
const FLOW_MIN_BPS = 1024;

function stateTone(state?: string): "ok" | "warn" | "danger" | "muted" {
  switch (state) {
    case "serving":
    case "serving_worker":
      return "ok";
    case "loading":
    case "stray":
      return "warn";
    case "offline":
    case "unreachable":
      return "danger";
    default:
      return "muted";
  }
}

/** Human label for a node state — the dot's accessible name and the badge share it. */
function stateLabel(node: ClusterNode): string {
  switch (node.state) {
    case "serving":
      return "Serving";
    case "serving_worker":
      return node.tp_rank != null ? `TP worker · rank ${node.tp_rank}` : "TP worker";
    case "offline":
      return "Offline";
    case "unreachable":
      return "SSH failed";
    case "loading":
      return "Loading";
    case "stray":
      return "Stray container";
    case "idle":
      return "Idle";
    default:
      return node.state || "Unknown";
  }
}

const isDown = (n: ClusterNode) => n.state === "offline" || n.state === "unreachable";

function placementLabel(mode?: string): string {
  switch (mode) {
    case "multi_aligned":
      return "Tensor parallel";
    case "single":
      return "Single node";
    case "loading":
      return "Loading";
    case "multi_partial":
      return "Partial multi-node";
    case "multi_mismatch":
      return "Model mismatch";
    case "none":
      return "No model loaded";
    default:
      return mode || "Unknown";
  }
}

function placementTone(mode?: string): "ok" | "warn" | "danger" | "muted" {
  switch (mode) {
    case "multi_aligned":
    case "single":
      return "ok";
    case "loading":
    case "multi_partial":
      return "warn";
    case "multi_mismatch":
      return "danger";
    default:
      return "muted";
  }
}

function speedLabel(mbps?: number | null): string | null {
  if (!mbps || mbps <= 0) return null;
  return mbps >= 1000 ? `${Math.round(mbps / 1000)}G` : `${mbps}M`;
}

function pts(samples: NodeSample[] | undefined, pick: (s: NodeSample) => number | null): SparkPoint[] {
  return (samples ?? []).map((s) => ({ t: s.t, v: pick(s) }));
}

/** Age (s) of a server-clock timestamp, from the snapshot's server "now". */
function ageS(serverNow: number | null | undefined, at: number | null | undefined): number | null {
  return serverNow != null && at != null ? Math.max(0, (serverNow - at) / 1000) : null;
}

const NodeCard = memo(function NodeCard({
  node,
  samples,
  serverNow,
  stale,
}: {
  node: ClusterNode;
  samples?: NodeSample[];
  serverNow?: number | null;
  stale?: boolean;
}) {
  const down = isDown(node);
  const serving = node.state === "serving" || node.state === "serving_worker";
  const transport = useLabStatusStore((s) => s.transport);
  const age = down ? null : ageS(serverNow, node.sampled_at);
  const old = stale || (age != null && age > staleAfterS(transport));
  const domain = serverNow != null ? ([serverNow - WINDOW_MS, serverNow] as const) : undefined;
  const series = down ? [] : samples;
  const nil = down ? "Offline" : "Awaiting";
  const memUsed = node.ram_gib != null && node.available_gib != null ? Math.max(0, node.ram_gib - node.available_gib) : null;
  const hot = node.temperature_c != null && node.temperature_c >= TEMP_WARN_C;
  const label = stateLabel(node);
  const minorTemps = [
    node.soc_temp_c != null ? `SoC ${fmtTemp(node.soc_temp_c)}` : null,
    node.nic_temp_c != null ? `NIC ${fmtTemp(node.nic_temp_c)}` : null,
    node.nvme_temp_c != null ? `NVMe ${fmtTemp(node.nvme_temp_c)}` : null,
  ].filter(Boolean);
  const rails = node.rails ?? [];

  return (
    <div
      className={cn(
        "animus-chamfer relative flex min-w-0 flex-col border bg-[color:var(--animus-glass)] p-3.5 transition-[border-color,opacity] duration-300",
        serving && "border-[color:color-mix(in_srgb,var(--color-lab-ok)_45%,transparent)]",
        (node.state === "loading" || node.state === "stray") &&
          "border-[color:color-mix(in_srgb,var(--color-lab-warn)_40%,transparent)]",
        down && "border-[color:color-mix(in_srgb,var(--color-lab-danger)_40%,transparent)]",
        !serving && !down && node.state !== "loading" && node.state !== "stray" && "border-lab-border",
      )}
      aria-label={`${node.label || node.id}: ${label}`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex min-w-0 items-center gap-2">
            <SyncRing state={old && !down ? "stale" : syncStateFromNode(node.state)} label={label} />
            <span className="truncate font-[family-name:var(--font-display)] text-[15px] font-semibold uppercase leading-none tracking-[0.14em] text-lab-text">
              {node.label || node.id}
            </span>
            <span className="animus-chamfer-sm shrink-0 border border-[color:var(--animus-hairline)] px-1.5 py-[3px] font-[family-name:var(--font-display)] text-[9px] font-semibold uppercase leading-none tracking-[0.16em] text-lab-muted">
              {node.local ? "local" : "remote"}
            </span>
          </div>
          <div className="mt-1.5 truncate font-mono text-[10px] text-lab-muted" title={node.cpu || undefined}>
            {node.hostname || node.id}
            {node.role ? ` · ${node.role}` : ""}
            {node.cpu ? ` · ${node.cpu}` : ""}
          </div>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1.5">
          <Badge tone={stateTone(node.state)} dot>
            {label}
          </Badge>
          {!down && (
            <Eyebrow
              className={cn("lab-num text-[9px]", old ? "text-lab-warn" : "text-lab-muted")}
              title={node.local ? "Telemetry sampled on this host every second" : "Telemetry streamed over ssh every second"}
            >
              {age == null ? "no reading" : old ? `${fmtAge(age)} old` : "live"}
            </Eyebrow>
          )}
        </div>
      </div>

      <div className={cn("flex flex-1 flex-col", (old || down) && "opacity-55")}>
        <div aria-hidden className="animus-rule my-3" />

        <div className="grid grid-cols-2 gap-x-3 gap-y-2.5 sm:grid-cols-3">
          <SparkStat
            label="GPU temp"
            unit="°C"
            points={pts(series, (s) => s.temp)}
            value={down ? null : (node.temperature_c ?? null)}
            domain={domain}
            min={25}
            max={95}
            nil={nil}
            tone={hot ? "text-lab-warn" : "text-lab-line"}
            className={hot ? "text-lab-warn" : undefined}
            title={hot ? `At or above ${TEMP_WARN_C}°C` : "nvidia-smi temperature.gpu"}
          />
          <SparkStat
            label="Power"
            unit=" W"
            points={pts(series, (s) => s.power)}
            value={down ? null : (node.power_w ?? null)}
            domain={domain}
            nil={nil}
            format={fmtWatts}
            tone="text-lab-line-2"
            title="GPU power draw (nvidia-smi)"
          />
          <SparkStat
            label="GPU util"
            unit="%"
            points={pts(series, (s) => s.util)}
            value={down ? null : (node.gpu_util_pct ?? null)}
            domain={domain}
            max={100}
            nil={nil}
            tone="text-lab-line"
            title="nvidia-smi utilization.gpu"
          />
        </div>

        <div className="lab-num mt-2 flex flex-wrap items-center gap-x-2.5 gap-y-1 font-mono text-[10px] text-lab-muted">
          <span title="All CPU cores, from /proc/stat">
            CPU{" "}
            {down || node.cpu_util_pct == null ? (
              <Nil word={nil} />
            ) : (
              // fixed width: "2%" → "20%" must not shift the temperatures after it
              <span className="inline-block w-[4ch] text-lab-text-dim">{fmtPct(node.cpu_util_pct)}</span>
            )}
          </span>
          {!down &&
            minorTemps.map((t) => (
              <span key={t} title="hwmon: acpitz (SoC), mlx5 (NIC), nvme (composite)">
                {t}
              </span>
            ))}
        </div>

        <MemoryBar
          className="mt-3"
          usedGib={down ? null : memUsed}
          totalGib={down ? null : node.ram_gib}
          reservedGib={node.engine_reserved_gib}
          swapUsedGib={node.swap_used_gib}
          swapTotalGib={node.swap_total_gib}
          pressure={node.mem_pressure}
          source="/proc/meminfo: MemTotal − MemAvailable"
          trend={series ? pts(series, (s) => s.mem) : undefined}
          domain={domain}
          offline={down}
        />

        <div className="mt-auto flex flex-wrap items-center gap-x-2 gap-y-1.5 border-t border-[color:var(--animus-hairline)] pt-2.5 font-mono text-[10px] text-lab-muted">
          {rails.length ? (
            rails.map((r) => (
              <span key={r.if} className="inline-flex items-center gap-1" title={`${r.if} ${r.ip}/${r.prefix}`}>
                <span
                  aria-hidden
                  className={cn("h-1.5 w-1.5 rotate-45", r.carrier === 1 ? "bg-lab-ok" : r.carrier === 0 ? "bg-lab-danger" : "bg-lab-muted")}
                />
                {r.ip}
                {speedLabel(r.speed_mbps) ? <span className="text-lab-text-dim">{speedLabel(r.speed_mbps)}</span> : null}
                {r.carrier === 0 ? <span className="text-lab-danger">down</span> : null}
              </span>
            ))
          ) : node.qsfp_ip ? (
            <span>qsfp {node.qsfp_ip}</span>
          ) : null}
          {node.tailscale_ip && <span>ts {node.tailscale_ip}</span>}
          {node.lan_ip && <span>lan {node.lan_ip}</span>}
          {!rails.length && !node.qsfp_ip && !node.tailscale_ip && !node.lan_ip && <Nil word="None" />}
        </div>
      </div>

      {(node.probe_error || node.telemetry_error) && (
        <div
          className="animus-chamfer-sm mt-2.5 truncate border border-[color:color-mix(in_srgb,var(--color-lab-danger)_35%,transparent)] bg-[color:color-mix(in_srgb,var(--color-lab-danger)_10%,transparent)] px-2 py-1 font-mono text-[10px] text-lab-danger"
          title={node.probe_error || node.telemetry_error || undefined}
        >
          {node.probe_error || `telemetry: ${node.telemetry_error}`}
        </div>
      )}
    </div>
  );
});

type Link = NonNullable<NonNullable<ClusterStatus["fabric"]>["links"]>[number];

/** Live RDMA bytes/s on a link's rail, from the sending side's counters. */
function linkRate(link: Link, byId: Map<string, ClusterNode>): { tx: number; rx: number } | null {
  const node = byId.get(link.from);
  const r = link.iface ? node?.rail_rates?.[link.iface] : undefined;
  if (!node || isDown(node) || !r) return null;
  return { tx: r.tx_bps, rx: r.rx_bps };
}

/**
 * Seconds per dash cycle for a rail carrying `peakBps`: one of three fixed speeds by
 * link utilisation (< 1 %, < 10 %, more). Stepped, not continuous: changing a running
 * CSS animation's duration re-phases it, so a value that moved with every 1 s sample
 * made the flow jump once a second. Unknown link speed: the middle step.
 */
export function flowDurS(peakBps: number, speedMbps: number | null | undefined): number {
  if (!speedMbps) return 1.2;
  const util = (peakBps * 8) / (speedMbps * 1e6);
  return util < 0.01 ? 2.4 : util < 0.1 ? 1.2 : 0.45;
}

/**
 * One rail: a line that flows only while RDMA bytes cross it (faster with more
 * link utilisation), a static dashed line when idle or unmeasured, red when down.
 */
function RailLine({ link, rate, stale }: { link: Link; rate: { tx: number; rx: number } | null; stale?: boolean }) {
  const up = !!link.ok;
  const speed = link.from_speed_mbps || link.to_speed_mbps || null;
  const peak = rate ? Math.max(rate.tx, rate.rx) : 0;
  const flowing = up && !stale && peak >= FLOW_MIN_BPS;
  const dur = `${flowDurS(peak, speed)}s`;
  return (
    <div className="min-w-0">
      <div className="flex items-baseline justify-between gap-2 font-mono text-[9.5px] text-lab-muted">
        <span className="truncate" title={`${link.from} ${link.iface ?? ""} → ${link.to} ${link.target_ip ?? ""}`}>
          {link.iface ?? link.target_ip ?? "link"}
        </span>
        <span className="lab-num shrink-0">
          {[speedLabel(speed), link.rtt_ms != null ? `${link.rtt_ms.toFixed(1)} ms` : null].filter(Boolean).join(" · ")}
        </span>
      </div>
      <div
        aria-hidden
        className={cn("lab-flow mt-1", up ? "text-lab-ok" : "text-lab-danger")}
        data-flowing={flowing ? "true" : undefined}
        style={{ "--flow-dur": dur } as CSSProperties}
      />
      <div className={cn("lab-num mt-1 font-mono text-[10px]", flowing ? "text-lab-text-dim" : "text-lab-muted")}>
        {!up ? (
          <span className="text-lab-danger">{link.error ? "down" : "no reply"}</span>
        ) : rate ? (
          <>
            ↑ {fmtBytesRate(rate.tx)} <span className="text-lab-muted">·</span> ↓ {fmtBytesRate(rate.rx)}
          </>
        ) : (
          <Nil word="Awaiting" />
        )}
      </div>
    </div>
  );
}

/** Between two cards: every rail joining them, stacked. */
function FabricBridge({ links, byId, stale }: { links: Link[]; byId: Map<string, ClusterNode>; stale?: boolean }) {
  const ok = links.length > 0 && links.every((l) => l.ok);
  return (
    <div
      className="flex flex-col justify-center gap-3 px-1 py-3 lg:w-[176px]"
      role="group"
      aria-label={ok ? `QSFP RoCE fabric up, ${links.length} rail${links.length === 1 ? "" : "s"}` : "Fabric problem"}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="animus-eyebrow">Fabric</span>
        <span className={cn("animus-eyebrow", ok ? "text-lab-ok!" : "text-lab-danger!")}>{ok ? "RoCE up" : links.length ? "check" : "no link"}</span>
      </div>
      {links.map((l, i) => (
        <RailLine key={`${l.from}-${l.to}-${l.iface ?? i}`} link={l} rate={linkRate(l, byId)} stale={stale} />
      ))}
    </div>
  );
}

/** Three or more nodes: every link as a row (head ↔ worker, worker ↔ worker). */
function FabricLinks({ links, byId, stale }: { links: Link[]; byId: Map<string, ClusterNode>; stale?: boolean }) {
  return (
    <div className="mt-3 border-t border-lab-border-subtle pt-3">
      <Eyebrow>Fabric · {links.length} link{links.length === 1 ? "" : "s"}</Eyebrow>
      <div className="mt-2 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {links.map((l, i) => (
          <div key={`${l.from}-${l.to}-${l.iface ?? i}`} className="min-w-0">
            <div className="mb-1 font-[family-name:var(--font-display)] text-[11px] font-semibold uppercase tracking-[0.14em] text-lab-text-dim">
              {l.from} → {l.to}
            </div>
            <RailLine link={l} rate={linkRate(l, byId)} stale={stale} />
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * How the served model is placed across the nodes, and how many answer — one line.
 * The model, TP size and per-node states are shown elsewhere (the endpoint hero and
 * each card); the message is kept only where it explains a problem.
 */
function PlacementStrip({ cluster, online, total }: { cluster: ClusterStatus; online: number; total: number }) {
  const multi = cluster.summary?.multi;
  const mode = multi?.mode || "none";
  const explain = mode === "multi_partial" || mode === "multi_mismatch";
  return (
    <div className="border-b border-lab-border-subtle px-3.5 py-2.5 sm:px-4">
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
        <div className="flex min-w-0 flex-wrap items-center gap-2.5">
          <Badge tone={placementTone(mode)} dot>
            {placementLabel(mode)}
          </Badge>
          {explain && multi?.message && <span className="min-w-0 text-[11px] leading-snug text-lab-muted">{multi.message}</span>}
        </div>
        <Eyebrow className={cn("lab-num", online === total ? "text-lab-ok" : "text-lab-danger")}>
          {online}/{total} online
        </Eyebrow>
      </div>
    </div>
  );
}

function ProbingPanel({ note }: { note: string }) {
  return (
    <Panel className="overflow-hidden">
      <div aria-busy="true" aria-label="Probing the cluster">
        <div className="border-b border-lab-border-subtle px-3.5 py-2.5 sm:px-4">
          <Badge tone="muted">{note}</Badge>
        </div>
        <div className="grid grid-cols-1 gap-3 p-3.5 sm:p-4 lg:grid-cols-[1fr_auto_1fr]">
          {/* about a node card's height, so the cards replace it without a jump */}
          <Skeleton className="min-h-[300px] lg:min-h-[240px]" />
          <Skeleton className="hidden min-h-[240px] w-[176px] lg:block" />
          <Skeleton className="min-h-[300px] lg:min-h-[240px]" />
        </div>
      </div>
    </Panel>
  );
}

export function ClusterPanel({
  cluster,
  loading,
  samples,
  serverNow,
  stale,
}: {
  cluster: ClusterStatus | null | undefined;
  loading?: boolean;
  /** per-node 60 s series from the lab-status store */
  samples?: Record<string, NodeSample[]>;
  /** the serve-engine host's clock now (ms): node ages and the sparkline window */
  serverNow?: number | null;
  /** the whole snapshot stopped updating: dim, and never animate */
  stale?: boolean;
}) {
  if (loading) return <ProbingPanel note="connecting…" />;
  // serve-engine just started: the first inventory (ssh to every node) is still running.
  if (cluster?.pending) return <ProbingPanel note="probing the Sparks…" />;

  if (!cluster) {
    return (
      <Panel padded>
        <EmptyState title="Cluster probe unavailable">
          Serve-engine didn’t return cluster topology. Check that the engine is up on :8765.
        </EmptyState>
      </Panel>
    );
  }

  if (cluster.error) {
    return (
      <Panel padded>
        <EmptyState title="Cluster probe failed">
          <span className="text-lab-danger">{cluster.error}</span>
        </EmptyState>
      </Panel>
    );
  }

  const nodes = cluster.nodes || [];
  const summary = cluster.summary;
  const online = summary?.nodes_online ?? nodes.filter((n) => !isDown(n)).length;
  const total = summary?.nodes_total ?? nodes.length;
  const links = cluster.fabric?.links ?? [];
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const pair = nodes.length === 2;
  const card = (n: ClusterNode) => (
    <NodeCard key={n.id} node={n} samples={samples?.[n.id]} serverNow={serverNow} stale={stale} />
  );

  return (
    // Untitled: the page's "Sparks" band already names it.
    <Panel className="overflow-hidden">
      <PlacementStrip cluster={cluster} online={online} total={total} />

      <div className="p-3.5 sm:p-4">
        {pair ? (
          <div className="grid grid-cols-1 items-stretch gap-3 lg:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)]">
            {card(nodes[0])}
            <FabricBridge links={links} byId={byId} stale={stale} />
            {card(nodes[1])}
          </div>
        ) : (
          <>
            <div className={cn("grid gap-3", nodes.length > 1 && "sm:grid-cols-2 xl:grid-cols-3")}>{nodes.map(card)}</div>
            {links.length > 0 && <FabricLinks links={links} byId={byId} stale={stale} />}
          </>
        )}
      </div>
    </Panel>
  );
}
