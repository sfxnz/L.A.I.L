"use client";

import type { ReactNode } from "react";
import type { ClusterNode, ClusterStatus } from "@/lib/api";
import { Badge, EmptyState, Panel, Skeleton, StatusDot } from "@/components/ui";
import { cn } from "@/lib/utils";

/*
  Cluster readout — the hero instrument on Status.

  Two node cards bridged by the fabric spine. Everything here obeys the Animus
  contract in app/globals.css: cut corners, condensed uppercase eyebrows, cyan/
  slate hairlines as STRUCTURE, crimson reserved for the page's single accent
  (so TP hints ride lab-line, not lab-accent). All state colour goes through
  color-mix on a lab-* token so the light reconstruction plate resolves too.
*/

/**
 * The one null treatment on this surface, shared with app/status/page.tsx: an
 * absent value is a condensed STATE WORD, never a bare em-dash — an em-dash
 * reads as broken data. "Awaiting" = hasn't reported yet, "None" = settled and
 * genuinely empty. .animus-eyebrow supplies the muted colour and the caps.
 */
function Nil({ word = "Awaiting" }: { word?: "Awaiting" | "None" }) {
  return (
    <span className="inline-flex items-center gap-1.5 align-middle">
      <span
        aria-hidden
        className="h-[5px] w-[5px] shrink-0 rotate-45 border border-[color:var(--animus-hairline)]"
      />
      <span className="animus-eyebrow">{word}</span>
    </span>
  );
}

/** Eyebrow + value stack. The unit every HUD readout on this panel is built from. */
function Readout({
  label,
  children,
  className,
}: {
  label: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("min-w-0", className)}>
      <div className="animus-eyebrow truncate">{label}</div>
      <div className="mt-1">{children}</div>
    </div>
  );
}

function stateTone(state?: string): "ok" | "warn" | "danger" | "muted" | "accent" {
  switch (state) {
    case "serving":
    case "serving_worker":
      return "ok";
    case "loading":
      return "warn";
    case "offline":
      return "danger";
    case "idle":
      return "muted";
    default:
      return "muted";
  }
}

/**
 * Human label for a node state — used for BOTH the dot's accessible name and
 * the badge, so the raw enum never leaks. `serving_worker` is a headless
 * multi-node TP worker: it has no /v1/models by design and counts as serving.
 */
function stateLabel(state?: string): string {
  switch (state) {
    case "serving":
      return "Serving";
    case "serving_worker":
      return "TP worker";
    case "offline":
      return "Offline";
    case "loading":
      return "Loading";
    case "idle":
      return "Idle";
    default:
      return state || "Unknown";
  }
}

function multiTone(mode?: string): "ok" | "warn" | "danger" | "muted" | "accent" {
  switch (mode) {
    case "multi_aligned":
      return "ok";
    case "single":
      return "accent";
    case "loading":
    case "multi_partial":
      return "warn";
    case "multi_mismatch":
      return "danger";
    default:
      return "muted";
  }
}

function multiLabel(mode?: string): string {
  switch (mode) {
    case "multi_aligned":
      return "Multi-node aligned";
    case "single":
      return "Single-node serve";
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

function NodeCard({ node }: { node: ClusterNode }) {
  const modelShort = node.model_id?.split("/").pop() || null;
  const free = node.available_gib;
  const speed =
    node.qsfp_speed_mbps && node.qsfp_speed_mbps > 0
      ? node.qsfp_speed_mbps >= 1000
        ? `${Math.round(node.qsfp_speed_mbps / 1000)}G`
        : `${node.qsfp_speed_mbps}M`
      : null;
  const serving = node.state === "serving" || node.state === "serving_worker";
  const label = stateLabel(node.state);
  const hasAddrs = !!(node.qsfp_ip || node.tailscale_ip || node.lan_ip);

  return (
    <div
      className={cn(
        // Cut corners, not radius. The chamfer clips box-shadow, so the serving
        // tell is an INSET bloom rather than an outer glow.
        "animus-chamfer animus-bracketed relative flex min-h-[172px] flex-col border bg-[color:var(--animus-glass)] p-3.5",
        // The chamfer clip-path would shear brackets sitting on the -1px edge —
        // inset them so all four corners actually render (same trick as Panel).
        "before:top-[3px]! before:left-[3px]! after:right-[3px]! after:bottom-[3px]!",
        "transition-[border-color,box-shadow] duration-300",
        serving &&
          "border-[color:color-mix(in_srgb,var(--color-lab-ok)_45%,transparent)] shadow-[inset_0_0_0_1px_color-mix(in_srgb,var(--color-lab-ok)_10%,transparent),inset_0_0_30px_-14px_color-mix(in_srgb,var(--color-lab-ok)_75%,transparent)]",
        node.state === "loading" &&
          "border-[color:color-mix(in_srgb,var(--color-lab-warn)_40%,transparent)]",
        node.state === "offline" &&
          "border-[color:color-mix(in_srgb,var(--color-lab-danger)_40%,transparent)]",
        !serving &&
          node.state !== "loading" &&
          node.state !== "offline" &&
          "border-lab-border",
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex min-w-0 items-center gap-2">
            <StatusDot
              live={serving ? true : node.state === "offline" ? false : null}
              label={label}
            />
            <span className="truncate font-[family-name:var(--font-display)] text-[15px] font-semibold uppercase leading-none tracking-[0.14em] text-lab-text">
              {node.label || node.id}
            </span>
            {/* Always render the origin tag, never only for the local node —
                an occupied slot on one card and an empty one on the other
                breaks the mirror the two-column composition rests on. */}
            <span className="animus-chamfer-sm shrink-0 border border-[color:var(--animus-hairline)] px-1.5 py-[3px] font-[family-name:var(--font-display)] text-[9px] font-semibold uppercase leading-none tracking-[0.16em] text-lab-muted">
              {node.local ? "local" : "remote"}
            </span>
          </div>
          <div className="mt-1.5 truncate font-mono text-[10px] text-lab-muted">
            {node.hostname || node.id}
            {node.role ? ` · ${node.role}` : ""}
          </div>
        </div>
        <Badge tone={stateTone(node.state)} dot>
          {label}
        </Badge>
      </div>

      <div aria-hidden className="animus-rule my-3" />

      <Readout label="Model">
        <div
          className="truncate text-[13px] font-medium tracking-[-0.01em] text-lab-text"
          title={node.model_id || undefined}
        >
          {modelShort ?? <Nil word="None" />}
        </div>
        {node.tensor_parallel_size != null && (
          <div className="mt-1 font-mono text-[10px] tabular-nums text-lab-line-bright">
            TP={node.tensor_parallel_size}
            {node.ray_hint ? " · ray" : ""}
          </div>
        )}
      </Readout>

      <div className="mt-auto grid grid-cols-2 gap-x-3 gap-y-2.5 border-t border-[color:var(--animus-hairline)] pt-3">
        <Readout label="Free">
          <div
            className={cn(
              "font-[family-name:var(--font-display)] text-[15px] font-semibold leading-none tabular-nums",
              free != null && free < 15 ? "text-lab-warn" : "text-lab-text-dim",
            )}
          >
            {free != null ? `${free} GiB` : <Nil />}
          </div>
        </Readout>
        <Readout label="QSFP">
          <div className="font-[family-name:var(--font-display)] text-[15px] font-semibold uppercase leading-none tabular-nums">
            {node.qsfp_carrier === 1 ? (
              <span className="text-lab-ok">{speed || "up"}</span>
            ) : node.qsfp_carrier === 0 ? (
              <span className="text-lab-danger">down</span>
            ) : (
              <Nil />
            )}
          </div>
        </Readout>
        <Readout label="Addrs" className="col-span-2">
          <div className="space-y-0.5 font-mono text-[10px] leading-[1.5] text-lab-muted">
            {node.qsfp_ip && (
              <div className="truncate">
                <span className="text-lab-line">qsfp</span> {node.qsfp_ip}
              </div>
            )}
            {node.tailscale_ip && (
              <div className="truncate">
                <span className="text-lab-line">ts</span> {node.tailscale_ip}
              </div>
            )}
            {node.lan_ip && (
              <div className="truncate">
                <span className="text-lab-line">lan</span> {node.lan_ip}
              </div>
            )}
            {!hasAddrs && <Nil word="None" />}
          </div>
        </Readout>
      </div>

      {node.probe_error && (
        <div
          className="animus-chamfer-sm mt-3 truncate border border-[color:color-mix(in_srgb,var(--color-lab-danger)_35%,transparent)] bg-[color:color-mix(in_srgb,var(--color-lab-danger)_10%,transparent)] px-2 py-1 font-mono text-[10px] text-lab-danger"
          title={node.probe_error}
        >
          {node.probe_error}
        </div>
      )}
    </div>
  );
}

/**
 * The spine. Animated dashes (.lab-fabric-line, gated on prefers-reduced-motion
 * in globals.css) run node → node through a diamond hub, so a live QSFP fabric
 * reads as flow rather than as a static divider.
 */
function FabricBridge({ cluster }: { cluster: ClusterStatus }) {
  const link = cluster.fabric?.links?.[0];
  const ok = !!cluster.fabric?.ok;
  const rtt = link?.rtt_ms;
  const speed = link?.from_speed_mbps || link?.to_speed_mbps;
  const speedG = speed && speed > 0 ? Math.round(speed / 1000) : null;
  const readout =
    [speedG ? `${speedG}G` : null, rtt != null ? `${rtt.toFixed(1)} ms` : null]
      .filter(Boolean)
      .join(" · ") ||
    link?.target_ip ||
    null;

  return (
    <div
      className="flex min-h-[172px] flex-col items-center justify-center gap-3 px-1 py-4 lg:w-[172px]"
      role="img"
      aria-label={
        ok
          ? `QSFP RoCE fabric up${speedG ? `, ${speedG}G` : ""}${rtt != null ? `, ${rtt.toFixed(1)} ms` : ""}`
          : "Cluster fabric down"
      }
    >
      <div className="animus-eyebrow">Fabric</div>

      <div
        aria-hidden
        className={cn(
          "flex w-full min-w-[124px] items-center",
          ok ? "text-lab-ok" : "text-lab-danger",
        )}
      >
        <span className="h-[5px] w-[5px] shrink-0 rotate-45 border border-current" />
        <span
          className={cn(
            "h-[2px] flex-1",
            ok ? "lab-fabric-line" : "bg-current opacity-45",
          )}
        />
        <span
          className={cn(
            "mx-1.5 flex h-[26px] w-[26px] shrink-0 rotate-45 items-center justify-center border",
            ok
              ? "border-current bg-[color:color-mix(in_srgb,var(--color-lab-ok)_14%,transparent)] shadow-[0_0_16px_color-mix(in_srgb,var(--color-lab-ok)_35%,transparent)]"
              : "border-current bg-[color:color-mix(in_srgb,var(--color-lab-danger)_12%,transparent)]",
          )}
        >
          <span className="-rotate-45 font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.1em]">
            {ok ? "ok" : "!"}
          </span>
        </span>
        <span
          className={cn(
            "h-[2px] flex-1",
            ok ? "lab-fabric-line" : "bg-current opacity-45",
          )}
        />
        <span className="h-[5px] w-[5px] shrink-0 rotate-45 border border-current" />
      </div>

      <div className="flex flex-col items-center gap-1 text-center">
        <div
          className={cn(
            "font-[family-name:var(--font-display)] text-[11px] font-semibold uppercase leading-none tracking-[0.16em]",
            ok ? "text-lab-text-dim" : "text-lab-danger",
          )}
        >
          {ok ? "QSFP RoCE" : "Fabric down"}
        </div>
        <div className="font-mono text-[10px] tabular-nums text-lab-muted">
          {readout ?? <Nil />}
        </div>
      </div>
    </div>
  );
}

/**
 * Load map — a flush readout bar welded to the panel head by a hairline, not a
 * nested card. Keeping it borderless is what stops the cluster reading as a
 * box-inside-a-box.
 */
function LoadStrip({ cluster }: { cluster: ClusterStatus }) {
  const multi = cluster.summary?.multi;
  const nodes = cluster.nodes || [];
  const mode = multi?.mode || "none";
  const modelShort = multi?.model_id?.split("/").pop();

  return (
    <div
      className={cn(
        "border-b border-lab-border-subtle px-3.5 py-2.5 transition-shadow duration-300 sm:px-4",
        mode === "multi_aligned" && "lab-strip-live",
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="animus-eyebrow shrink-0">Load map</span>
          <span
            aria-hidden
            className="h-3 w-px shrink-0 bg-[color:var(--animus-hairline)]"
          />
          <Badge tone={multiTone(mode)} dot>
            {multiLabel(mode)}
          </Badge>
          {modelShort && (
            <span
              className="truncate font-mono text-[11px] text-lab-text-dim"
              title={multi?.model_id || undefined}
            >
              {modelShort}
            </span>
          )}
          {multi?.tensor_parallel_hint != null && (
            <span className="shrink-0 font-mono text-[10px] tabular-nums text-lab-line-bright">
              TP={multi.tensor_parallel_hint}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {nodes.map((n) => {
            const filled = n.state === "serving" || n.state === "serving_worker";
            const loading = n.state === "loading";
            return (
              <div key={n.id} className="flex items-center gap-1.5">
                <div
                  className={cn(
                    "h-2 w-10 border transition-colors",
                    filled &&
                      "border-lab-ok bg-lab-ok shadow-[0_0_12px_color-mix(in_srgb,var(--color-lab-ok)_45%,transparent)]",
                    loading && "animate-pulse border-lab-warn bg-lab-warn/70",
                    !filled &&
                      !loading &&
                      n.state === "idle" &&
                      "border-lab-border bg-lab-hover",
                    n.state === "offline" &&
                      "border-[color:color-mix(in_srgb,var(--color-lab-danger)_50%,transparent)] bg-[color:color-mix(in_srgb,var(--color-lab-danger)_20%,transparent)]",
                  )}
                  title={`${n.id}: ${n.state === "serving_worker" ? "TP worker (headless)" : n.state}${n.model_id ? ` · ${n.model_id}` : ""}`}
                />
                <span className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.14em] text-lab-muted">
                  {n.id}
                </span>
              </div>
            );
          })}
        </div>
      </div>
      {multi?.message && (
        <p className="mt-1.5 text-[11px] leading-snug text-lab-muted">{multi.message}</p>
      )}
    </div>
  );
}

export function ClusterPanel({
  cluster,
  loading,
}: {
  cluster: ClusterStatus | null | undefined;
  loading?: boolean;
}) {
  if (loading) {
    return (
      <Panel
        title="Cluster"
        action={<Badge tone="muted">probing…</Badge>}
        className="overflow-hidden"
      >
        <div aria-busy="true" aria-label="Loading cluster">
          <div className="border-b border-lab-border-subtle px-3.5 py-2.5 sm:px-4">
            <div className="flex items-center justify-between gap-3">
              <Skeleton className="h-3.5 w-44" />
              <Skeleton className="h-2.5 w-28" />
            </div>
          </div>
          <div className="grid grid-cols-1 gap-3 p-3.5 sm:p-4 lg:grid-cols-[1fr_auto_1fr]">
            <Skeleton className="min-h-[172px]" />
            <Skeleton className="hidden min-h-[172px] w-[172px] lg:block" />
            <Skeleton className="min-h-[172px]" />
          </div>
        </div>
      </Panel>
    );
  }

  if (!cluster) {
    return (
      <Panel title="Cluster" padded>
        <EmptyState title="Cluster probe unavailable">
          Serve-engine didn’t return dual-Spark topology. Check :8765 and SSH to spark2 when you
          need the fabric map.
        </EmptyState>
      </Panel>
    );
  }

  if (cluster.error) {
    return (
      <Panel title="Cluster" padded>
        <EmptyState title="Cluster probe failed">
          <span className="text-lab-danger">{cluster.error}</span>
        </EmptyState>
      </Panel>
    );
  }

  const nodes = cluster.nodes || [];
  const summary = cluster.summary;
  const healthy = !!summary?.healthy;

  return (
    <Panel
      className="overflow-hidden"
      title="Cluster"
      action={
        <span className="flex shrink-0 items-center gap-2">
          <StatusDot live={healthy} label={healthy ? "Cluster healthy" : "Cluster issue"} />
          <span
            className={cn(
              "font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase leading-none tracking-[0.16em] tabular-nums",
              healthy ? "text-lab-ok" : "text-lab-danger",
            )}
          >
            {summary?.nodes_online ?? 0}/{summary?.nodes_total ?? nodes.length} online
            {summary?.nodes_serving ? ` · ${summary.nodes_serving} serving` : ""}
          </span>
        </span>
      }
    >
      <LoadStrip cluster={cluster} />

      <div className="p-3.5 sm:p-4">
        <div className="grid grid-cols-1 items-stretch gap-3 lg:grid-cols-[1fr_auto_1fr]">
          {nodes[0] ? <NodeCard node={nodes[0]} /> : <div />}
          {nodes.length >= 2 ? (
            <FabricBridge cluster={cluster} />
          ) : (
            <div className="hidden lg:block" />
          )}
          {nodes[1] ? <NodeCard node={nodes[1]} /> : nodes.length < 2 ? null : <div />}
        </div>

        {nodes.length > 2 && (
          <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {nodes.slice(2).map((n) => (
              <NodeCard key={n.id} node={n} />
            ))}
          </div>
        )}
      </div>
    </Panel>
  );
}
