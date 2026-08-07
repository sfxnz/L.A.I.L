"use client";

import type { ClusterNode, ClusterStatus } from "@/lib/api";
import { Badge, EmptyState, Panel, Skeleton, StatusDot } from "@/components/ui";
import { cn } from "@/lib/utils";

function stateTone(state?: string): "ok" | "warn" | "danger" | "muted" | "accent" {
  switch (state) {
    case "serving":
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

  return (
    <div
      className={cn(
        "relative flex min-h-[168px] flex-col rounded-[14px] border bg-lab-panel2/60 p-4 transition-all duration-300",
        node.state === "serving" &&
          "border-[rgba(48,209,88,0.4)] shadow-[0_0_24px_rgba(48,209,88,0.12),inset_0_0_0_1px_rgba(48,209,88,0.08)]",
        node.state === "loading" && "border-[rgba(255,214,10,0.28)]",
        node.state === "offline" && "border-[rgba(255,69,58,0.28)]",
        node.state === "idle" && "border-lab-border",
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <StatusDot
              live={
                node.state === "serving" ? true : node.state === "offline" ? false : null
              }
              label={
                node.state === "serving"
                  ? "Serving"
                  : node.state === "offline"
                    ? "Offline"
                    : node.state === "loading"
                      ? "Loading"
                      : node.state === "idle"
                        ? "Idle"
                        : node.state || "Unknown"
              }
            />
            <span className="text-[15px] font-semibold tracking-[-0.02em] text-lab-text">
              {node.label || node.id}
            </span>
            {node.local && (
              <span className="rounded-md bg-lab-hover px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-[0.06em] text-lab-muted">
                local
              </span>
            )}
          </div>
          <div className="mt-1 font-mono text-[11px] text-lab-muted">
            {node.hostname || node.id}
            {node.role ? ` · ${node.role}` : ""}
          </div>
        </div>
        <Badge tone={stateTone(node.state)} dot>
          {node.state || "—"}
        </Badge>
      </div>

      <div className="mt-4 flex flex-1 flex-col justify-between gap-3">
        <div>
          <div className="text-[10px] font-medium uppercase tracking-[0.08em] text-lab-muted">
            Model
          </div>
          <div
            className={cn(
              "mt-1 truncate text-[14px] font-medium tracking-[-0.02em]",
              modelShort ? "text-lab-text" : "text-lab-muted",
            )}
            title={node.model_id || undefined}
          >
            {modelShort || "—"}
          </div>
          {node.tensor_parallel_size != null && (
            <div className="mt-0.5 font-mono text-[11px] text-lab-accent-bright">
              TP={node.tensor_parallel_size}
              {node.ray_hint ? " · ray" : ""}
            </div>
          )}
        </div>

        <div className="grid grid-cols-2 gap-2 border-t border-lab-border-subtle pt-3">
          <div>
            <div className="text-[10px] uppercase tracking-[0.06em] text-lab-muted">Free</div>
            <div
              className={cn(
                "mt-0.5 tabular-nums text-[13px] font-medium",
                free != null && free < 15 ? "text-lab-warn" : "text-lab-text-dim",
              )}
            >
              {free != null ? `${free} GiB` : "—"}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-[0.06em] text-lab-muted">QSFP</div>
            <div className="mt-0.5 tabular-nums text-[13px] font-medium text-lab-text-dim">
              {node.qsfp_carrier === 1 ? (
                <span className="text-lab-ok">{speed || "up"}</span>
              ) : node.qsfp_carrier === 0 ? (
                <span className="text-lab-danger">down</span>
              ) : (
                "—"
              )}
            </div>
          </div>
          <div className="col-span-2">
            <div className="text-[10px] uppercase tracking-[0.06em] text-lab-muted">Addrs</div>
            <div className="mt-0.5 space-y-0.5 font-mono text-[10px] leading-relaxed text-lab-muted">
              {node.qsfp_ip && <div>qsfp {node.qsfp_ip}</div>}
              {node.tailscale_ip && <div>ts {node.tailscale_ip}</div>}
              {node.lan_ip && <div>lan {node.lan_ip}</div>}
              {!node.qsfp_ip && !node.tailscale_ip && !node.lan_ip && <div>—</div>}
            </div>
          </div>
        </div>
      </div>

      {node.probe_error && (
        <div className="mt-3 truncate rounded-lg bg-[rgba(255,69,58,0.1)] px-2 py-1 text-[10px] text-lab-danger" title={node.probe_error}>
          {node.probe_error}
        </div>
      )}
    </div>
  );
}

function FabricBridge({ cluster }: { cluster: ClusterStatus }) {
  const link = cluster.fabric?.links?.[0];
  const ok = !!cluster.fabric?.ok;
  const rtt = link?.rtt_ms;
  const speed = link?.from_speed_mbps || link?.to_speed_mbps;
  const speedG = speed && speed > 0 ? Math.round(speed / 1000) : null;

  return (
    <div
      className="flex min-h-[168px] flex-col items-center justify-center gap-2 px-2 py-4"
      role="img"
      aria-label={
        ok
          ? `QSFP RoCE fabric up${speedG ? `, ${speedG}G` : ""}${rtt != null ? `, ${rtt.toFixed(1)} ms` : ""}`
          : "Cluster fabric down"
      }
    >
      <div className="text-[10px] font-medium uppercase tracking-[0.08em] text-lab-muted">
        Fabric
      </div>
      <div className="relative flex w-full max-w-[150px] items-center">
        <div
          className={cn(
            "h-[2px] flex-1 rounded-full",
            ok ? "lab-fabric-line text-lab-ok/80" : "bg-lab-danger/50",
          )}
          aria-hidden
        />
        <div
          className={cn(
            "mx-1.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border text-[10px] font-semibold",
            ok
              ? "border-[rgba(48,209,88,0.4)] bg-[rgba(48,209,88,0.12)] text-lab-ok shadow-[0_0_14px_rgba(48,209,88,0.25)]"
              : "border-[rgba(255,69,58,0.35)] bg-[rgba(255,69,58,0.1)] text-lab-danger",
          )}
        >
          {ok ? "OK" : "!"}
        </div>
        <div
          className={cn(
            "h-[2px] flex-1 rounded-full",
            ok ? "lab-fabric-line text-lab-ok/80" : "bg-lab-danger/50",
          )}
          aria-hidden
        />
      </div>
      <div className="text-center">
        <div className="text-[12px] font-medium text-lab-text-dim">
          {ok ? "QSFP RoCE" : "Fabric down"}
        </div>
        <div className="mt-0.5 font-mono text-[10px] tabular-nums text-lab-muted">
          {[speedG ? `${speedG}G` : null, rtt != null ? `${rtt.toFixed(1)} ms` : null]
            .filter(Boolean)
            .join(" · ") || link?.target_ip || "—"}
        </div>
      </div>
    </div>
  );
}

function LoadStrip({ cluster }: { cluster: ClusterStatus }) {
  const multi = cluster.summary?.multi;
  const nodes = cluster.nodes || [];
  const mode = multi?.mode || "none";
  const modelShort = multi?.model_id?.split("/").pop();

  return (
    <div
      className={cn(
        "rounded-[12px] border border-lab-border-subtle bg-lab-editor/80 px-4 py-3.5 transition-shadow duration-300",
        mode === "multi_aligned" && "lab-strip-live",
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <Badge tone={multiTone(mode)} dot>
            {multiLabel(mode)}
          </Badge>
          {modelShort && (
            <span className="truncate font-mono text-[12px] text-lab-text-dim" title={multi?.model_id || undefined}>
              {modelShort}
            </span>
          )}
          {multi?.tensor_parallel_hint != null && (
            <span className="font-mono text-[11px] text-lab-accent-bright">
              TP={multi.tensor_parallel_hint}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          {nodes.map((n) => {
            const filled = n.state === "serving";
            const loading = n.state === "loading";
            return (
              <div key={n.id} className="flex items-center gap-1.5">
                <div
                  className={cn(
                    "h-2.5 w-10 rounded-full border transition-colors",
                    filled && "border-lab-ok bg-lab-ok shadow-[0_0_12px_rgba(48,209,88,0.35)]",
                    loading && "border-lab-warn bg-lab-warn/70 animate-pulse",
                    !filled && !loading && n.state === "idle" && "border-lab-border bg-lab-hover",
                    n.state === "offline" && "border-lab-danger/50 bg-[rgba(255,69,58,0.2)]",
                  )}
                  title={`${n.id}: ${n.state}${n.model_id ? ` · ${n.model_id}` : ""}`}
                />
                <span className="text-[10px] font-medium uppercase tracking-[0.04em] text-lab-muted">
                  {n.id}
                </span>
              </div>
            );
          })}
        </div>
      </div>
      {multi?.message && (
        <p className="mt-2 text-[12px] leading-snug text-lab-muted">{multi.message}</p>
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
        <div className="space-y-3 p-3 sm:p-4" aria-busy="true" aria-label="Loading cluster">
          <Skeleton className="h-14 w-full rounded-[12px]" />
          <div className="grid grid-cols-1 gap-2 lg:grid-cols-[1fr_auto_1fr]">
            <Skeleton className="min-h-[168px] rounded-[14px]" />
            <Skeleton className="hidden h-24 w-16 rounded-full lg:block" />
            <Skeleton className="min-h-[168px] rounded-[14px]" />
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
        <span
          className={cn(
            "flex items-center gap-2 text-[11px] font-medium tabular-nums",
            healthy ? "text-lab-ok" : "text-lab-danger",
          )}
        >
          <StatusDot live={healthy} label={healthy ? "Cluster healthy" : "Cluster issue"} />
          {summary?.nodes_online ?? 0}/{summary?.nodes_total ?? nodes.length} online
          {summary?.nodes_serving ? ` · ${summary.nodes_serving} serving` : ""}
        </span>
      }
    >
      <div className="space-y-3 p-3 sm:p-4">
        <LoadStrip cluster={cluster} />

        <div className="grid grid-cols-1 items-stretch gap-2 lg:grid-cols-[1fr_auto_1fr]">
          {nodes[0] ? <NodeCard node={nodes[0]} /> : <div />}
          {nodes.length >= 2 ? (
            <FabricBridge cluster={cluster} />
          ) : (
            <div className="hidden lg:block" />
          )}
          {nodes[1] ? <NodeCard node={nodes[1]} /> : nodes.length < 2 ? null : <div />}
        </div>

        {nodes.length > 2 && (
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {nodes.slice(2).map((n) => (
              <NodeCard key={n.id} node={n} />
            ))}
          </div>
        )}
      </div>
    </Panel>
  );
}
