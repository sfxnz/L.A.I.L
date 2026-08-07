"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { api, type LabStatus, type RunRow } from "@/lib/api";
import { ClusterPanel } from "@/components/ClusterPanel";
import {
  Badge,
  Btn,
  Callout,
  EmptyState,
  Metric,
  Panel,
  Skeleton,
  StatusDot,
  btnClass,
} from "@/components/ui";
import { cn } from "@/lib/utils";
import { usePageTitle } from "@/lib/usePageTitle";

function backendLabel(k: string) {
  if (k.toLowerCase() === "vllm") return "vLLM";
  if (k.toLowerCase() === "llamacpp") return "llama.cpp";
  return k;
}

export default function StatusPage() {
  usePageTitle("Status");
  const [status, setStatus] = useState<LabStatus | null>(null);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const refresh = useCallback(async (opts?: { soft?: boolean }) => {
    if (!opts?.soft) setRefreshing(true);
    try {
      const [s, r] = await Promise.all([
        api.labStatus(),
        api.runs().catch(() => [] as RunRow[]),
      ]);
      setStatus(s);
      setRuns(r);
      setErr(null);
    } catch (e) {
      setErr(String((e as Error).message || e));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh({ soft: true }), 6000);
    return () => clearInterval(t);
  }, [refresh]);

  const serve = status?.serve;
  const cluster = status?.cluster || serve?.cluster || null;
  const healthy = !!(serve && !serve.unreachable && serve.healthy);
  const modelShort = loading ? "—" : serve?.model_id?.split("/").pop() || "None loaded";
  const freeGib = serve?.hardware?.available_gib;
  const clusterHealthy = !!cluster?.summary?.healthy;
  const multiMode = cluster?.summary?.multi?.mode;

  return (
    <div className="space-y-6 lab-fade-in">
      <div className="page-header">
        <div>
          <div className="mb-2 flex flex-wrap items-center gap-3" aria-live="polite">
            <div className="flex items-center gap-2">
              <StatusDot
                live={loading ? null : healthy}
                label={
                  loading
                    ? "Checking endpoint"
                    : healthy
                      ? "Endpoint live"
                      : err
                        ? "Controller unreachable"
                        : "No model serving"
                }
              />
              <span className="text-[11px] font-medium uppercase tracking-[0.08em] text-lab-muted">
                {loading
                  ? "Checking…"
                  : healthy
                    ? "Endpoint live"
                    : err
                      ? "Controller down"
                      : "No model serving"}
              </span>
            </div>
            {!loading && cluster && (
              <div className="flex items-center gap-2">
                <StatusDot
                  live={clusterHealthy}
                  label={clusterHealthy ? "Cluster fabric ok" : "Cluster issue"}
                />
                <span className="text-[11px] font-medium uppercase tracking-[0.08em] text-lab-muted">
                  {clusterHealthy ? "Cluster fabric ok" : "Cluster issue"}
                  {multiMode && multiMode !== "none"
                    ? ` · ${multiMode.replace(/_/g, " ")}`
                    : ""}
                </span>
              </div>
            )}
          </div>
          <h1 className="page-title">Status</h1>
          <p className="page-sub">
            Dual-Spark cluster health, model load map, hardware headroom, and recent runs.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Btn variant="secondary" size="sm" onClick={() => void refresh()} loading={refreshing}>
            Refresh
          </Btn>
          <Link href="/server" className={btnClass("primary", "sm")}>
            Serve
          </Link>
          <Link
            href="/evals"
            className="px-1.5 text-[12px] font-medium text-lab-muted transition-colors hover:text-lab-text"
          >
            Evals →
          </Link>
        </div>
      </div>

      {err && (
        <Callout
          tone="danger"
          title="Couldn’t reach the lab controller"
          action={
            <Btn variant="secondary" size="sm" onClick={() => void refresh()} loading={refreshing}>
              Retry
            </Btn>
          }
          onDismiss={() => setErr(null)}
        >
          {err}. Check that <code className="text-lab-text">bun run dev</code> is up on spark1
          (ports 3000 / 8787 / 8765).
        </Callout>
      )}

      {!loading && !healthy && !err && (
        <Callout
          tone="warn"
          title="No model endpoint right now"
          action={
            <Link href="/server" className={btnClass("secondary", "sm")}>
              Open Serve
            </Link>
          }
        >
          Status is up, but vLLM isn’t healthy. Load a model from Serve when you’re ready — cluster
          fabric can still be fine with nothing loaded.
        </Callout>
      )}

      <div className="bento lab-rise">
        <div className="bento-span-12">
          <ClusterPanel cluster={cluster} loading={loading} />
        </div>
      </div>

      <div className="bento lab-rise lab-rise-1">
        <div className="bento-span-3">
          <Metric
            label="vLLM endpoint"
            value={loading ? "—" : healthy ? "Healthy" : "Idle"}
            sub={loading ? "Probing :8000" : serve?.base_url || "No endpoint configured"}
            tone={loading ? undefined : healthy ? "ok" : "muted"}
            large
            loading={loading}
          />
        </div>
        <div className="bento-span-3">
          <Metric
            label="Model"
            value={modelShort}
            sub={serve?.model_id || "No model loaded"}
            loading={loading}
          />
        </div>
        <div className="bento-span-3">
          <Metric
            label="Memory free"
            value={freeGib != null ? `${freeGib} GiB` : "—"}
            sub={(() => {
              const raw = [serve?.hardware?.gpu_sku, serve?.hardware?.cpu]
                .filter(Boolean)
                .map(String)
                .join(" · ")
                .replace(/\s*,?\s*\[?N\/A\]?/gi, "")
                .replace(/\s{2,}/g, " ")
                .replace(/\s*·\s*$/g, "")
                .trim();
              return raw || "Host headroom";
            })()}
            tone={freeGib != null && freeGib < 15 ? "warn" : undefined}
            loading={loading}
            progress={
              freeGib != null && serve?.hardware?.ram_gib
                ? (freeGib / serve.hardware.ram_gib) * 100
                : null
            }
          />
        </div>
        <div className="bento-span-3">
          <Metric
            label="Default backend"
            value={
              status?.defaultBackend === "vllm" ? "vLLM" : status?.defaultBackend || "—"
            }
            sub={status?.defaultModel || undefined}
            loading={loading}
          />
        </div>
      </div>

      <div className="bento lab-rise lab-rise-2">
        <div className="bento-span-6">
          <Panel
            className="flex h-full min-h-[220px] flex-col"
            title="Backends"
            action={
              <span className="text-[11px] tabular-nums text-lab-muted">
                {status ? Object.keys(status.backends || {}).length : "—"} registered
              </span>
            }
          >
            <div className="flex flex-1 flex-col space-y-0.5 p-1.5">
              {loading && (
                <div className="space-y-2 p-2" aria-busy="true" aria-label="Loading backends">
                  {[0, 1].map((i) => (
                    <div key={i} className="flex items-center justify-between gap-3 px-3 py-2.5">
                      <div className="min-w-0 flex-1 space-y-2">
                        <Skeleton className="h-3.5 w-24" />
                        <Skeleton className="h-3 w-40" />
                      </div>
                      <Skeleton className="h-5 w-12 rounded-full" />
                    </div>
                  ))}
                </div>
              )}
              {!loading &&
                status &&
                Object.entries(status.backends || {}).map(([k, v]) => (
                  <div
                    key={k}
                    className="flex items-center justify-between gap-3 rounded-[10px] px-3 py-2.5 transition-colors hover:bg-lab-hover"
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <StatusDot
                          live={v.ok ? true : null}
                          label={`${backendLabel(k)} ${v.ok ? "up" : "idle"}`}
                        />
                        <span className="text-[13px] font-medium tracking-[-0.01em] text-lab-text">
                          {backendLabel(k)}
                        </span>
                      </div>
                      <div className="mt-0.5 truncate pl-[15px] font-mono text-[11px] text-lab-muted">
                        {v.url}
                      </div>
                    </div>
                    <Badge tone={v.ok ? "ok" : "muted"}>{v.ok ? "up" : "idle"}</Badge>
                  </div>
                ))}
              {!loading && status && Object.keys(status.backends || {}).length === 0 && (
                <EmptyState title="No backends registered">
                  Check Configure for vLLM / llama.cpp URLs.
                </EmptyState>
              )}
            </div>
          </Panel>
        </div>

        <div className="bento-span-6">
          <Panel
            className="flex h-full min-h-[220px] flex-col"
            title="Containers"
            action={
              serve?.headroom ? (
                <Badge
                  tone={
                    serve.headroom === "critical"
                      ? "danger"
                      : serve.headroom === "tight"
                        ? "warn"
                        : "ok"
                  }
                >
                  headroom {serve.headroom}
                </Badge>
              ) : null
            }
          >
            <div className="flex flex-1 flex-col space-y-0.5 p-1.5">
              {loading && (
                <div className="space-y-2 p-2" aria-busy="true" aria-label="Loading containers">
                  {[0, 1].map((i) => (
                    <div key={i} className="space-y-2 px-3 py-2.5">
                      <Skeleton className="h-3.5 w-32" />
                      <Skeleton className="h-3 w-48" />
                    </div>
                  ))}
                </div>
              )}
              {!loading && (serve?.containers || []).length === 0 && (
                <div className="flex flex-1 items-center justify-center">
                  <EmptyState
                    title="No containers"
                    action={
                      <Link href="/server" className={btnClass("secondary", "sm")}>
                        Open Serve
                      </Link>
                    }
                  >
                    Nothing running — load a model when you’re ready.
                  </EmptyState>
                </div>
              )}
              {!loading &&
                (serve?.containers || []).map((c) => (
                  <div
                    key={c.name}
                    className="rounded-[10px] px-3 py-2.5 transition-colors hover:bg-lab-hover"
                  >
                    <div className="text-[13px] font-medium tracking-[-0.01em] text-lab-text">
                      {c.name}
                    </div>
                    <div className="mt-0.5 text-[11px] text-lab-muted">
                      <span
                        className={cn(
                          "font-medium",
                          String(c.status).toLowerCase().includes("up")
                            ? "text-lab-ok"
                            : "text-lab-muted",
                        )}
                      >
                        {c.status}
                      </span>
                      <span className="text-lab-muted/40"> · </span>
                      <span className="font-mono">{c.image}</span>
                    </div>
                  </div>
                ))}
            </div>
          </Panel>
        </div>

        <div className="bento-span-12">
          <Panel
            title="Recent runs"
            action={
              <Link
                href="/evals"
                className="text-[12px] font-medium text-lab-accent-bright transition-colors hover:text-lab-accent"
              >
                Open Evals →
              </Link>
            }
          >
            <div className="overflow-x-auto">
              <table className="lab-table">
                <thead>
                  <tr>
                    <th scope="col">Run</th>
                    <th scope="col">Kind</th>
                    <th scope="col">Intent</th>
                    <th scope="col">Model</th>
                  </tr>
                </thead>
                <tbody>
                  {!loading &&
                    runs.slice(0, 8).map((r) => {
                      const isTool =
                        r.kind?.includes("tool") || r.kind === "agentic_tool_eval";
                      const href = isTool ? `/evals/tool/${r.run_id}` : "/evals";
                      return (
                        <tr key={r.run_id} className="group">
                          <td>
                            <Link
                              href={href}
                              className="font-mono text-[11px] text-lab-accent-bright underline-offset-2 hover:underline"
                            >
                              {r.run_id}
                            </Link>
                          </td>
                          <td>
                            <span className="font-mono text-[11px] text-lab-muted">{r.kind}</span>
                          </td>
                          <td className="text-[12px]">{r.intent || "—"}</td>
                          <td className="max-w-[220px] truncate font-mono text-[12px]">
                            {r.model_id?.split("/").pop() || "—"}
                          </td>
                        </tr>
                      );
                    })}
                  {!loading && !runs.length && (
                    <tr>
                      <td colSpan={4}>
                        <EmptyState
                          title="No runs yet"
                          action={
                            <Link href="/evals" className={btnClass("secondary", "sm")}>
                              Open Evals
                            </Link>
                          }
                        >
                          Use Serve → Perf / Smoke when an endpoint is healthy.
                        </EmptyState>
                      </td>
                    </tr>
                  )}
                  {loading && (
                    <tr>
                      <td colSpan={4} className="!p-3">
                        <div className="space-y-2" aria-busy="true" aria-label="Loading runs">
                          {[0, 1, 2].map((i) => (
                            <div key={i} className="grid grid-cols-4 gap-3">
                              <Skeleton className="h-3 w-full" />
                              <Skeleton className="h-3 w-[66%]" />
                              <Skeleton className="h-3 w-[50%]" />
                              <Skeleton className="h-3 w-[75%]" />
                            </div>
                          ))}
                        </div>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
