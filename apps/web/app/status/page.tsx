"use client";

import Link from "next/link";
import { useCallback, useEffect, useState, type ReactNode } from "react";
import { api, type RunRow } from "@/lib/api";
import { serveHealthy, useLabStatus } from "@/lib/lab-status-store";
import { ClusterPanel } from "@/components/ClusterPanel";
import { LastSyncCard } from "@/components/bench/LastSyncCard";
import {
  Badge,
  Btn,
  Callout,
  Corridor,
  EmptyState,
  Eyebrow,
  Metric,
  Nil,
  Panel,
  Skeleton,
  SyncRing,
  Tick,
  btnClass,
  eyebrowClass,
} from "@/components/ui";
import { cn } from "@/lib/utils";

/*
  Status — the console's front page, read as ONE instrument cluster.

  Structure: three numbered bands (fabric → endpoint → runtime), each opened by
  an eyebrow that runs into a hairline .animus-rule. The rules are the path down
  the page; without them the panels fall back into a generic card grid.

  NULL TREATMENT (one rule, this file + ClusterPanel): an absent value is a
  condensed ALL-CAPS STATE WORD, never an em-dash — an em-dash reads as broken
  data, a word reads as the instrument telling you where it is. Two words cover
  every case on this surface:
    AWAITING · a live readout that hasn't reported yet
    NONE     · a settled state that genuinely has no value
  Where the markup is ours, <Nil/> (components/ui) renders it muted at eyebrow
  scale behind a hollow diamond glyph. Where a primitive owns the slot —
  Metric.value is typed `string`, so no JSX — the same word goes in as the
  value; all-caps in a field of mixed-case readings ("Healthy", "42 GiB") is
  what marks it as a state token rather than data.
*/

const NIL_AWAITING = "AWAITING";
const NIL_NONE = "NONE";

/** Band opener: index + label, then a hairline running to the right margin. */
function Band({
  index,
  label,
  meta,
}: {
  index: string;
  label: string;
  meta?: ReactNode;
}) {
  return (
    <div className="flex items-center gap-2.5">
      <span className="animus-eyebrow shrink-0 tabular-nums text-lab-line!">{index}</span>
      <span className="animus-eyebrow shrink-0 text-lab-text-dim!">{label}</span>
      <div aria-hidden className="animus-rule min-w-6 flex-1" />
      {meta ? <Eyebrow className="lab-num shrink-0">{meta}</Eyebrow> : null}
    </div>
  );
}

function backendLabel(k: string) {
  if (k.toLowerCase() === "vllm") return "vLLM";
  if (k.toLowerCase() === "llamacpp") return "llama.cpp";
  return k;
}

export default function StatusPage() {
  const { status, loading, needToken, unreachable, error, refresh: refreshStatus } = useLabStatus();
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const err = unreachable ? error : null;

  // Lab status comes from the shell's shared poll; runs load on mount and
  // re-load when a bench settles or the operator asks.
  const loadRuns = useCallback(
    () =>
      api
        .runs()
        .then(setRuns)
        .catch(() => {}),
    [],
  );
  useEffect(() => {
    void loadRuns();
  }, [loadRuns]);

  const refresh = useCallback(
    async (opts?: { soft?: boolean }) => {
      if (!opts?.soft) setRefreshing(true);
      try {
        await Promise.all([refreshStatus(), loadRuns()]);
      } finally {
        setRefreshing(false);
      }
    },
    [refreshStatus, loadRuns],
  );

  const serve = status?.serve;
  const cluster = status?.cluster || serve?.cluster || null;
  const healthy = serveHealthy(status);
  const modelShort = serve?.model_id?.split("/").pop() || NIL_NONE;
  const freeGib = serve?.hardware?.available_gib;
  const clusterHealthy = !!cluster?.summary?.healthy;
  const multiMode = cluster?.summary?.multi?.mode;
  const backendCount = status ? Object.keys(status.backends || {}).length : null;
  const containers = serve?.containers || [];
  const nodeCount = cluster?.summary?.nodes_total ?? cluster?.nodes?.length ?? 0;
  const fabricMeta = loading ? "probing" : nodeCount >= 2 ? `${nodeCount} nodes` : "this host";

  return (
    <div className="lab-fade-in space-y-4">
      <div className="page-header">
        <div className="min-w-0">
          <div
            className="mb-2.5 flex flex-wrap items-center gap-2.5"
            aria-live="polite"
          >
            <div className="flex items-center gap-2">
              <SyncRing
                state={
                  needToken
                    ? "token"
                    : err
                      ? "offline"
                      : loading
                        ? null
                        : healthy
                          ? "serving"
                          : "idle"
                }
                label={
                  loading
                    ? "Checking endpoint"
                    : needToken
                      ? "Token required"
                      : healthy
                        ? "Endpoint live"
                        : err
                          ? "Controller unreachable"
                          : "No model serving"
                }
              />
              <Eyebrow
                className={cn(
                  "tracking-[0.18em]",
                  loading || needToken
                    ? "text-lab-muted"
                    : healthy
                      ? "text-lab-ok"
                      : err
                        ? "text-lab-danger"
                        : "text-lab-muted",
                )}
              >
                {loading
                  ? "Checking…"
                  : needToken
                    ? "Token required"
                    : healthy
                      ? "Endpoint live"
                      : err
                        ? "Controller unreachable"
                        : "No model serving"}
              </Eyebrow>
            </div>
            {!loading && cluster && (
              <>
                <Tick />
                <div className="flex items-center gap-2">
                  <SyncRing
                    state={clusterHealthy ? "serving" : "offline"}
                    label={clusterHealthy ? "Cluster fabric ok" : "Cluster issue"}
                  />
                  <Eyebrow
                    className={cn(
                      "tracking-[0.18em]",
                      clusterHealthy ? "text-lab-text-dim" : "text-lab-danger",
                    )}
                  >
                    {clusterHealthy ? "Cluster fabric ok" : "Cluster issue"}
                    {multiMode && multiMode !== "none"
                      ? ` · ${multiMode.replace(/_/g, " ")}`
                      : ""}
                  </Eyebrow>
                </div>
              </>
            )}
          </div>
          <h1 className="page-title">Status</h1>
          <p className="page-sub">
            Sparks, live instruments, and the last synchronization on this page.
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
            className="px-1 font-[family-name:var(--font-display)] text-[11px] font-semibold uppercase leading-none tracking-[0.14em] text-lab-muted transition-colors hover:text-lab-text"
          >
            Tool eval →
          </Link>
        </div>
      </div>

      {needToken && (
        <Callout tone="warn" title="LAIL_TOKEN required — paste it to synchronize">
          The controller is up. Paste the token in the banner — it stays in sessionStorage and is
          sent as <code className="text-lab-text">X-Lail-Token</code>.
        </Callout>
      )}

      {err && (
        <Callout
          tone="danger"
          title="Controller unreachable on :8787"
          action={
            <Btn variant="secondary" size="sm" onClick={() => void refresh()} loading={refreshing}>
              Retry
            </Btn>
          }
        >
          {err}. Check that <code className="text-lab-text">bun run dev</code> is up on this host
          (ports 3000 / 8787 / 8765).
        </Callout>
      )}

      {!loading && !healthy && !err && !needToken && (
        <Panel>
          <Corridor
            action={
              <Link href="/server" className={btnClass("primary", "sm")}>
                Serve a model
              </Link>
            }
          >
            No memory loaded. Serve a model to begin synchronization — cluster fabric can still be
            fine with nothing loaded.
          </Corridor>
        </Panel>
      )}

      <section className="space-y-2.5">
        <Band index="01" label="Fabric" meta={fabricMeta} />
        <div className="lab-rise">
          {needToken ? (
            <Panel title="Cluster" padded>
              <EmptyState title="Token required">
                Paste LAIL_TOKEN in the banner to load Spark instruments.
              </EmptyState>
            </Panel>
          ) : (
            <ClusterPanel cluster={cluster} loading={loading} />
          )}
        </div>
      </section>

      <section className="space-y-2.5">
        <Band
          index="02"
          label="Bench"
          meta={healthy ? "armed" : loading ? "probing" : "locked"}
        />
        <div className="lab-rise lab-rise-1">
          <LastSyncCard runs={runs} loading={loading} />
        </div>
      </section>

      <section className="space-y-2.5">
        <Band index="03" label="Endpoint" meta={loading ? "probing" : "2s poll"} />
        <div className="bento lab-rise lab-rise-1">
          <div className="bento-span-3">
            <Metric
              label="vLLM endpoint"
              value={loading ? NIL_AWAITING : healthy ? "Healthy" : "Idle"}
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
              tone={serve?.model_id ? undefined : "muted"}
              loading={loading}
            />
          </div>
          <div className="bento-span-3">
            <Metric
              label="Memory free"
              value={freeGib != null ? `${freeGib} GiB` : NIL_AWAITING}
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
              tone={freeGib == null ? "muted" : freeGib < 15 ? "warn" : undefined}
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
                status?.defaultBackend === "vllm" ? "vLLM" : status?.defaultBackend || NIL_NONE
              }
              sub={status?.defaultModel || undefined}
              tone={status?.defaultBackend ? undefined : "muted"}
              loading={loading}
            />
          </div>
        </div>
      </section>

      <section className="space-y-2.5">
        <Band
          index="04"
          label="Runtime"
          meta={
            loading
              ? "probing"
              : `${backendCount ?? 0} backend${backendCount === 1 ? "" : "s"} · ${containers.length} container${containers.length === 1 ? "" : "s"}`
          }
        />
        <div className="bento lab-rise lab-rise-2">
          <div className="bento-span-6">
            <Panel
              className="flex h-full min-h-[212px] flex-col"
              title="Backends"
              action={
                <Eyebrow className="lab-num">
                  {backendCount != null ? `${backendCount} registered` : <Nil />}
                </Eyebrow>
              }
            >
              <div className="flex h-full flex-1 flex-col">
                {loading && (
                  <div className="p-2" aria-busy="true" aria-label="Loading backends">
                    {[0, 1].map((i) => (
                      <div
                        key={i}
                        className="flex items-center justify-between gap-3 border-b border-lab-border-subtle px-2 py-3 last:border-b-0"
                      >
                        <div className="min-w-0 flex-1 space-y-2">
                          <Skeleton className="h-3 w-24" />
                          <Skeleton className="h-2.5 w-40" />
                        </div>
                        <Skeleton className="h-4 w-12" />
                      </div>
                    ))}
                  </div>
                )}
                {!loading &&
                  status &&
                  Object.entries(status.backends || {}).map(([k, v]) => (
                    <div
                      key={k}
                      className="animus-notch flex items-center justify-between gap-3 border-b border-lab-border-subtle px-4 py-3 transition-[background,box-shadow] last:border-b-0 hover:bg-[color:var(--animus-accent-wash)] hover:shadow-[inset_2px_0_0_var(--color-lab-accent)]"
                    >
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <SyncRing
                            state={v.ok ? "serving" : "idle"}
                            label={`${backendLabel(k)} ${v.ok ? "up" : "idle"}`}
                          />
                          <span className="font-[family-name:var(--font-display)] text-[13px] font-semibold uppercase leading-none tracking-[0.14em] text-lab-text">
                            {backendLabel(k)}
                          </span>
                        </div>
                        <div className="mt-1.5 truncate pl-[15px] font-mono text-[10px] text-lab-muted">
                          {v.url}
                        </div>
                      </div>
                      <Badge tone={v.ok ? "ok" : "muted"}>{v.ok ? "up" : "idle"}</Badge>
                    </div>
                  ))}
                {!loading && status && backendCount === 0 && (
                  <div className="flex flex-1 items-center justify-center">
                    <EmptyState title="No backends registered">
                      Check Configure for vLLM / llama.cpp URLs.
                    </EmptyState>
                  </div>
                )}
              </div>
            </Panel>
          </div>

          <div className="bento-span-6">
            <Panel
              className="flex h-full min-h-[212px] flex-col"
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
              <div className="flex h-full flex-1 flex-col">
                {loading && (
                  <div className="p-2" aria-busy="true" aria-label="Loading containers">
                    {[0, 1].map((i) => (
                      <div
                        key={i}
                        className="space-y-2 border-b border-lab-border-subtle px-2 py-3 last:border-b-0"
                      >
                        <Skeleton className="h-3 w-32" />
                        <Skeleton className="h-2.5 w-48" />
                      </div>
                    ))}
                  </div>
                )}
                {!loading && containers.length === 0 && (
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
                  containers.map((c) => {
                    const up = String(c.status).toLowerCase().includes("up");
                    return (
                      <div
                        key={c.name}
                        className="animus-notch border-b border-lab-border-subtle px-4 py-3 transition-[background,box-shadow] last:border-b-0 hover:bg-[color:var(--animus-accent-wash)] hover:shadow-[inset_2px_0_0_var(--color-lab-accent)]"
                      >
                        <div className="flex items-center gap-2">
                          <SyncRing state={up ? "serving" : "idle"} label={`${c.name} ${c.status}`} />
                          <span className="truncate font-[family-name:var(--font-display)] text-[13px] font-semibold uppercase leading-none tracking-[0.14em] text-lab-text">
                            {c.name}
                          </span>
                        </div>
                        <div className="mt-1.5 flex items-center gap-2 pl-[15px] text-[10px]">
                          <span
                            className={cn(
                              "shrink-0 font-[family-name:var(--font-display)] font-semibold uppercase leading-none tracking-[0.14em]",
                              up ? "text-lab-ok" : "text-lab-muted",
                            )}
                          >
                            {c.status}
                          </span>
                          <Tick className="h-2.5" />
                          <span className="truncate font-mono text-lab-muted">{c.image}</span>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </Panel>
          </div>

          <div className="bento-span-12">
            <Panel
              title="Recent runs"
              action={
                <Link
                  href="/evals"
                  className={eyebrowClass("text-lab-accent-bright transition-colors hover:text-lab-accent")}
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
                        const model = r.model_id?.split("/").pop();
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
                            <td className="text-[12px]">{r.intent || <Nil word="None" />}</td>
                            <td className="max-w-[220px] truncate font-mono text-[12px]">
                              {model || <Nil word="None" />}
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
                            Run a decode bench above when an endpoint is healthy.
                          </EmptyState>
                        </td>
                      </tr>
                    )}
                    {loading && (
                      <tr>
                        <td colSpan={4} className="!p-3">
                          <div className="space-y-2.5" aria-busy="true" aria-label="Loading runs">
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
      </section>
    </div>
  );
}
