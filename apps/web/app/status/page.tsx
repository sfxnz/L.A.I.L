"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type LabStatus, type RunRow } from "@/lib/api";
import { Badge, Btn, EmptyState, Metric, Panel, StatusDot, btnClass } from "@/components/ui";
import { cn } from "@/lib/utils";

function backendLabel(k: string) {
  if (k.toLowerCase() === "vllm") return "vLLM";
  if (k.toLowerCase() === "llamacpp") return "llama.cpp";
  return k;
}

export default function StatusPage() {
  const [status, setStatus] = useState<LabStatus | null>(null);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const refresh = () => {
    api
      .labStatus()
      .then(setStatus)
      .catch((e) => setErr(String(e.message || e)));
    api.runs().then(setRuns).catch(() => {});
  };

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 6000);
    return () => clearInterval(t);
  }, []);

  const serve = status?.serve;
  const healthy = !!(serve && !serve.unreachable && serve.healthy);
  const modelShort = serve?.model_id?.split("/").pop() || "—";
  const freeGib = serve?.hardware?.available_gib;

  return (
    <div className="space-y-6">
      <div className="page-header">
        <div>
          <div className="mb-2 flex items-center gap-2">
            <StatusDot live={healthy} />
            <span className="text-[11px] font-medium uppercase tracking-[0.08em] text-lab-muted">
              {healthy ? "Endpoint live" : "Endpoint down"}
            </span>
          </div>
          <h1 className="page-title">Status</h1>
          <p className="page-sub">
            Lab health, hardware headroom, containers, and recent runs.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Btn variant="secondary" size="sm" onClick={refresh}>
            Refresh
          </Btn>
          <Link href="/server" className={btnClass("primary", "sm")}>
            Serve
          </Link>
          <Link href="/evals" className={btnClass("secondary", "sm")}>
            Evals
          </Link>
          <Link href="/connect" className={btnClass("ghost", "sm")}>
            Connect
          </Link>
        </div>
      </div>

      {err && (
        <div className="rounded-[12px] border border-[rgba(255,69,58,0.28)] bg-[rgba(255,69,58,0.1)] px-3.5 py-2.5 text-[13px] text-lab-danger">
          {err}
        </div>
      )}

      <div className="bento">
        <div className="bento-span-3">
          <Metric
            label="vLLM endpoint"
            value={healthy ? "Healthy" : "Down"}
            sub={serve?.base_url || "—"}
            tone={healthy ? "ok" : "danger"}
            large
          />
        </div>
        <div className="bento-span-3">
          <Metric label="Model" value={modelShort} sub={serve?.model_id || "No model loaded"} />
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
          />
        </div>
        <div className="bento-span-3">
          <Metric
            label="Default backend"
            value={status?.defaultBackend === "vllm" ? "vLLM" : status?.defaultBackend || "—"}
            sub={status?.defaultModel || undefined}
          />
        </div>
      </div>

      <div className="bento">
        <div className="bento-span-6">
          <Panel
            className="flex h-full min-h-[220px] flex-col"
            title="Backends"
            action={
              <span className="text-[11px] tabular-nums text-lab-muted">
                {status ? Object.keys(status.backends || {}).length : 0} registered
              </span>
            }
          >
            <div className="flex flex-1 flex-col space-y-0.5 p-1.5">
              {status &&
                Object.entries(status.backends || {}).map(([k, v]) => (
                  <div
                    key={k}
                    className="flex items-center justify-between gap-3 rounded-[10px] px-3 py-2.5 transition-colors hover:bg-lab-hover"
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <StatusDot live={!!v.ok} />
                        <span className="text-[13px] font-medium tracking-[-0.01em] text-lab-text">
                          {backendLabel(k)}
                        </span>
                      </div>
                      <div className="mt-0.5 truncate pl-[15px] font-mono text-[11px] text-lab-muted">
                        {v.url}
                      </div>
                    </div>
                    <Badge tone={v.ok ? "ok" : "danger"}>{v.ok ? "up" : "down"}</Badge>
                  </div>
                ))}
              {!status && <EmptyState>Loading backends…</EmptyState>}
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
              {(serve?.containers || []).length === 0 && (
                <div className="flex flex-1 items-center justify-center">
                  <EmptyState title="No containers">
                    Nothing running — open Serve to load a model.
                  </EmptyState>
                </div>
              )}
              {(serve?.containers || []).map((c) => (
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
                    <th>Run</th>
                    <th>Kind</th>
                    <th>Intent</th>
                    <th>Model</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.slice(0, 8).map((r) => (
                    <tr key={r.run_id}>
                      <td className="font-mono text-[11px] text-lab-text-dim">{r.run_id}</td>
                      <td>
                        <span className="font-mono text-[11px] text-lab-muted">{r.kind}</span>
                      </td>
                      <td className="text-[12px]">{r.intent || "—"}</td>
                      <td className="max-w-[220px] truncate font-mono text-[12px]">
                        {r.model_id?.split("/").pop() || "—"}
                      </td>
                    </tr>
                  ))}
                  {!runs.length && (
                    <tr>
                      <td colSpan={4}>
                        <EmptyState title="No runs yet">
                          Use Serve → Perf / Smoke when an endpoint is healthy.
                        </EmptyState>
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
