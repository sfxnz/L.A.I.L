"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type LabStatus, type RunRow } from "@/lib/api";
import { Badge, Btn, Metric, Panel } from "@/components/ui";

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
  const healthy = !!serve && !serve.unreachable && serve.healthy;

  return (
    <div className="space-y-4 p-4 md:p-5">
      <div className="page-header">
        <div>
          <h1 className="page-title">Status</h1>
          <p className="page-sub">Lab health, backends, hardware, recent activity</p>
        </div>
        <div className="flex gap-1.5">
          <Btn variant="secondary" size="sm" onClick={refresh}>
            Refresh
          </Btn>
          <Link href="/server">
            <Btn size="sm">Open Server</Btn>
          </Link>
        </div>
      </div>

      {err && (
        <div className="rounded-md border border-lab-danger/30 bg-lab-danger/10 px-3 py-2 text-xs text-lab-danger">
          {err}
        </div>
      )}

      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        <Metric
          label="vLLM endpoint"
          value={healthy ? "HEALTHY" : "DOWN"}
          sub={serve?.base_url || "—"}
          accent={healthy}
        />
        <Metric
          label="Model"
          value={serve?.model_id?.split("/").pop() || "—"}
          sub={serve?.model_id || ""}
        />
        <Metric
          label="Memory free"
          value={
            serve?.hardware?.available_gib != null
              ? `${serve.hardware.available_gib} GiB`
              : "—"
          }
          sub={serve?.hardware?.gpu_sku || serve?.hardware?.cpu || ""}
        />
        <Metric
          label="Default backend"
          value={status?.defaultBackend || "—"}
          sub={status?.defaultModel}
        />
      </div>

      <div className="grid gap-2 lg:grid-cols-2">
        <Panel className="p-3" title="Backends">
          <div className="space-y-1.5 p-1">
            {status &&
              Object.entries(status.backends || {}).map(([k, v]) => (
                <div
                  key={k}
                  className="flex items-center justify-between rounded-md border border-lab-border-subtle bg-lab-bg/40 px-2.5 py-1.5"
                >
                  <div>
                    <div className="text-[12px] capitalize text-lab-text">{k}</div>
                    <div className="font-mono text-[10px] text-lab-muted">{v.url}</div>
                  </div>
                  <Badge tone={v.ok ? "ok" : "danger"}>{v.ok ? "up" : "down"}</Badge>
                </div>
              ))}
          </div>
        </Panel>

        <Panel className="p-3" title="Containers">
          <div className="space-y-1.5 p-1">
            {(serve?.containers || []).length === 0 && (
              <div className="text-[12px] text-lab-muted">No vLLM containers</div>
            )}
            {(serve?.containers || []).map((c) => (
              <div
                key={c.name}
                className="rounded-md border border-lab-border-subtle bg-lab-bg/40 px-2.5 py-1.5"
              >
                <div className="text-[12px] font-medium text-lab-text">{c.name}</div>
                <div className="text-[11px] text-lab-muted">
                  {c.status} · {c.image}
                </div>
              </div>
            ))}
            {serve?.headroom && (
              <div className="pt-1 text-[11px] text-lab-muted">
                Headroom:{" "}
                <Badge
                  tone={
                    serve.headroom === "critical"
                      ? "danger"
                      : serve.headroom === "tight"
                        ? "warn"
                        : "ok"
                  }
                >
                  {serve.headroom}
                </Badge>
              </div>
            )}
          </div>
        </Panel>
      </div>

      <Panel title="Recent runs">
        <div className="overflow-x-auto p-2">
          <div className="mb-1 flex justify-end px-1">
            <Link href="/server" className="text-[11px] text-lab-accent-bright hover:underline">
              Bench on Server →
            </Link>
          </div>
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
                  <td className="font-mono text-[11px]">{r.run_id}</td>
                  <td>{r.kind}</td>
                  <td>{r.intent || "—"}</td>
                  <td className="max-w-[200px] truncate">{r.model_id?.split("/").pop() || "—"}</td>
                </tr>
              ))}
              {!runs.length && (
                <tr>
                  <td colSpan={4} className="text-lab-muted">
                    No runs yet — use Server → Perf / Smoke
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
