"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { api, type LabArtifactRun } from "@/lib/api";
import { Badge, Btn, EmptyState, Panel } from "@/components/ui";

export default function LabGalleryPage() {
  const [runs, setRuns] = useState<LabArtifactRun[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [task, setTask] = useState("");

  const refresh = () => {
    api.lab
      .list({ limit: 80, task_type: task || undefined })
      .then((r) => setRuns(r.runs || []))
      .catch((e) => setErr(String(e.message || e)));
  };

  useEffect(() => {
    refresh();
  }, [task]);

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return runs;
    return runs.filter(
      (r) =>
        r.title.toLowerCase().includes(q) ||
        r.model_id.toLowerCase().includes(q) ||
        r.tags?.some((t) => t.toLowerCase().includes(q)),
    );
  }, [runs, filter]);

  const taskTypes = useMemo(() => {
    const s = new Set(runs.map((r) => r.task_type).filter(Boolean));
    return [...s].sort();
  }, [runs]);

  return (
    <div className="space-y-5">
      <div className="page-header">
        <div>
          <h1 className="page-title">Lab</h1>
          <p className="page-sub">
            What local models actually built — HTML games, animations, mini-apps. Serve → eval → Hermes
            task → live here.
          </p>
        </div>
        <Badge tone="ok">{runs.length} runs</Badge>
      </div>

      {err && (
        <div className="rounded-xl border border-lab-danger/30 bg-lab-danger/10 px-3 py-2 text-sm text-lab-danger">
          {err}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <input
          className="inputCls min-w-[200px] flex-1 rounded-lg border border-lab-border bg-lab-surface px-3 py-2 text-sm"
          placeholder="Filter title, model, tags…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        <select
          className="rounded-lg border border-lab-border bg-lab-surface px-3 py-2 text-sm"
          value={task}
          onChange={(e) => setTask(e.target.value)}
        >
          <option value="">All tasks</option>
          {taskTypes.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        <Btn variant="ghost" onClick={refresh}>
          Refresh
        </Btn>
      </div>

      {!filtered.length ? (
        <Panel title="No lab runs yet">
          <EmptyState title="Gallery is empty">
            <p className="mb-3 text-sm text-lab-muted">
              Import a Hermes-built HTML file (or any artifact folder) into the gallery.
            </p>
            <pre className="overflow-x-auto rounded-lg bg-black/40 p-3 text-xs text-lab-muted">
{`curl -s http://127.0.0.1:8787/api/lab/runs/import \\
  -H 'Content-Type: application/json' \\
  -d '{
    "title": "My game",
    "task_type": "html-game",
    "model_id": "nvidia/Qwen3.6-27B-NVFP4",
    "from": "workspaces/demo/geometry-dash-like.html",
    "tags": ["html","game"]
  }'`}
            </pre>
          </EmptyState>
        </Panel>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((r) => (
            <Link
              key={r.id}
              href={`/lab/${r.id}`}
              className="group flex flex-col overflow-hidden rounded-2xl border border-lab-border bg-lab-surface transition hover:border-lab-accent/40 hover:shadow-[0_0_0_1px_rgba(10,132,255,0.15)]"
            >
              <div className="relative aspect-[16/10] bg-black/50">
                <iframe
                  title={r.title}
                  src={r.play_url}
                  className="pointer-events-none h-full w-full scale-[0.5] origin-top-left"
                  style={{ width: "200%", height: "200%" }}
                  sandbox="allow-scripts"
                  loading="lazy"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-lab-surface via-transparent to-transparent opacity-80" />
              </div>
              <div className="flex flex-1 flex-col gap-2 p-3">
                <div className="text-[15px] font-semibold tracking-tight text-lab-text group-hover:text-lab-accent">
                  {r.title}
                </div>
                <div className="text-xs text-lab-muted">
                  {(r.model_id || "unknown").split("/").pop()} · {r.task_type}
                </div>
                <div className="mt-auto flex flex-wrap gap-1">
                  {(r.tags || []).slice(0, 4).map((t) => (
                    <span
                      key={t}
                      className="rounded-md bg-white/5 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-lab-muted"
                    >
                      {t}
                    </span>
                  ))}
                </div>
                <div className="text-[10px] text-lab-muted/80">
                  {r.created_at ? new Date(r.created_at).toLocaleString() : r.id}
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}

      <Panel title="How this fits">
        <ol className="list-decimal space-y-1 pl-5 text-sm text-lab-muted">
          <li>
            <strong className="text-lab-text">Serve</strong> a model on Spark
          </li>
          <li>
            <strong className="text-lab-text">Evals</strong> — tool quality board
          </li>
          <li>
            <strong className="text-lab-text">Hermes</strong> builds a self-contained HTML game /
            animation (Connect tab for base URL)
          </li>
          <li>
            <strong className="text-lab-text">Import</strong> into Lab — open anytime, compare models,
            share play link on Tailscale
          </li>
        </ol>
      </Panel>
    </div>
  );
}
