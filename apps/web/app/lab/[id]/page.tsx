"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { api, type LabArtifactRun } from "@/lib/api";
import { Badge, Btn, Panel, btnClass } from "@/components/ui";

export default function LabRunDetailPage() {
  const params = useParams();
  const id = String(params?.id || "");
  const [run, setRun] = useState<(LabArtifactRun & { files?: string[] }) | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!id) return;
    api.lab
      .get(id)
      .then(setRun)
      .catch((e) => setErr(String(e.message || e)));
  }, [id]);

  const shareUrl =
    typeof window !== "undefined" ? `${window.location.origin}/lab/${id}` : `/lab/${id}`;
  const playUrl = run?.play_url || `/api/lab/runs/${id}/play`;

  async function copyShare() {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* */
    }
  }

  if (err) {
    return (
      <div className="space-y-4">
        <Link href="/lab" className="text-sm text-lab-accent">
          ← Lab
        </Link>
        <p className="text-lab-danger">{err}</p>
      </div>
    );
  }

  if (!run) {
    return <div className="text-sm text-lab-muted">Loading…</div>;
  }

  return (
    <div className="space-y-5">
      <div className="page-header">
        <div>
          <Link href="/lab" className="text-xs font-medium text-lab-muted hover:text-lab-accent">
            ← Lab gallery
          </Link>
          <h1 className="page-title mt-1">{run.title}</h1>
          <p className="page-sub">
            {(run.model_id || "unknown").split("/").pop()} · {run.task_type} ·{" "}
            {run.created_at ? new Date(run.created_at).toLocaleString() : run.id}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="ok">{run.task_type}</Badge>
          <a className={btnClass("primary")} href={playUrl} target="_blank" rel="noreferrer">
            Open fullscreen
          </a>
          <Btn variant="ghost" onClick={copyShare}>
            {copied ? "Copied" : "Copy link"}
          </Btn>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <div className="overflow-hidden rounded-2xl border border-lab-border bg-black">
            <iframe
              title={run.title}
              src={playUrl}
              className="h-[min(70vh,640px)] w-full"
              sandbox="allow-scripts allow-same-origin"
            />
          </div>
        </div>
        <div className="space-y-4">
          <Panel title="Model">
            <div className="space-y-2 text-sm">
              <div>
                <div className="text-[11px] uppercase tracking-wide text-lab-muted">Model id</div>
                <div className="font-mono text-xs text-lab-text break-all">{run.model_id}</div>
              </div>
              {run.hermes?.source && (
                <div>
                  <div className="text-[11px] uppercase tracking-wide text-lab-muted">Source</div>
                  <div>{run.hermes.source}</div>
                </div>
              )}
              {run.eval_run_id && (
                <div>
                  <div className="text-[11px] uppercase tracking-wide text-lab-muted">Eval run</div>
                  <Link
                    className="text-lab-accent hover:underline"
                    href={`/evals/tool/${run.eval_run_id}`}
                  >
                    {run.eval_run_id}
                  </Link>
                </div>
              )}
              <div className="flex flex-wrap gap-1 pt-1">
                {(run.tags || []).map((t) => (
                  <span
                    key={t}
                    className="rounded-md bg-white/5 px-1.5 py-0.5 text-[10px] uppercase text-lab-muted"
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>
          </Panel>

          {run.brief && (
            <Panel title="Brief">
              <pre className="whitespace-pre-wrap text-xs text-lab-muted">{run.brief}</pre>
            </Panel>
          )}

          <Panel title="Files">
            <ul className="space-y-1 text-xs font-mono text-lab-muted">
              {(run.files || []).map((f) => (
                <li key={f}>
                  <a
                    className="hover:text-lab-accent"
                    href={`/api/lab/runs/${run.id}/files/artifacts/${f}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {f}
                  </a>
                </li>
              ))}
              {!run.files?.length && <li className="text-lab-muted">—</li>}
            </ul>
          </Panel>

          <Panel title="Share">
            <p className="mb-2 text-xs text-lab-muted">
              Anyone on your Tailscale can open this page and play the artifact. Public internet
              share comes later (static export only).
            </p>
            <code className="block break-all rounded-lg bg-black/40 p-2 text-[11px] text-lab-text">
              {shareUrl}
            </code>
          </Panel>
        </div>
      </div>
    </div>
  );
}
