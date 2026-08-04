"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { api, type LabArtifactRun } from "@/lib/api";
import { Badge, Btn, Panel, btnClass } from "@/components/ui";

export default function LabRunDetailPage() {
  const params = useParams();
  const id = String(params?.id || "");
  const [run, setRun] = useState<(LabArtifactRun & { files?: string[]; siblings?: LabArtifactRun[] }) | null>(
    null,
  );
  const [err, setErr] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const reload = () => {
    if (!id) return;
    api.lab
      .get(id)
      .then(setRun)
      .catch((e) => setErr(String(e.message || e)));
  };

  useEffect(() => {
    reload();
  }, [id]);

  const origin = typeof window !== "undefined" ? window.location.origin : "";
  const shareUrl = `${origin}/lab/${id}`;
  const playUrl = run?.play_url || `/api/lab/runs/${id}/play`;
  const publicUrl = run?.public_url ? `${origin}${run.public_url}` : null;

  async function copy(label: string, text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(label);
      setTimeout(() => setCopied(null), 1500);
    } catch {
      /* */
    }
  }

  async function togglePublic() {
    if (!run) return;
    setBusy(true);
    try {
      const next = !(run.share?.public);
      const updated = await api.lab.share(run.id, next);
      setRun({ ...run, ...updated });
    } catch (e) {
      setErr(String((e as Error).message || e));
    } finally {
      setBusy(false);
    }
  }

  if (err && !run) {
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

  const siblingIds = [run.id, ...(run.siblings || []).map((s) => s.id)].slice(0, 4);

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
          {run.share?.public && <Badge tone="ok">public</Badge>}
          <a className={btnClass("primary")} href={playUrl} target="_blank" rel="noreferrer">
            Open fullscreen
          </a>
          {(run.siblings?.length || 0) > 0 && (
            <Link
              className={btnClass("secondary")}
              href={`/lab/compare?ids=${siblingIds.join(",")}`}
            >
              Compare siblings
            </Link>
          )}
          <Btn variant="ghost" onClick={() => copy("link", shareUrl)}>
            {copied === "link" ? "Copied" : "Copy gallery link"}
          </Btn>
        </div>
      </div>

      {err && (
        <div className="rounded-xl border border-lab-danger/30 bg-lab-danger/10 px-3 py-2 text-sm text-lab-danger">
          {err}
        </div>
      )}

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
                <div className="break-all font-mono text-xs text-lab-text">{run.model_id}</div>
              </div>
              {run.task_fingerprint && (
                <div>
                  <div className="text-[11px] uppercase tracking-wide text-lab-muted">
                    Task fingerprint
                  </div>
                  <Link
                    className="font-mono text-xs text-lab-accent hover:underline"
                    href={`/lab/compare?fingerprint=${run.task_fingerprint}`}
                  >
                    {run.task_fingerprint}
                  </Link>
                </div>
              )}
              {run.hermes?.source && (
                <div>
                  <div className="text-[11px] uppercase tracking-wide text-lab-muted">Source</div>
                  <div>{run.hermes.source}</div>
                </div>
              )}
              <div className="flex flex-wrap gap-1 pt-1">
                {(run.tags || []).map((t: string) => (
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

          <Panel title="Share for X / internet">
            <p className="mb-2 text-xs text-lab-muted">
              Serves <strong className="text-lab-text">only game files</strong> — not L.A.I.L, not Hermes.
              Secret-like strings blocked at publish. CSP locks down the page.
            </p>
            <p className="mb-2 text-xs text-lab-muted">
              <strong className="text-lab-text">For people on X</strong> you need Tailscale Funnel on the
              artifacts-only server (port 8791), not the full lab:
            </p>
            <pre className="mb-2 overflow-x-auto rounded-lg bg-black/40 p-2 text-[11px] text-lab-muted">
{`cd ~/projects/ai-lab/local-ai-lab
bun run lab:funnel
# then restart: bun run dev
# links become https://spark1.<tailnet>.ts.net/s/<slug>/index.html`}
            </pre>
            <p className="mb-2 text-xs text-lab-muted">
              Without Funnel, Create share link only works on your Tailscale/LAN (fine for you, not for
              random people on X).
            </p>
            <Btn variant="secondary" disabled={busy} onClick={togglePublic}>
              {run.share?.public ? "Unpublish" : "Create share link"}
            </Btn>
            {publicUrl && (
              <div className="mt-3 space-y-2">
                <code className="block break-all rounded-lg bg-black/40 p-2 text-[11px] text-lab-text">
                  {publicUrl}
                </code>
                <Btn variant="ghost" onClick={() => copy("pub", publicUrl)}>
                  {copied === "pub" ? "Copied" : "Copy share URL"}
                </Btn>
              </div>
            )}
          </Panel>

          {(run.siblings?.length || 0) > 0 && (
            <Panel title="Same brief (other models)">
              <ul className="space-y-1 text-sm">
                {run.siblings!.map((s) => (
                  <li key={s.id}>
                    <Link className="text-lab-accent hover:underline" href={`/lab/${s.id}`}>
                      {(s.model_id || s.id).split("/").pop()}
                    </Link>
                  </li>
                ))}
              </ul>
            </Panel>
          )}

          <Panel title="Files">
            <ul className="space-y-1 font-mono text-xs text-lab-muted">
              {(run.files || []).map((f: string) => (
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
              {!run.files?.length && <li>—</li>}
            </ul>
          </Panel>
        </div>
      </div>
    </div>
  );
}
