"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import { api, type LabArtifactRun } from "@/lib/api";
import { Badge, Btn, Panel } from "@/components/ui";

function CompareInner() {
  const sp = useSearchParams();
  const idsParam = sp.get("ids") || "";
  const fp = sp.get("fingerprint") || "";
  const [runs, setRuns] = useState<LabArtifactRun[]>([]);
  const [brief, setBrief] = useState<string | null>(null);
  const [same, setSame] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<string[]>([]);

  useEffect(() => {
    const load = async () => {
      try {
        if (idsParam) {
          const ids = idsParam.split(",").map((s) => s.trim()).filter(Boolean);
          const c = await api.lab.compare(ids);
          setRuns(c.runs);
          setBrief(c.brief);
          setSame(c.same_brief);
          setSelected(c.runs.map((r) => r.id));
        } else if (fp) {
          const list = await api.lab.list({ fingerprint: fp, limit: 8 });
          setRuns(list.runs);
          setBrief(list.runs[0]?.brief || null);
          setSame(true);
          setSelected(list.runs.slice(0, 3).map((r) => r.id));
        }
      } catch (e) {
        setErr(String((e as Error).message || e));
      }
    };
    load();
  }, [idsParam, fp]);

  const shown = useMemo(
    () => runs.filter((r) => selected.includes(r.id)).slice(0, 4),
    [runs, selected],
  );

  function toggle(id: string) {
    setSelected((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 4) return prev;
      return [...prev, id];
    });
  }

  if (err) return <p className="text-lab-danger">{err}</p>;

  return (
    <div className="space-y-5">
      <div className="page-header">
        <div>
          <Link href="/lab" className="text-xs font-medium text-lab-muted hover:text-lab-accent">
            ← Lab
          </Link>
          <h1 className="page-title mt-1">Compare</h1>
          <p className="page-sub">
            Side-by-side playable outputs · same brief across models
            {!same && " · warning: fingerprints differ"}
          </p>
        </div>
        <Badge tone={same ? "ok" : "warn"}>{shown.length} panes</Badge>
      </div>

      {brief && (
        <Panel title="Brief">
          <pre className="whitespace-pre-wrap text-xs text-lab-muted">{brief}</pre>
        </Panel>
      )}

      {runs.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {runs.map((r) => {
            const on = selected.includes(r.id);
            return (
              <button
                key={r.id}
                type="button"
                onClick={() => toggle(r.id)}
                className={`rounded-lg border px-2.5 py-1.5 text-xs ${
                  on
                    ? "border-lab-accent bg-lab-accent/15 text-lab-text"
                    : "border-lab-border text-lab-muted"
                }`}
              >
                {(r.model_id || "?").split("/").pop()}
              </button>
            );
          })}
        </div>
      )}

      {!shown.length ? (
        <Panel title="Nothing to compare">
          <p className="text-sm text-lab-muted">
            Open from a lab run’s “Compare siblings”, or pass{" "}
            <code className="text-lab-text">?ids=a,b</code> /{" "}
            <code className="text-lab-text">?fingerprint=…</code>
          </p>
        </Panel>
      ) : (
        <div
          className={`grid gap-3 ${
            shown.length === 1
              ? "grid-cols-1"
              : shown.length === 2
                ? "md:grid-cols-2"
                : "md:grid-cols-2 xl:grid-cols-3"
          }`}
        >
          {shown.map((r) => (
            <div
              key={r.id}
              className="flex min-h-0 flex-col overflow-hidden rounded-2xl border border-lab-border bg-lab-surface"
            >
              <div className="flex items-center justify-between gap-2 border-b border-lab-border-subtle px-3 py-2">
                <div className="min-w-0">
                  <div className="truncate text-sm font-semibold">{r.title}</div>
                  <div className="truncate text-[11px] text-lab-muted">
                    {(r.model_id || "unknown").split("/").pop()}
                  </div>
                </div>
                <Link href={`/lab/${r.id}`} className="shrink-0 text-xs text-lab-accent">
                  Detail
                </Link>
              </div>
              <iframe
                title={r.title}
                src={r.play_url}
                className="h-[min(55vh,480px)] w-full bg-black"
                sandbox="allow-scripts allow-same-origin"
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function LabComparePage() {
  return (
    <Suspense fallback={<div className="text-sm text-lab-muted">Loading compare…</div>}>
      <CompareInner />
    </Suspense>
  );
}
