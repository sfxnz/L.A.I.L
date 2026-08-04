"use client";

import { useEffect, useState } from "react";
import { Download, Search } from "lucide-react";
import { api, type ModelCard } from "@/lib/api";
import { onWsEvent, wsSubscribe } from "@/lib/ws";
import { Badge, Btn, Panel, inputCls } from "@/components/ui";
import { cn } from "@/lib/utils";

export default function ModelsPage() {
  const [q, setQ] = useState("gguf");
  const [results, setResults] = useState<ModelCard[]>([]);
  const [local, setLocal] = useState<ModelCard[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<ModelCard | null>(null);
  const [jobs, setJobs] = useState<Record<string, { progress: number; message: string }>>({});

  const loadLocal = () => api.models.local().then((r) => setLocal(r.local)).catch(() => {});

  useEffect(() => {
    loadLocal();
    search();
  }, []);

  useEffect(() => {
    return onWsEvent((event) => {
      if (String(event.type).startsWith("download_")) {
        const jobId = String(event.jobId || "");
        if (event.type === "download_progress") {
          setJobs((j) => ({
            ...j,
            [jobId]: {
              progress: Number(event.progress || 0),
              message: String(event.message || ""),
            },
          }));
        }
        if (event.type === "download_done") {
          setJobs((j) => ({ ...j, [jobId]: { progress: 1, message: "done" } }));
          loadLocal();
        }
        if (event.type === "download_error") {
          setJobs((j) => ({
            ...j,
            [jobId]: { progress: 0, message: String(event.message || "error") },
          }));
        }
      }
    });
  }, []);

  async function search() {
    setBusy(true);
    setErr(null);
    try {
      const r = await api.models.search(q);
      setResults(r.results || []);
      if (r.error) setErr(r.error);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function pull(model: string) {
    try {
      const { jobId } = await api.models.pull(model, "hf");
      wsSubscribe(`download:${jobId}`);
      setJobs((j) => ({ ...j, [jobId]: { progress: 0.05, message: "queued" } }));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="space-y-5">
      <div className="page-header">
        <div>
          <h1 className="page-title">Models</h1>
          <p className="page-sub">
            Find weights, check hardware fit, and download for vLLM / llama.cpp.
          </p>
        </div>
      </div>

      <div className="lab-card flex min-h-[min(70vh,640px)] overflow-hidden">
        <div className="flex min-w-0 flex-1 flex-col border-r border-lab-border-subtle">
          <div className="border-b border-lab-border-subtle p-4">
            <p className="mb-3 text-[12px] text-lab-muted">
              Search Hugging Face, inspect fit, pull weights when ready.
            </p>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-lab-muted" />
                <input
                  className={cn(inputCls, "pl-8")}
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && search()}
                  placeholder="Search Hugging Face models"
                />
              </div>
              <Btn size="sm" onClick={search} disabled={busy}>
                {busy ? "…" : "Search"}
              </Btn>
            </div>
            {err && <p className="mt-2 text-[12px] text-lab-danger">{err}</p>}
          </div>

          {Object.keys(jobs).length > 0 && (
            <div className="space-y-2 border-b border-lab-border-subtle p-4">
              {Object.entries(jobs).map(([id, j]) => (
                <div key={id} className="text-[11px] text-lab-muted">
                  <div className="mb-1 flex justify-between gap-2">
                    <span className="font-mono text-lab-text-dim">{id.slice(0, 8)}</span>
                    <span>{j.message}</span>
                  </div>
                  <div className="h-1 overflow-hidden rounded-full bg-lab-hover">
                    <div
                      className="h-full rounded-full bg-lab-accent transition-[width] duration-300"
                      style={{ width: `${Math.round(j.progress * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="flex-1 overflow-y-auto">
            {local.length > 0 && (
              <div className="border-b border-lab-border-subtle px-4 py-2 text-[10px] font-medium uppercase tracking-[0.08em] text-lab-muted">
                Local
              </div>
            )}
            {local.map((m) => (
              <ModelRow
                key={`l-${m.id}`}
                m={m}
                selected={selected?.id === m.id}
                onSelect={() => setSelected(m)}
                local
              />
            ))}
            <div className="border-b border-lab-border-subtle px-4 py-2 text-[10px] font-medium uppercase tracking-[0.08em] text-lab-muted">
              Model results
            </div>
            {results.map((m) => (
              <ModelRow
                key={m.id}
                m={m}
                selected={selected?.id === m.id}
                onSelect={() => setSelected(m)}
                onPull={() => pull(m.id)}
              />
            ))}
          </div>
        </div>

        <div className="hidden w-[320px] shrink-0 overflow-y-auto p-5 lg:block">
          {selected ? (
            <div className="space-y-4">
              <div>
                <h2 className="text-[15px] font-semibold tracking-[-0.02em] text-lab-text">
                  {selected.name}
                </h2>
                <div className="mt-1 font-mono text-[11px] text-lab-muted">{selected.id}</div>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {selected.license && <Badge tone="muted">{selected.license}</Badge>}
                <Badge tone="accent">fit: {selected.hardwareFit || "unknown"}</Badge>
                {selected.pipeline_tag && <Badge tone="muted">{selected.pipeline_tag}</Badge>}
              </div>
              <Panel className="p-3.5" title="Model card">
                <p className="text-[12px] leading-relaxed text-lab-muted">
                  Open on Hugging Face for full card, files, and quant variants. Pull downloads
                  weights for vLLM / llama.cpp via HF CLI.
                </p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {!selected.local && (
                    <Btn size="sm" onClick={() => pull(selected.id)}>
                      <Download className="h-3 w-3" /> HF download
                    </Btn>
                  )}
                  <a
                    href={`https://huggingface.co/${selected.id}`}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex h-8 items-center rounded-[8px] border border-lab-border px-2.5 text-[11px] font-medium text-lab-text-dim transition-colors hover:bg-lab-hover hover:text-lab-text"
                  >
                    Open card
                  </a>
                </div>
              </Panel>
            </div>
          ) : (
            <p className="pt-8 text-center text-[12px] text-lab-muted">Select a model to inspect</p>
          )}
        </div>
      </div>
    </div>
  );
}

function ModelRow({
  m,
  selected,
  onSelect,
  onPull,
  local,
}: {
  m: ModelCard;
  selected?: boolean;
  onSelect: () => void;
  onPull?: () => void;
  local?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex w-full items-center gap-2 border-b border-lab-border-subtle px-4 py-2.5 transition-colors hover:bg-lab-hover",
        selected && "bg-lab-active",
      )}
    >
      <button
        type="button"
        onClick={onSelect}
        className="min-w-0 flex-1 text-left"
      >
        <div className="truncate text-[13px] font-medium tracking-[-0.01em] text-lab-text">
          {m.name}
        </div>
        <div className="truncate font-mono text-[10px] text-lab-muted">{m.id}</div>
      </button>
      {local ? (
        <Badge tone="ok">local</Badge>
      ) : (
        onPull && (
          <button
            type="button"
            onClick={onPull}
            className="shrink-0 text-[12px] font-medium text-lab-accent-bright hover:underline"
          >
            Get
          </button>
        )
      )}
    </div>
  );
}
