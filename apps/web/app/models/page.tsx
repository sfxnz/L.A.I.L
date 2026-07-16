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
    <div className="flex h-full min-h-0 flex-col bg-[#0e0e0e]">
      <div className="flex h-10 shrink-0 items-center justify-between border-b border-[#1f1f1f] px-4">
        <h1 className="text-[13px] font-semibold text-[#e4e4e4]">Models</h1>
        <span className="text-[11px] text-[#666]">MODEL LIBRARY</span>
      </div>

      <div className="flex min-h-0 flex-1">
        {/* Left subnav like inspo Get / Serves / Downloads */}
        <div className="w-[140px] shrink-0 border-r border-[#1f1f1f] bg-[#111] p-2 text-[12px]">
          <div className="rounded-md bg-[#2a2a2a] px-2 py-1.5 text-white">Get</div>
          <div className="mt-0.5 rounded-md px-2 py-1.5 text-[#888]">Serves</div>
          <div className="rounded-md px-2 py-1.5 text-[#888]">Downloads</div>
        </div>

        {/* Results list */}
        <div className="flex min-w-0 flex-1 flex-col border-r border-[#1f1f1f]">
          <div className="border-b border-[#1f1f1f] p-3">
            <p className="mb-2 text-[12px] text-[#888]">
              Find the right model, check hardware fit, and download its weights.
            </p>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-2.5 top-2 h-3.5 w-3.5 text-[#555]" />
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
            {err && <p className="mt-2 text-[11px] text-[#e07070]">{err}</p>}
          </div>

          {Object.keys(jobs).length > 0 && (
            <div className="space-y-1 border-b border-[#1f1f1f] p-3">
              {Object.entries(jobs).map(([id, j]) => (
                <div key={id} className="text-[11px] text-[#888]">
                  <div className="mb-0.5 flex justify-between">
                    <span className="font-mono">{id.slice(0, 8)}</span>
                    <span>{j.message}</span>
                  </div>
                  <div className="h-1 overflow-hidden rounded bg-[#222]">
                    <div
                      className="h-full bg-[#81a1c1]"
                      style={{ width: `${Math.round(j.progress * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="flex-1 overflow-y-auto">
            {local.length > 0 && (
              <div className="border-b border-[#1f1f1f] px-3 py-2 text-[10px] uppercase tracking-wider text-[#555]">
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
            <div className="border-b border-[#1f1f1f] px-3 py-2 text-[10px] uppercase tracking-wider text-[#555]">
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

        {/* Card detail */}
        <div className="hidden w-[340px] shrink-0 overflow-y-auto p-4 lg:block">
          {selected ? (
            <div className="space-y-3">
              <h2 className="text-[14px] font-semibold text-white">{selected.name}</h2>
              <div className="font-mono text-[11px] text-[#888]">{selected.id}</div>
              <div className="flex flex-wrap gap-1">
                {selected.license && <Badge tone="muted">{selected.license}</Badge>}
                <Badge tone="accent">fit: {selected.hardwareFit || "unknown"}</Badge>
                {selected.pipeline_tag && <Badge tone="muted">{selected.pipeline_tag}</Badge>}
              </div>
              <Panel className="p-3" title="Model card">
                <p className="text-[12px] leading-relaxed text-[#aaa]">
                  Open on Hugging Face for full card, files, and quant variants. Pull downloads
                  weights for vLLM / llama.cpp via HF CLI.
                </p>
                <div className="mt-3 flex gap-2">
                  {!selected.local && (
                    <Btn size="sm" onClick={() => pull(selected.id)}>
                      <Download className="h-3 w-3" /> HF download
                    </Btn>
                  )}
                  <a
                    href={`https://huggingface.co/${selected.id}`}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex h-7 items-center rounded-md border border-[#2b2b2b] px-2 text-[11px] text-[#aaa] hover:bg-[#1a1a1a]"
                  >
                    Open card
                  </a>
                </div>
              </Panel>
            </div>
          ) : (
            <p className="text-[12px] text-[#555]">Select a model to inspect</p>
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
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "flex w-full items-center gap-2 border-b border-[#1a1a1a] px-3 py-2 text-left hover:bg-[#161616]",
        selected && "bg-[#1c1c1c]",
      )}
    >
      <div className="min-w-0 flex-1">
        <div className="truncate text-[12.5px] text-[#e4e4e4]">{m.name}</div>
        <div className="truncate font-mono text-[10px] text-[#666]">{m.id}</div>
      </div>
      {local ? (
        <Badge tone="ok">local</Badge>
      ) : (
        onPull && (
          <span
            role="button"
            tabIndex={0}
            onClick={(e) => {
              e.stopPropagation();
              onPull();
            }}
            className="text-[11px] text-[#81a1c1] hover:underline"
          >
            Get
          </span>
        )
      )}
    </button>
  );
}
