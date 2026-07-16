"use client";

import { useEffect, useState } from "react";
import {
  api,
  watchJob,
  type LabStatus,
  type ServeExample,
  type ServeRecommend,
} from "@/lib/api";
import { Badge, Btn, Field, Input, LogView, ModeBanner, Panel, inputCls } from "@/components/ui";
import { cn } from "@/lib/utils";

type Tab = "serve" | "perf" | "agentic" | "history";

export default function ServerPage() {
  const [tab, setTab] = useState<Tab>("serve");
  const [status, setStatus] = useState<LabStatus | null>(null);
  const [mode, setMode] = useState<"lab_safe" | "workflow_max">("lab_safe");
  const [model, setModel] = useState("");
  const [util, setUtil] = useState("");
  const [maxLen, setMaxLen] = useState("");
  const [port, setPort] = useState("8000");
  const [image, setImage] = useState("");
  const [quantization, setQuantization] = useState("");
  const [kvCacheDtype, setKvCacheDtype] = useState("");
  const [moeBackend, setMoeBackend] = useState("");
  const [maxNumSeqs, setMaxNumSeqs] = useState("");
  const [loadFormat, setLoadFormat] = useState("");
  const [trustRemoteCode, setTrustRemoteCode] = useState(false);
  const [enableAutoTool, setEnableAutoTool] = useState(true);
  const [toolCallParser, setToolCallParser] = useState("qwen3_coder");
  const [reasoningParser, setReasoningParser] = useState("");
  const [chunkedPrefill, setChunkedPrefill] = useState(false);
  const [prefixCaching, setPrefixCaching] = useState(false);
  const [mtp, setMtp] = useState(false);
  const [mtpTokens, setMtpTokens] = useState("2");
  const [dockerEnv, setDockerEnv] = useState("");
  const [extra, setExtra] = useState("");
  const [download, setDownload] = useState(false);
  const [rec, setRec] = useState<ServeRecommend | null>(null);
  const [recBusy, setRecBusy] = useState(false);
  const [recError, setRecError] = useState<string | null>(null);
  const [logs, setLogs] = useState("");
  const [jobMsg, setJobMsg] = useState("");
  const [jobProgress, setJobProgress] = useState(0);
  const [jobStatus, setJobStatus] = useState("");

  const refresh = () => api.labStatus().then(setStatus).catch(console.error);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 8000);
    return () => clearInterval(t);
  }, []);

  const examples = (status?.serve?.serve_examples || {}) as Record<string, ServeExample>;
  const modelHints = status?.serve?.presets || [];

  function track(jobId: string) {
    setLogs("");
    setJobMsg("starting…");
    setJobProgress(0);
    setJobStatus("running");
    watchJob(
      jobId,
      (chunk) => setLogs((l) => (l + chunk).slice(-80_000)),
      (s) => {
        setJobStatus(s.status);
        setJobProgress(s.progress);
        setJobMsg(s.message);
      },
      () => refresh(),
    );
  }

  function applyConfig(c: Record<string, unknown>, modelId?: string) {
    if (modelId) setModel(modelId);
    if (c.model) setModel(String(c.model));
    setQuantization(String(c.quantization ?? ""));
    setKvCacheDtype(String(c.kv_cache_dtype ?? ""));
    setMoeBackend(String(c.moe_backend ?? ""));
    setTrustRemoteCode(!!c.trust_remote_code);
    setReasoningParser(String(c.reasoning_parser ?? ""));
    setToolCallParser(String(c.tool_call_parser ?? ""));
    setEnableAutoTool(!!c.enable_auto_tool_choice);
    setMaxNumSeqs(c.max_num_seqs != null ? String(c.max_num_seqs) : "");
    setDockerEnv(((c.docker_env as string[]) || []).join("\n"));
    setExtra(String(c.extra_flags || ""));
    setMtp(!!c.mtp);
    if (c.mtp_num_tokens != null) setMtpTokens(String(c.mtp_num_tokens));
    setLoadFormat(String(c.load_format || ""));
    setChunkedPrefill(!!c.enable_chunked_prefill);
    setPrefixCaching(!!c.enable_prefix_caching);
    if (c.image) setImage(String(c.image));
    if (c.util != null) setUtil(String(c.util));
    if (c.max_model_len != null) setMaxLen(String(c.max_model_len));
  }

  function applyExample(ex: ServeExample) {
    if (ex.model != null) setModel(ex.model);
    if (ex.quantization != null) setQuantization(ex.quantization);
    if (ex.kv_cache_dtype != null) setKvCacheDtype(ex.kv_cache_dtype);
    if (ex.moe_backend != null) setMoeBackend(ex.moe_backend);
    if (ex.trust_remote_code != null) setTrustRemoteCode(!!ex.trust_remote_code);
    if (ex.reasoning_parser != null) setReasoningParser(ex.reasoning_parser);
    if (ex.tool_call_parser != null) setToolCallParser(ex.tool_call_parser);
    if (ex.enable_auto_tool_choice != null) setEnableAutoTool(!!ex.enable_auto_tool_choice);
    if (ex.max_num_seqs != null && ex.max_num_seqs !== "") setMaxNumSeqs(String(ex.max_num_seqs));
    setDockerEnv((ex.docker_env || []).join("\n"));
    setExtra(ex.extra_flags || "");
    setMtp(!!ex.mtp);
    setRec(null);
  }

  async function autoConfigure() {
    if (!model.trim()) return;
    setRecBusy(true);
    setRecError(null);
    try {
      const r = await api.recommendServe(model.trim(), mode, true);
      setRec(r);
      applyConfig(r.config, r.model);
    } catch (e) {
      setRec(null);
      setRecError(e instanceof Error ? e.message : String(e));
    } finally {
      setRecBusy(false);
    }
  }

  async function start() {
    if (!model.trim()) {
      alert("Model is required");
      return;
    }
    const envLines = dockerEnv
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l && !l.startsWith("#") && l.includes("="));
    const body: Record<string, unknown> = {
      model: model.trim(),
      mode,
      port: parseInt(port, 10) || 8000,
      docker_env: envLines,
      quantization: quantization.trim(),
      kv_cache_dtype: kvCacheDtype.trim(),
      moe_backend: moeBackend.trim(),
      trust_remote_code: trustRemoteCode,
      enable_auto_tool_choice: enableAutoTool,
      tool_call_parser: toolCallParser.trim(),
      reasoning_parser: reasoningParser.trim(),
      mtp,
      mtp_num_tokens: parseInt(mtpTokens, 10) || 2,
      load_format: loadFormat.trim(),
      enable_chunked_prefill: chunkedPrefill,
      enable_prefix_caching: prefixCaching,
      extra_flags: extra,
      stop_first: true,
      download,
    };
    if (util) body.util = parseFloat(util);
    if (maxLen) body.max_model_len = parseInt(maxLen, 10);
    if (image.trim()) body.image = image.trim();
    if (maxNumSeqs) body.max_num_seqs = parseInt(maxNumSeqs, 10);
    const { job_id } = await api.startServe(body);
    track(job_id);
  }

  async function stop() {
    const { job_id } = await api.stopServe();
    track(job_id);
    setTimeout(refresh, 2000);
  }

  async function restore() {
    const { job_id } = await api.agentRestore();
    track(job_id);
  }

  const serve = status?.serve;
  const healthy = !!serve && !serve.unreachable && serve.healthy;

  return (
    <div className="space-y-3 p-4 md:p-5">
      <div className="page-header">
        <div>
          <h1 className="page-title">Server</h1>
          <p className="page-sub">vLLM Lab Safe / Workflow Max · benches · history</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge tone={healthy ? "ok" : "danger"}>{healthy ? "endpoint up" : "endpoint down"}</Badge>
          <Btn variant="secondary" size="sm" onClick={refresh}>
            Refresh
          </Btn>
        </div>
      </div>

      <div className="inline-flex flex-wrap gap-0.5 rounded-md border border-lab-border bg-lab-panel2 p-0.5">
        {(
          [
            ["serve", "Serve"],
            ["perf", "Perf"],
            ["agentic", "Agentic"],
            ["history", "History"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={cn(
              "rounded px-2.5 py-1 text-[12px] transition",
              tab === id
                ? "bg-lab-active font-medium text-lab-text"
                : "text-lab-muted hover:text-lab-text",
            )}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "serve" && (
        <div className="space-y-4">
          <ModeBanner mode={mode} />

          <div className="flex flex-wrap gap-2">
            <Btn
              variant={mode === "lab_safe" ? "primary" : "secondary"}
              onClick={() => setMode("lab_safe")}
            >
              Lab Safe
            </Btn>
            <Btn
              variant={mode === "workflow_max" ? "primary" : "secondary"}
              onClick={() => setMode("workflow_max")}
            >
              Workflow Max
            </Btn>
            <Btn variant="danger" onClick={stop}>
              Stop all
            </Btn>
            <Btn variant="secondary" onClick={restore}>
              Agent restore
            </Btn>
          </div>

          <Panel className="space-y-3 p-4">
            <div className="flex flex-wrap items-end gap-3">
              <div className="min-w-[16rem] flex-1">
                <Field label="Model (HF id)">
                  <input
                    className={inputCls}
                    list="model-hints"
                    placeholder="org/model-name"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                  />
                  <datalist id="model-hints">
                    {modelHints.map((p) => (
                      <option key={p} value={p} />
                    ))}
                  </datalist>
                </Field>
              </div>
              <Btn onClick={autoConfigure} disabled={recBusy || !model.trim()}>
                {recBusy ? "Fetching card…" : "Auto-configure from HF"}
              </Btn>
              <Btn onClick={start}>Start serve</Btn>
            </div>
            {recError && <p className="text-sm text-rose-300 whitespace-pre-wrap">{recError}</p>}
            {rec && (
              <div className="rounded-lg border border-lab-border bg-black/20 p-3 text-xs space-y-1">
                <div className="flex flex-wrap gap-2">
                  <Badge tone="accent">confidence: {rec.confidence}</Badge>
                </div>
                {rec.notes && <p className="text-lab-muted">{rec.notes}</p>}
                <ul className="list-disc pl-4 text-lab-muted">
                  {(rec.rationale || []).slice(0, 6).map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </div>
            )}
          </Panel>

          {Object.keys(examples).length > 0 && (
            <Panel className="p-4">
              <h2 className="mb-2 text-sm font-semibold">Examples</h2>
              <div className="flex flex-wrap gap-2">
                {Object.entries(examples).map(([k, ex]) => (
                  <Btn key={k} variant="secondary" onClick={() => applyExample(ex)}>
                    {ex.label || k}
                  </Btn>
                ))}
              </div>
            </Panel>
          )}

          <Panel className="grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-3">
            <Field label="Util">
              <Input value={util} onChange={(e) => setUtil(e.target.value)} placeholder="0.4" />
            </Field>
            <Field label="Max model len">
              <Input value={maxLen} onChange={(e) => setMaxLen(e.target.value)} placeholder="65536" />
            </Field>
            <Field label="Port">
              <Input value={port} onChange={(e) => setPort(e.target.value)} />
            </Field>
            <Field label="Image">
              <Input value={image} onChange={(e) => setImage(e.target.value)} placeholder="vllm/…" />
            </Field>
            <Field label="Quantization">
              <Input value={quantization} onChange={(e) => setQuantization(e.target.value)} />
            </Field>
            <Field label="KV cache dtype">
              <Input value={kvCacheDtype} onChange={(e) => setKvCacheDtype(e.target.value)} />
            </Field>
            <Field label="MoE backend">
              <Input value={moeBackend} onChange={(e) => setMoeBackend(e.target.value)} />
            </Field>
            <Field label="Max num seqs">
              <Input value={maxNumSeqs} onChange={(e) => setMaxNumSeqs(e.target.value)} />
            </Field>
            <Field label="Load format">
              <Input value={loadFormat} onChange={(e) => setLoadFormat(e.target.value)} />
            </Field>
            <Field label="Tool call parser">
              <Input value={toolCallParser} onChange={(e) => setToolCallParser(e.target.value)} />
            </Field>
            <Field label="Reasoning parser">
              <Input value={reasoningParser} onChange={(e) => setReasoningParser(e.target.value)} />
            </Field>
            <Field label="MTP tokens">
              <Input value={mtpTokens} onChange={(e) => setMtpTokens(e.target.value)} />
            </Field>
            <label className="flex items-center gap-2 text-sm text-lab-muted">
              <input type="checkbox" checked={trustRemoteCode} onChange={(e) => setTrustRemoteCode(e.target.checked)} />
              trust remote code
            </label>
            <label className="flex items-center gap-2 text-sm text-lab-muted">
              <input type="checkbox" checked={enableAutoTool} onChange={(e) => setEnableAutoTool(e.target.checked)} />
              auto tool choice
            </label>
            <label className="flex items-center gap-2 text-sm text-lab-muted">
              <input type="checkbox" checked={chunkedPrefill} onChange={(e) => setChunkedPrefill(e.target.checked)} />
              chunked prefill
            </label>
            <label className="flex items-center gap-2 text-sm text-lab-muted">
              <input type="checkbox" checked={prefixCaching} onChange={(e) => setPrefixCaching(e.target.checked)} />
              prefix caching
            </label>
            <label className="flex items-center gap-2 text-sm text-lab-muted">
              <input type="checkbox" checked={mtp} onChange={(e) => setMtp(e.target.checked)} />
              MTP
            </label>
            <label className="flex items-center gap-2 text-sm text-lab-muted">
              <input type="checkbox" checked={download} onChange={(e) => setDownload(e.target.checked)} />
              download weights first
            </label>
            <div className="sm:col-span-2 lg:col-span-3">
              <Field label="Docker env (KEY=VALUE per line)">
                <textarea className={cn(inputCls, "min-h-[72px] font-mono text-xs")} value={dockerEnv} onChange={(e) => setDockerEnv(e.target.value)} />
              </Field>
            </div>
            <div className="sm:col-span-2 lg:col-span-3">
              <Field label="Extra flags">
                <Input value={extra} onChange={(e) => setExtra(e.target.value)} placeholder="--flag value" />
              </Field>
            </div>
          </Panel>

          <Panel className="space-y-2 p-4">
            <div className="flex items-center justify-between text-xs text-lab-muted">
              <span>
                Job: {jobStatus || "idle"} · {jobMsg}
              </span>
              <span>{Math.round(jobProgress * 100)}%</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-zinc-800">
              <div className="h-full bg-lab-accent transition-all" style={{ width: `${Math.round(jobProgress * 100)}%` }} />
            </div>
            <LogView text={logs} />
          </Panel>
        </div>
      )}

      {tab === "perf" && <PerfTab track={track} />}
      {tab === "agentic" && <AgenticTab track={track} />}
      {tab === "history" && <HistoryTab />}
    </div>
  );
}

function PerfTab({ track }: { track: (id: string) => void }) {
  const [intent, setIntent] = useState("lab_safe");
  return (
    <Panel className="space-y-3 p-4">
      <h2 className="text-sm font-semibold">Performance bench</h2>
      <p className="text-xs text-lab-muted">
        Requires a healthy endpoint on :8000. Serve first, then smoke, then bench.
      </p>
      <Field label="Intent tag">
        <select
          className={inputCls}
          value={intent}
          onChange={(e) => setIntent(e.target.value)}
        >
          <option value="lab_safe">lab_safe</option>
          <option value="workflow_max">workflow_max</option>
          <option value="attach">attach</option>
        </select>
      </Field>
      <div className="flex flex-wrap gap-2">
        <Btn
          onClick={async () => {
            await api.smoke();
            alert("Smoke complete — check server logs / response");
          }}
        >
          Smoke
        </Btn>
        <Btn
          onClick={async () => {
            const { job_id } = await api.benchPerf({ intent, kind: "workflow" });
            track(job_id);
          }}
        >
          Run workflow perf
        </Btn>
      </div>
    </Panel>
  );
}

function AgenticTab({ track }: { track: (id: string) => void }) {
  return (
    <Panel className="space-y-3 p-4">
      <h2 className="text-sm font-semibold">Agentic eval</h2>
      <p className="text-xs text-lab-muted">
        Golden tools suite (requires tool-call parser on serve). Optional tool-eval-bench if installed.
      </p>
      <Btn
        onClick={async () => {
          const { job_id } = await api.benchAgentic({ suite: "golden" });
          track(job_id);
        }}
      >
        Run golden tools
      </Btn>
    </Panel>
  );
}

function HistoryTab() {
  const [runs, setRuns] = useState<Awaited<ReturnType<typeof api.runs>>>([]);
  useEffect(() => {
    api.runs().then(setRuns).catch(() => {});
  }, []);
  return (
    <Panel className="p-4">
      <h2 className="mb-3 text-sm font-semibold">Run history</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="text-lab-muted">
            <tr>
              <th className="pb-2">Run</th>
              <th className="pb-2">Kind</th>
              <th className="pb-2">Intent</th>
              <th className="pb-2">Model</th>
              <th className="pb-2">Created</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run_id} className="border-t border-lab-border/50">
                <td className="py-2 font-mono">{r.run_id}</td>
                <td className="py-2">{r.kind}</td>
                <td className="py-2">{r.intent || "—"}</td>
                <td className="py-2">{r.model_id?.split("/").pop() || "—"}</td>
                <td className="py-2">{r.created_at}</td>
              </tr>
            ))}
            {!runs.length && (
              <tr>
                <td colSpan={5} className="py-4 text-lab-muted">
                  No runs yet
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
