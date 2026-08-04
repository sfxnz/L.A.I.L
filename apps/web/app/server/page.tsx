"use client";

import { useEffect, useState } from "react";
import {
  api,
  watchJob,
  type LabStatus,
  type ServeExample,
  type ServeRecommend,
} from "@/lib/api";
import { Badge, Btn, Field, Input, LogView, ModeBanner, Panel, inputCls, btnClass } from "@/components/ui";
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
  const [enableAutoTool, setEnableAutoTool] = useState(false);
  const [toolCallParser, setToolCallParser] = useState("");
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
  const [startError, setStartError] = useState<string | null>(null);

  const refresh = () => api.labStatus().then(setStatus).catch(console.error);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 8000);
    return () => clearInterval(t);
  }, []);

  const examples = (status?.serve?.serve_examples || {}) as Record<string, ServeExample>;
  const modelHints = status?.serve?.presets || [];
  const jobRunning = jobStatus === "running" || jobStatus === "queued";

  function track(jobId: string) {
    setLogs("");
    setJobMsg("starting…");
    setJobProgress(0);
    setJobStatus("running");
    setTab((t) => t); // keep current tab; job log is always visible below
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

  // Re-attach Job panel to an in-flight server job after refresh/navigation
  useEffect(() => {
    let cancelled = false;
    api
      .jobs()
      .then((list) => {
        if (cancelled) return;
        const active = list.find((j) => j.status === "running" || j.status === "queued");
        if (active?.job_id) track(active.job_id);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mount-only attach
  }, []);

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
    setMaxNumSeqs(c.max_num_seqs != null && c.max_num_seqs !== "" ? String(c.max_num_seqs) : "");
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
    else setMaxNumSeqs("");
    setDockerEnv((ex.docker_env || []).join("\n"));
    setExtra(ex.extra_flags || "");
    setMtp(!!ex.mtp);
    setRec(null);
  }

  function applyRecipeConfig(cfg: Record<string, unknown> | undefined) {
    if (!cfg) return;
    // Merge recipe onto current form without wiping envelope fields unless set
    applyConfig({ ...cfg, model: model || cfg.model });
  }

  function clearForm() {
    setModel("");
    setUtil("");
    setMaxLen("");
    setPort("8000");
    setImage("");
    setQuantization("");
    setKvCacheDtype("");
    setMoeBackend("");
    setMaxNumSeqs("");
    setLoadFormat("");
    setTrustRemoteCode(false);
    setEnableAutoTool(false);
    setToolCallParser("");
    setReasoningParser("");
    setChunkedPrefill(false);
    setPrefixCaching(false);
    setMtp(false);
    setMtpTokens("2");
    setDockerEnv("");
    setExtra("");
    setDownload(false);
    setRec(null);
    setRecError(null);
    setStartError(null);
  }

  async function autoConfigure(nextMode?: "lab_safe" | "workflow_max") {
    if (!model.trim()) {
      setRecError("Enter a model id first (e.g. unsloth/Qwen3.6-35B-A3B-NVFP4)");
      return;
    }
    const m = nextMode ?? mode;
    if (nextMode) setMode(nextMode);
    setRecBusy(true);
    setRecError(null);
    try {
      const r = await api.recommendServe(model.trim(), m, true);
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
    setStartError(null);
    if (!model.trim()) {
      setStartError("Model is required");
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
    try {
      const { job_id } = await api.startServe(body);
      track(job_id);
    } catch (e) {
      setStartError(e instanceof Error ? e.message : String(e));
    }
  }

  async function stop() {
    try {
      const { job_id } = await api.stopServe();
      track(job_id);
      setTimeout(refresh, 2000);
    } catch (e) {
      setStartError(e instanceof Error ? e.message : String(e));
    }
  }

  async function restore() {
    try {
      const { job_id } = await api.agentRestore();
      track(job_id);
    } catch (e) {
      setStartError(e instanceof Error ? e.message : String(e));
    }
  }

  const serve = status?.serve;
  const healthy = Boolean(serve && !serve.unreachable && serve.healthy);
  const avail = serve?.hardware?.available_gib;
  const headroom = serve?.headroom;
  const confTone =
    rec?.confidence === "high" ? "ok" : rec?.confidence === "medium" ? "warn" : "muted";

  return (
    <div className="space-y-4">
      <div className="page-header">
        <div>
          <h1 className="page-title">Serve</h1>
          <p className="page-sub">
            Serve local / HF models on Spark · Auto-configure from live model card · benches · history
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge tone={healthy ? "ok" : "danger"}>{healthy ? "endpoint up" : "endpoint down"}</Badge>
          {avail != null && (
            <Badge
              tone={
                headroom === "critical" ? "danger" : headroom === "tight" ? "warn" : "ok"
              }
            >
              free {avail} GiB
            </Badge>
          )}
          <Btn variant="secondary" size="sm" onClick={refresh}>
            Refresh
          </Btn>
          <Btn variant="ghost" size="sm" onClick={clearForm}>
            Clear form
          </Btn>
        </div>
      </div>

      <div className="inline-flex flex-wrap gap-0.5 rounded-full border border-lab-border bg-lab-panel p-1">
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
            type="button"
            onClick={() => setTab(id)}
            className={cn(
              "rounded-full px-3.5 py-1.5 text-[12px] font-medium tracking-[-0.01em] transition-colors",
              tab === id
                ? "bg-lab-active text-lab-text shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]"
                : "text-lab-muted hover:text-lab-text-dim",
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
            <Btn variant="danger" onClick={stop} disabled={jobRunning}>
              Stop all
            </Btn>
            <Btn variant="secondary" onClick={restore} disabled={jobRunning}>
              Agent restore
            </Btn>
          </div>
          <p className="text-[11px] text-lab-muted -mt-2">
            After switching Lab Safe / Workflow Max, re-run Auto-configure so util / max-model-len
            match the envelope (card flags stay).
          </p>

          <div className="grid gap-3 lg:grid-cols-3">
            <Panel className="space-y-3 p-4 lg:col-span-2">
              <h2 className="text-sm font-semibold">Auto-configure from Hugging Face card</h2>
              <p className="text-[11px] text-lab-muted">
                Fetches the live model card + config from huggingface.co, scores every{" "}
                <code className="text-lab-text">vllm serve</code> recipe, applies checkpoint
                safety (e.g. strips flashinfer_b12x on mixed FP8 MoE), then fills Lab Safe /
                Workflow Max envelope gaps.
              </p>
              <div className="flex flex-wrap items-end gap-3">
                <div className="min-w-[16rem] flex-1">
                  <Field label="Model (HF id)">
                    <input
                      className={inputCls}
                      list="model-hints"
                      placeholder="org/model-name"
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && model.trim() && !recBusy) void autoConfigure();
                      }}
                    />
                    <datalist id="model-hints">
                      {modelHints.map((p) => (
                        <option key={p} value={p} />
                      ))}
                    </datalist>
                  </Field>
                </div>
                <Btn onClick={() => void autoConfigure()} disabled={recBusy || !model.trim()}>
                  {recBusy ? "Fetching card…" : "Auto-configure from HF"}
                </Btn>
                <Btn onClick={() => void start()} disabled={jobRunning || !model.trim()}>
                  Start serve
                </Btn>
              </div>
              {recError && <p className="text-sm text-lab-danger whitespace-pre-wrap">{recError}</p>}
              {startError && (
                <p className="text-sm text-lab-danger whitespace-pre-wrap">{startError}</p>
              )}
              {rec && (
                <div className="rounded-lg border border-lab-border bg-lab-editor p-3 text-xs space-y-2">
                  <div className="flex flex-wrap gap-2 items-center">
                    <Badge tone={confTone}>confidence: {rec.confidence}</Badge>
                    <Badge tone={rec.from_website ? "ok" : "warn"}>
                      {rec.from_website ? "live HF card" : "offline / cache"}
                    </Badge>
                    {rec.hf_token_ok === false && <Badge tone="warn">HF token invalid</Badge>}
                    {rec.label && <span className="text-lab-text">{rec.label}</span>}
                    {!!rec.detected?.family && (
                      <Badge tone="accent">{String(rec.detected.family)}</Badge>
                    )}
                    {!!rec.detected?.is_moe && <Badge tone="accent">MoE</Badge>}
                    {!!rec.detected?.quant_flag && (
                      <Badge tone="muted">quant={String(rec.detected.quant_flag)}</Badge>
                    )}
                    {!!rec.detected?.is_mixed_nvfp4_fp8 && (
                      <Badge tone="warn">mixed NVFP4+FP8</Badge>
                    )}
                  </div>
                  {rec.card_url && (
                    <a
                      className="text-lab-accent-bright underline break-all"
                      href={rec.card_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {rec.card_url}
                    </a>
                  )}
                  {rec.notes && <p className="text-lab-muted">{rec.notes}</p>}
                  {(rec.warnings || []).length > 0 && (
                    <ul className="list-disc pl-4 space-y-1 text-lab-warn">
                      {rec.warnings.map((w, i) => (
                        <li key={i}>{w}</li>
                      ))}
                    </ul>
                  )}
                  {(rec.card_recipes || []).length > 0 && (
                    <details open className="text-lab-muted">
                      <summary className="cursor-pointer text-lab-text">
                        Card recipes ({rec.card_recipes!.length}) — click Apply to try another
                      </summary>
                      <ul className="mt-2 space-y-2">
                        {rec.card_recipes!.map((cr, i) => (
                          <li
                            key={i}
                            className={cn(
                              "rounded border px-2 py-1.5 font-mono text-[11px]",
                              cr.selected
                                ? "border-lab-accent/40 bg-lab-accent/10"
                                : "border-lab-border",
                            )}
                          >
                            <div className="flex flex-wrap items-center gap-2 text-lab-text">
                              <span>score {cr.score}</span>
                              {cr.selected && <Badge tone="ok">selected</Badge>}
                              {cr.section && (
                                <span className="opacity-70 font-sans">{cr.section}</span>
                              )}
                              {!cr.selected && cr.config && (
                                <Btn
                                  size="sm"
                                  variant="secondary"
                                  onClick={() => applyRecipeConfig(cr.config)}
                                >
                                  Apply raw recipe
                                </Btn>
                              )}
                            </div>
                            <div className="mt-1 whitespace-pre-wrap break-all opacity-90">
                              {cr.raw}
                            </div>
                            {(cr.reasons || []).length > 0 && (
                              <ul className="mt-1 list-disc space-y-0.5 pl-4 font-sans text-[10px] text-lab-muted">
                                {cr.reasons!.map((reason, ri) => (
                                  <li
                                    key={ri}
                                    className={
                                      /penalty|not supported|crash|unsafe|salvage/i.test(reason)
                                        ? "text-lab-warn"
                                        : undefined
                                    }
                                  >
                                    {reason}
                                  </li>
                                ))}
                              </ul>
                            )}
                          </li>
                        ))}
                      </ul>
                      <p className="mt-2 font-sans text-[10px] text-lab-muted">
                        &quot;Apply raw recipe&quot; fills form fields from that card snippet only —
                        checkpoint safety / envelope are not re-run. Prefer the selected recipe
                        (already safety-merged) unless you know what you&apos;re doing.
                      </p>
                    </details>
                  )}
                  <details className="text-lab-muted">
                    <summary className="cursor-pointer text-lab-text">Why these flags</summary>
                    <ul className="mt-2 list-disc space-y-1 pl-4">
                      {(rec.rationale || []).map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                    {(rec.sources || []).length > 0 && (
                      <div className="mt-2">
                        <div className="font-medium text-lab-text">Fetched from</div>
                        <ul className="mt-1 space-y-1">
                          {rec.sources!.map((s, i) => (
                            <li key={i} className="font-mono text-[11px] break-all">
                              [{s.kind}] {s.ref}
                              {s.notes ? ` — ${s.notes}` : ""}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </details>
                </div>
              )}
            </Panel>

            <Panel className="p-4 space-y-3">
              <h2 className="text-sm font-semibold">Live status</h2>
              <dl className="space-y-2 text-sm">
                <div className="flex justify-between gap-4">
                  <dt className="text-lab-muted">Served model</dt>
                  <dd className="font-mono text-right text-xs break-all">
                    {serve?.model_id || "—"}
                  </dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-lab-muted">Endpoint</dt>
                  <dd className="font-mono text-xs">{serve?.base_url || "—"}</dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-lab-muted">Available UMA</dt>
                  <dd className="font-mono">
                    {avail != null ? `${avail} GiB` : "—"}
                    {headroom ? ` · ${headroom}` : ""}
                  </dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-lab-muted">GPU</dt>
                  <dd className="text-right text-xs">{serve?.hardware?.gpu_sku || "—"}</dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-lab-muted">Health</dt>
                  <dd>
                    <Badge tone={healthy ? "ok" : "danger"}>
                      {healthy ? "HEALTHY" : serve?.unreachable ? "ENGINE DOWN" : "DOWN"}
                    </Badge>
                  </dd>
                </div>
              </dl>
              <div className="space-y-1">
                <div className="text-[10px] uppercase tracking-wide text-lab-muted">Containers</div>
                {(serve?.containers || []).length === 0 && (
                  <p className="text-xs text-lab-muted">None matching vLLM.</p>
                )}
                {(serve?.containers || []).map((c) => (
                  <div
                    key={c.name}
                    className="flex items-center justify-between gap-2 rounded border border-lab-border px-2 py-1.5 text-[11px]"
                  >
                    <span className="font-mono truncate">{c.name}</span>
                    <Badge tone={c.status.includes("Up") ? "ok" : "muted"}>{c.status}</Badge>
                  </div>
                ))}
              </div>
            </Panel>
          </div>

          {Object.keys(examples).length > 0 && (
            <Panel className="p-4">
              <h2 className="mb-1 text-sm font-semibold">Proven examples (fill form only)</h2>
              <p className="mb-2 text-[11px] text-lab-muted">
                Static Spark-proven presets. Prefer Auto-configure for newest HF cards.
              </p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(examples).map(([k, ex]) => (
                  <button
                    key={k}
                    type="button"
                    onClick={() => applyExample(ex)}
                    className="rounded-[10px] border border-lab-border px-3 py-2.5 text-left text-xs text-lab-muted hover:border-lab-accent/40 hover:text-lab-text"
                  >
                    <div className="font-medium text-lab-text">{ex.label || k}</div>
                    {ex.model && <div className="font-mono opacity-70">{ex.model}</div>}
                  </button>
                ))}
              </div>
            </Panel>
          )}

          <Panel className="grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-3">
            <Field label="gpu-memory-utilization">
              <Input
                value={util}
                onChange={(e) => setUtil(e.target.value)}
                placeholder={mode === "lab_safe" ? "0.4" : "0.85"}
              />
            </Field>
            <Field label="max-model-len">
              <Input
                value={maxLen}
                onChange={(e) => setMaxLen(e.target.value)}
                placeholder={mode === "lab_safe" ? "65536" : "262144"}
              />
            </Field>
            <Field label="Port">
              <Input value={port} onChange={(e) => setPort(e.target.value)} />
            </Field>
            <Field label="vLLM image">
              <Input
                value={image}
                onChange={(e) => setImage(e.target.value)}
                placeholder="vllm/vllm-openai:v0.26.0"
              />
            </Field>
            <Field label="--quantization">
              <Input
                value={quantization}
                onChange={(e) => setQuantization(e.target.value)}
                list="quant-hints"
                placeholder="modelopt | fp8 | compressed-tensors"
              />
              <datalist id="quant-hints">
                <option value="modelopt" />
                <option value="fp8" />
                <option value="compressed-tensors" />
              </datalist>
            </Field>
            <Field label="--kv-cache-dtype">
              <Input
                value={kvCacheDtype}
                onChange={(e) => setKvCacheDtype(e.target.value)}
                list="kv-hints"
                placeholder="fp8"
              />
              <datalist id="kv-hints">
                <option value="fp8" />
                <option value="auto" />
              </datalist>
            </Field>
            <Field label="--moe-backend">
              <Input
                value={moeBackend}
                onChange={(e) => setMoeBackend(e.target.value)}
                list="moe-hints"
                placeholder="empty = auto (recommended for mixed MoE)"
              />
              <datalist id="moe-hints">
                <option value="flashinfer_b12x" />
                <option value="triton" />
              </datalist>
            </Field>
            <Field label="--max-num-seqs">
              <Input value={maxNumSeqs} onChange={(e) => setMaxNumSeqs(e.target.value)} placeholder="4" />
            </Field>
            <Field label="--load-format">
              <Input value={loadFormat} onChange={(e) => setLoadFormat(e.target.value)} />
            </Field>
            <Field label="--tool-call-parser">
              <Input
                value={toolCallParser}
                onChange={(e) => setToolCallParser(e.target.value)}
                placeholder="qwen3_coder"
              />
            </Field>
            <Field label="--reasoning-parser">
              <Input
                value={reasoningParser}
                onChange={(e) => setReasoningParser(e.target.value)}
                placeholder="qwen3"
              />
            </Field>
            <Field label="MTP speculative tokens">
              <Input
                value={mtpTokens}
                onChange={(e) => setMtpTokens(e.target.value)}
                disabled={!mtp}
              />
            </Field>
            <label className="flex items-center gap-2 text-sm text-lab-muted">
              <input
                type="checkbox"
                checked={trustRemoteCode}
                onChange={(e) => setTrustRemoteCode(e.target.checked)}
              />
              --trust-remote-code
            </label>
            <label className="flex items-center gap-2 text-sm text-lab-muted">
              <input
                type="checkbox"
                checked={enableAutoTool}
                onChange={(e) => setEnableAutoTool(e.target.checked)}
              />
              --enable-auto-tool-choice
            </label>
            <label className="flex items-center gap-2 text-sm text-lab-muted">
              <input
                type="checkbox"
                checked={chunkedPrefill}
                onChange={(e) => setChunkedPrefill(e.target.checked)}
              />
              --enable-chunked-prefill
            </label>
            <label className="flex items-center gap-2 text-sm text-lab-muted">
              <input
                type="checkbox"
                checked={prefixCaching}
                onChange={(e) => setPrefixCaching(e.target.checked)}
              />
              --enable-prefix-caching
            </label>
            <label className="flex items-center gap-2 text-sm text-lab-muted">
              <input type="checkbox" checked={mtp} onChange={(e) => setMtp(e.target.checked)} />
              MTP (--speculative-config method=mtp)
            </label>
            <label className="flex items-center gap-2 text-sm text-lab-muted">
              <input
                type="checkbox"
                checked={download}
                onChange={(e) => setDownload(e.target.checked)}
              />
              download weights first (hf download)
            </label>
            <div className="sm:col-span-2 lg:col-span-3">
              <Field label="Docker env (KEY=VALUE per line)">
                <textarea
                  className={cn(inputCls, "min-h-[72px] font-mono text-xs")}
                  value={dockerEnv}
                  onChange={(e) => setDockerEnv(e.target.value)}
                  placeholder={"CUTE_DSL_ARCH=sm_121a\n# only lines you type are passed"}
                />
              </Field>
            </div>
            <div className="sm:col-span-2 lg:col-span-3">
              <Field label="Extra free-form vLLM flags">
                <textarea
                  className={cn(inputCls, "min-h-[56px] font-mono text-xs")}
                  value={extra}
                  onChange={(e) => setExtra(e.target.value)}
                  placeholder='--flag value   (appended after structured fields; duplicates of structured flags are stripped)'
                />
              </Field>
            </div>
          </Panel>
        </div>
      )}

      {tab === "perf" && <PerfTab track={track} healthy={healthy} />}
      {tab === "agentic" && (
        <AgenticTab
          track={track}
          healthy={healthy}
          toolEval={status?.serve?.tool_eval}
        />
      )}
      {tab === "history" && <HistoryTab />}

      {/* Job log always visible so Perf/Agentic jobs aren't invisible on other tabs */}
      <Panel className="space-y-2 p-4">
        <div className="flex items-center justify-between text-xs text-lab-muted">
          <span>
            Job: {jobStatus || "idle"} · {jobMsg}
          </span>
          <span>{Math.round(jobProgress * 100)}%</span>
        </div>
        <div className="h-1.5 overflow-hidden rounded-full bg-lab-hover">
          <div
            className="h-full bg-lab-accent transition-all"
            style={{ width: `${Math.round(jobProgress * 100)}%` }}
          />
        </div>
        <LogView text={logs} />
      </Panel>
    </div>
  );
}

function PerfTab({
  track,
  healthy,
}: {
  track: (id: string) => void;
  healthy: boolean;
}) {
  const [intent, setIntent] = useState("lab_safe");
  const [runner, setRunner] = useState<"workflow" | "prefill" | "concurrency">("workflow");
  const [err, setErr] = useState<string | null>(null);
  const [smokeResult, setSmokeResult] = useState<string | null>(null);

  return (
    <Panel className="space-y-3 p-4">
      <h2 className="text-sm font-semibold">Performance bench</h2>
      <p className="text-xs text-lab-muted">
        Requires a healthy endpoint (usually :8000). Serve first → smoke → bench. Logs appear in
        the Job panel below.
      </p>
      {!healthy && (
        <p className="text-xs text-lab-warn">Endpoint looks down — start a model on Serve first.</p>
      )}
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Intent tag">
          <select className={inputCls} value={intent} onChange={(e) => setIntent(e.target.value)}>
            <option value="lab_safe">lab_safe</option>
            <option value="workflow_max">workflow_max</option>
            <option value="attach">attach</option>
          </select>
        </Field>
        <Field label="Runner">
          <select
            className={inputCls}
            value={runner}
            onChange={(e) => setRunner(e.target.value as typeof runner)}
          >
            <option value="workflow">workflow (realistic multi-turn)</option>
            <option value="prefill">prefill / decode</option>
            <option value="concurrency">concurrency sweep</option>
          </select>
        </Field>
      </div>
      {err && <p className="text-sm text-lab-danger">{err}</p>}
      {smokeResult && (
        <pre className="rounded border border-lab-border bg-lab-editor p-2 text-[11px] whitespace-pre-wrap">
          {smokeResult}
        </pre>
      )}
      <div className="flex flex-wrap gap-2">
        <Btn
          onClick={async () => {
            setErr(null);
            try {
              const r = await api.smoke();
              setSmokeResult(JSON.stringify(r, null, 2));
            } catch (e) {
              setErr(e instanceof Error ? e.message : String(e));
            }
          }}
        >
          Smoke
        </Btn>
        <Btn
          onClick={async () => {
            setErr(null);
            try {
              const { job_id } = await api.benchPerf({ intent, kind: runner, runner });
              track(job_id);
            } catch (e) {
              setErr(e instanceof Error ? e.message : String(e));
            }
          }}
        >
          Run {runner} perf
        </Btn>
      </div>
    </Panel>
  );
}

function AgenticTab({
  track,
  healthy,
  toolEval,
}: {
  track: (id: string) => void;
  healthy: boolean;
  toolEval?: NonNullable<LabStatus["serve"]>["tool_eval"];
}) {
  const [err, setErr] = useState<string | null>(null);
  const [preset, setPreset] = useState<"short" | "full" | "hardmode" | "coding">("short");
  const [teb, setTeb] = useState<{
    available: boolean;
    path?: string | null;
    version?: string | null;
    install?: string;
    repo?: string;
  } | null>(toolEval ?? null);

  useEffect(() => {
    api.toolEvalStatus().then(setTeb).catch(() => {});
  }, []);

  const available = !!(teb?.available ?? toolEval?.available);

  return (
    <div className="space-y-4">
      <Panel className="space-y-3 p-4">
        <h2 className="text-sm font-semibold tracking-[-0.01em]">Golden tools</h2>
        <p className="text-[12px] leading-relaxed text-lab-muted">
          Fast 12-case smoke: does the model pick the right tool (or none)? Needs{" "}
          <code className="font-mono text-[11px] text-lab-text-dim">--enable-auto-tool-choice</code>{" "}
          + tool-call parser on serve. Logs in the Job panel below.
        </p>
        {!healthy && (
          <p className="text-[12px] text-lab-warn">Endpoint looks down — start a model on Serve first.</p>
        )}
        {err && <p className="text-[13px] text-lab-danger">{err}</p>}
        <Btn
          onClick={async () => {
            setErr(null);
            try {
              const { job_id } = await api.benchAgentic({ suite: "golden" });
              track(job_id);
            } catch (e) {
              setErr(e instanceof Error ? e.message : String(e));
            }
          }}
          disabled={!healthy}
        >
          Run golden tools
        </Btn>
      </Panel>

      <Panel className="space-y-3 p-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold tracking-[-0.01em]">Tool Eval Bench</h2>
            <p className="mt-1 max-w-xl text-[12px] leading-relaxed text-lab-muted">
              Full tool-calling quality suite (
              <a
                href="https://github.com/SeraphimSerapis/tool-eval-bench"
                target="_blank"
                rel="noreferrer"
                className="text-lab-accent-bright hover:underline"
              >
                SeraphimSerapis/tool-eval-bench
              </a>
              ) — selection, params, multi-step, restraint, safety, structured output. Scores 0–100
              with safety gating.
            </p>
          </div>
          <Badge tone={available ? "ok" : "warn"} dot>
            {available
              ? `installed${teb?.version ? ` · ${teb.version}` : ""}`
              : "not installed"}
          </Badge>
        </div>

        {!available && (
          <div className="rounded-[12px] border border-[rgba(255,214,10,0.22)] bg-[rgba(255,214,10,0.08)] px-3.5 py-2.5 text-[12px] text-lab-text-dim">
            Install on spark1, then restart <code className="font-mono text-[11px]">bun run dev</code>:
            <pre className="mt-2 overflow-x-auto rounded-[8px] bg-lab-editor p-2.5 font-mono text-[11px] text-lab-text-dim">
              {teb?.install ||
                "uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git"}
            </pre>
          </div>
        )}

        {available && (
          <>
            <div className="flex flex-wrap gap-2">
              {(
                [
                  ["short", "Short (15)"],
                  ["full", "Full (69)"],
                  ["hardmode", "Hard mode"],
                  ["coding", "Coding cats"],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setPreset(id)}
                  className={cn(
                    "rounded-full px-3 py-1.5 text-[12px] font-medium transition-colors",
                    preset === id
                      ? "bg-lab-active text-lab-text shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]"
                      : "text-lab-muted hover:bg-lab-hover hover:text-lab-text-dim",
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
            <p className="text-[11px] text-lab-muted">
              Runs with <code className="font-mono">--no-think</code> against the live OpenAI-compatible
              endpoint. Short ≈ minutes; full suite can take much longer on 27B.
            </p>
            <div className="flex flex-wrap gap-2">
              <Btn
                onClick={async () => {
                  setErr(null);
                  try {
                    const { job_id } = await api.benchAgentic({
                      suite: "tool_eval",
                      preset,
                    });
                    track(job_id);
                  } catch (e) {
                    setErr(e instanceof Error ? e.message : String(e));
                  }
                }}
                disabled={!healthy || !available}
              >
                Run tool-eval-bench ({preset})
              </Btn>
              <a href="/evals/tool" className={btnClass("secondary", "md")}>
                Open results board →
              </a>
            </div>
          </>
        )}
      </Panel>
    </div>
  );
}

function HistoryTab() {
  const [runs, setRuns] = useState<Awaited<ReturnType<typeof api.runs>>>([]);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api
      .runs()
      .then(setRuns)
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, []);
  return (
    <Panel className="p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold">Run history</h2>
        <Btn
          size="sm"
          variant="secondary"
          onClick={() => {
            api
              .runs()
              .then(setRuns)
              .catch((e) => setErr(e instanceof Error ? e.message : String(e)));
          }}
        >
          Refresh
        </Btn>
      </div>
      {err && <p className="mb-2 text-sm text-lab-danger">{err}</p>}
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
