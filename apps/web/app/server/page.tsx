"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  watchJob,
  type LabStatus,
  type ServeExample,
  type ServeRecommend,
} from "@/lib/api";
import {
  Badge,
  Btn,
  Callout,
  Field,
  Input,
  LogView,
  Panel,
  ProgressBar,
  SegmentedControl,
  Skeleton,
  inputCls,
  btnClass,
} from "@/components/ui";
import { cn } from "@/lib/utils";
import { usePageTitle } from "@/lib/usePageTitle";

type Tab = "serve" | "perf" | "agentic" | "history";

export default function ServerPage() {
  usePageTitle("Serve");
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
  const [tpSize, setTpSize] = useState("");
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
  const [statusLoading, setStatusLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [startBusy, setStartBusy] = useState(false);
  const [advOpen, setAdvOpen] = useState(false);
  const [formFlash, setFormFlash] = useState<string | null>(null);
  const [appliedExample, setAppliedExample] = useState<string | null>(null);
  const unwatchRef = useRef<null | (() => void)>(null);
  const jobPanelRef = useRef<HTMLDivElement>(null);

  const refresh = (opts?: { soft?: boolean }) => {
    if (!opts?.soft) setRefreshing(true);
    return api
      .labStatus()
      .then(setStatus)
      .catch((e) => setStartError(String((e as Error).message || e)))
      .finally(() => {
        setStatusLoading(false);
        setRefreshing(false);
      });
  };

  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh({ soft: true }), 8000);
    return () => clearInterval(t);
  }, []);

  const examples = (status?.serve?.serve_examples || {}) as Record<string, ServeExample>;
  const modelHints = status?.serve?.presets || [];
  const jobRunning = jobStatus === "running" || jobStatus === "queued";

  const advancedHasValues = useMemo(() => {
    return !!(
      image.trim() ||
      quantization.trim() ||
      kvCacheDtype.trim() ||
      moeBackend.trim() ||
      maxNumSeqs.trim() ||
      tpSize.trim() ||
      loadFormat.trim() ||
      toolCallParser.trim() ||
      reasoningParser.trim() ||
      dockerEnv.trim() ||
      extra.trim() ||
      trustRemoteCode ||
      enableAutoTool ||
      chunkedPrefill ||
      prefixCaching ||
      mtp
    );
  }, [
    image,
    quantization,
    kvCacheDtype,
    moeBackend,
    maxNumSeqs,
    tpSize,
    loadFormat,
    toolCallParser,
    reasoningParser,
    dockerEnv,
    extra,
    trustRemoteCode,
    enableAutoTool,
    chunkedPrefill,
    prefixCaching,
    mtp,
  ]);

  useEffect(() => {
    if (advancedHasValues) setAdvOpen(true);
  }, [advancedHasValues]);

  function track(jobId: string) {
    unwatchRef.current?.();
    setLogs("");
    setJobMsg("starting…");
    setJobProgress(0);
    setJobStatus("running");
    // Feedback is below the fold on Serve — pull the dock into view on start
    requestAnimationFrame(() => {
      jobPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    unwatchRef.current = watchJob(
      jobId,
      (chunk) => setLogs((l) => (l + chunk).slice(-80_000)),
      (s) => {
        setJobStatus(s.status);
        setJobProgress(s.progress);
        setJobMsg(s.message);
      },
      () => {
        void refresh({ soft: true });
      },
    );
  }

  // Re-attach Job panel only to a *live* job (not orphaned sqlite rows)
  useEffect(() => {
    let cancelled = false;
    const STALE_MS = 30 * 60 * 1000;
    api
      .jobs()
      .then(async (list) => {
        if (cancelled) return;
        const active = list.find((j) => j.status === "running" || j.status === "queued");
        if (!active?.job_id) return;
        const updated = active.updated_at ? Date.parse(active.updated_at) : NaN;
        if (Number.isFinite(updated) && Date.now() - updated > STALE_MS) {
          // Stale “running” row after engine restart — don't lie in the Job panel
          return;
        }
        try {
          const fresh = await api.job(active.job_id);
          if (cancelled) return;
          if (fresh.status === "running" || fresh.status === "queued") {
            track(active.job_id);
          }
        } catch {
          /* orphan / missing */
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
      unwatchRef.current?.();
      unwatchRef.current = null;
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
    if (c.tensor_parallel_size != null && c.tensor_parallel_size !== "")
      setTpSize(String(c.tensor_parallel_size));
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

  function applyExample(ex: ServeExample, key?: string) {
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
    setAppliedExample(key || ex.label || ex.model || "example");
    setFormFlash(`Filled form from ${ex.label || ex.model || "example"} — review flags, then Start`);
    window.setTimeout(() => setFormFlash(null), 3200);
  }

  function clearJobPanel() {
    unwatchRef.current?.();
    unwatchRef.current = null;
    setLogs("");
    setJobMsg("");
    setJobProgress(0);
    setJobStatus("");
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
    setTpSize("");
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
      setStartError("Model is required — pick an HF id or proven example first.");
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
    if (tpSize) body.tensor_parallel_size = parseInt(tpSize, 10);
    setStartBusy(true);
    try {
      const { job_id } = await api.startServe(body);
      track(job_id);
    } catch (e) {
      setStartError(e instanceof Error ? e.message : String(e));
    } finally {
      setStartBusy(false);
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
    <div className="space-y-4 lab-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Serve</h1>
          <p className="page-sub">
            Serve local / HF models on Spark · Auto-configure from live model card · benches · history
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2" aria-live="polite">
          {statusLoading ? (
            <Badge tone="muted">checking…</Badge>
          ) : (
            <Badge tone={healthy ? "ok" : "muted"} dot>
              {healthy ? "endpoint up" : "no model"}
            </Badge>
          )}
          {avail != null && (
            <Badge
              tone={
                headroom === "critical" ? "danger" : headroom === "tight" ? "warn" : "ok"
              }
            >
              free {avail} GiB
            </Badge>
          )}
          <Btn variant="secondary" size="sm" onClick={() => void refresh()} loading={refreshing}>
            Refresh
          </Btn>
          <Btn variant="ghost" size="sm" onClick={clearForm}>
            Clear form
          </Btn>
        </div>
      </div>

      <SegmentedControl
        ariaLabel="Serve sections"
        value={tab}
        onChange={setTab}
        options={[
          { id: "serve", label: "Serve" },
          { id: "perf", label: "Perf" },
          { id: "agentic", label: "Agentic" },
          { id: "history", label: "History" },
        ]}
      />

      {tab === "serve" && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <SegmentedControl
              ariaLabel="Serve mode envelope"
              value={mode}
              onChange={setMode}
              options={[
                { id: "lab_safe", label: "Lab Safe" },
                { id: "workflow_max", label: "Workflow Max" },
              ]}
            />
            <span className="hidden max-w-md text-[11px] leading-snug text-lab-muted sm:inline">
              {mode === "lab_safe"
                ? "util ≤ 0.4 · headroom for OS / Hermes · re-run Auto-configure after switch"
                : "util ~0.7–0.85 · large weights / long ctx · re-run Auto-configure after switch"}
            </span>
            <div className="ml-auto flex flex-wrap gap-2">
              <Btn
                variant="danger"
                onClick={stop}
                disabled={startBusy}
                title={startBusy ? "Wait for start request to register" : "Stop all vLLM containers"}
              >
                Stop all
              </Btn>
              <Btn
                variant="secondary"
                onClick={restore}
                disabled={jobRunning}
                title={jobRunning ? "Wait for the current job to finish" : "Restore agent-friendly serve"}
              >
                Agent restore
              </Btn>
            </div>
          </div>

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
                  <Field label="Model (HF id)" htmlFor="serve-model">
                    <input
                      id="serve-model"
                      className={inputCls}
                      list="model-hints"
                      placeholder="org/model-name"
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && model.trim() && !recBusy) void autoConfigure();
                      }}
                      aria-invalid={!!startError && !model.trim() ? true : undefined}
                    />
                    <datalist id="model-hints">
                      {modelHints.map((p) => (
                        <option key={p} value={p} />
                      ))}
                    </datalist>
                  </Field>
                </div>
                <Btn
                  variant="secondary"
                  onClick={() => void autoConfigure()}
                  disabled={!model.trim()}
                  loading={recBusy}
                  title={!model.trim() ? "Enter a model id first" : undefined}
                >
                  {recBusy ? "Fetching card…" : "Auto-configure from HF"}
                </Btn>
                <Btn
                  onClick={() => void start()}
                  disabled={jobRunning || !model.trim()}
                  loading={startBusy}
                  title={
                    !model.trim()
                      ? "Enter a model id first"
                      : jobRunning
                        ? "Wait for the current job to finish"
                        : undefined
                  }
                >
                  Start serve
                </Btn>
              </div>
              {recError && (
                <Callout
                  tone="danger"
                  title="Auto-configure failed"
                  onDismiss={() => setRecError(null)}
                >
                  <span className="whitespace-pre-wrap">{recError}</span>
                </Callout>
              )}
              {startError && (
                <Callout
                  tone="danger"
                  title="Serve action failed"
                  onDismiss={() => setStartError(null)}
                >
                  <span className="whitespace-pre-wrap">{startError}</span>
                </Callout>
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
                    {rec.topology && (rec.topology.nodes ?? 1) >= 1 && (
                      <Badge
                        tone={rec.topology.fits === false ? "danger" : rec.topology.fabric_ok || (rec.topology.nodes_used ?? 1) === 1 ? "ok" : "warn"}
                        dot
                      >
                        {(rec.topology.nodes_used ?? 1) >= 2
                          ? `${rec.topology.nodes_used}-node · TP=${rec.topology.tensor_parallel_size ?? rec.topology.nodes_used}`
                          : "single-node · TP=1"}
                        {rec.topology.weights_gib ? ` · ~${rec.topology.weights_gib} GiB` : ""}
                        {(rec.topology.nodes_used ?? 1) >= 2
                          ? rec.topology.fabric_ok
                            ? " · fabric ok"
                            : " · fabric check failed"
                          : ""}
                      </Badge>
                    )}
                    {rec.topology?.overlay && (
                      <Badge tone="accent">family: {rec.topology.overlay}</Badge>
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

            <Panel className="space-y-3 p-4">
              <h2 className="text-sm font-semibold">Live status</h2>
              {statusLoading ? (
                <div className="space-y-3" aria-busy="true" aria-label="Loading live status">
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-[80%]" />
                  <Skeleton className="h-4 w-[60%]" />
                  <Skeleton className="h-4 w-[66%]" />
                </div>
              ) : (
                <>
                  <dl className="space-y-2 text-sm">
                    <div className="flex justify-between gap-4">
                      <dt className="text-lab-muted">Served model</dt>
                      <dd className="break-all text-right font-mono text-xs">
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
                      <dd className="text-right text-xs">
                        {String(serve?.hardware?.gpu_sku || "—")
                          .replace(/\s*,?\s*\[?N\/A\]?/gi, "")
                          .replace(/\s{2,}/g, " ")
                          .trim() || "—"}
                      </dd>
                    </div>
                    <div className="flex justify-between gap-4">
                      <dt className="text-lab-muted">Health</dt>
                      <dd>
                        <Badge tone={healthy ? "ok" : "muted"} dot>
                          {healthy ? "Healthy" : serve?.unreachable ? "Engine down" : "Idle"}
                        </Badge>
                      </dd>
                    </div>
                  </dl>
                  <div className="space-y-1">
                    <div className="text-[10px] uppercase tracking-wide text-lab-muted">
                      Containers
                    </div>
                    {(serve?.containers || []).length === 0 && (
                      <p className="text-xs text-lab-muted">None matching vLLM.</p>
                    )}
                    {(serve?.containers || []).map((c) => (
                      <div
                        key={c.name}
                        className="flex items-center justify-between gap-2 rounded border border-lab-border px-2 py-1.5 text-[11px]"
                      >
                        <span className="truncate font-mono">{c.name}</span>
                        <Badge tone={c.status.includes("Up") ? "ok" : "muted"}>{c.status}</Badge>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </Panel>
          </div>

          {Object.keys(examples).length > 0 && (
            <Panel className="p-4">
              <h2 className="mb-1 text-sm font-semibold">Proven examples (fill form only)</h2>
              <p className="mb-2 text-[11px] text-lab-muted">
                Static Spark-proven presets. Prefer Auto-configure for newest HF cards.
              </p>
              {formFlash && (
                <Callout tone="ok" className="mb-3" title="Form updated">
                  {formFlash}
                </Callout>
              )}
              <div className="flex flex-wrap gap-2">
                {Object.entries(examples).map(([k, ex]) => {
                  const selected = appliedExample === k || appliedExample === (ex.label || ex.model);
                  return (
                  <button
                    key={k}
                    type="button"
                    onClick={() => applyExample(ex, k)}
                    aria-pressed={selected}
                    className={cn(
                      "rounded-[10px] border px-3 py-2.5 text-left text-xs transition-colors",
                      selected
                        ? "border-lab-accent/45 bg-[rgba(10,132,255,0.1)] text-lab-text"
                        : "border-lab-border text-lab-muted hover:border-lab-accent/40 hover:text-lab-text",
                    )}
                  >
                    <div className="font-medium text-lab-text">{ex.label || k}</div>
                    {ex.model && <div className="font-mono opacity-70">{ex.model}</div>}
                  </button>
                  );
                })}
              </div>
            </Panel>
          )}

          <Panel
            title="Engine flags"
            action={
              <span className="text-[11px] text-lab-muted">
                Envelope first · advanced on demand
              </span>
            }
          >
            <div className="space-y-4 p-4">
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                <Field label="gpu-memory-utilization" htmlFor="serve-util">
                  <Input
                    id="serve-util"
                    value={util}
                    onChange={(e) => setUtil(e.target.value)}
                    placeholder={mode === "lab_safe" ? "0.4" : "0.85"}
                  />
                </Field>
                <Field label="max-model-len" htmlFor="serve-maxlen">
                  <Input
                    id="serve-maxlen"
                    value={maxLen}
                    onChange={(e) => setMaxLen(e.target.value)}
                    placeholder={mode === "lab_safe" ? "65536" : "262144"}
                  />
                </Field>
                <Field label="Port" htmlFor="serve-port">
                  <Input id="serve-port" value={port} onChange={(e) => setPort(e.target.value)} />
                </Field>
                <Field label="tensor-parallel-size (Sparks)" htmlFor="serve-tp">
                  <Input
                    id="serve-tp"
                    value={tpSize}
                    onChange={(e) => setTpSize(e.target.value)}
                    placeholder="1"
                  />
                </Field>
              </div>

              <label className="flex cursor-pointer items-center gap-2.5 rounded-[10px] border border-lab-border-subtle bg-lab-editor/50 px-3 py-2 text-[12px] text-lab-text-dim">
                <input
                  type="checkbox"
                  className="h-3.5 w-3.5 accent-lab-accent"
                  checked={download}
                  onChange={(e) => setDownload(e.target.checked)}
                />
                Download weights first (hf download) before docker start
              </label>

              <details
                className="group rounded-[12px] border border-lab-border-subtle bg-lab-editor/40 open:bg-lab-editor/60"
                open={advOpen}
                onToggle={(e) => setAdvOpen((e.currentTarget as HTMLDetailsElement).open)}
              >
                <summary className="cursor-pointer list-none px-3.5 py-2.5 text-[12px] font-medium tracking-[-0.01em] text-lab-text-dim marker:content-none [&::-webkit-details-marker]:hidden">
                  <span className="inline-flex items-center gap-2">
                    <span className="text-lab-muted transition-transform group-open:rotate-90">▸</span>
                    Advanced flags
                    <span className="font-normal text-lab-muted">
                      image · quant · tools · MTP · docker env
                    </span>
                  </span>
                </summary>
                <div className="grid gap-3 border-t border-lab-border-subtle p-3.5 sm:grid-cols-2 lg:grid-cols-3">
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
                    <Input
                      value={maxNumSeqs}
                      onChange={(e) => setMaxNumSeqs(e.target.value)}
                      placeholder="4"
                    />
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
                      className="accent-lab-accent"
                      checked={trustRemoteCode}
                      onChange={(e) => setTrustRemoteCode(e.target.checked)}
                    />
                    --trust-remote-code
                  </label>
                  <label className="flex items-center gap-2 text-sm text-lab-muted">
                    <input
                      type="checkbox"
                      className="accent-lab-accent"
                      checked={enableAutoTool}
                      onChange={(e) => setEnableAutoTool(e.target.checked)}
                    />
                    --enable-auto-tool-choice
                  </label>
                  <label className="flex items-center gap-2 text-sm text-lab-muted">
                    <input
                      type="checkbox"
                      className="accent-lab-accent"
                      checked={chunkedPrefill}
                      onChange={(e) => setChunkedPrefill(e.target.checked)}
                    />
                    --enable-chunked-prefill
                  </label>
                  <label className="flex items-center gap-2 text-sm text-lab-muted">
                    <input
                      type="checkbox"
                      className="accent-lab-accent"
                      checked={prefixCaching}
                      onChange={(e) => setPrefixCaching(e.target.checked)}
                    />
                    --enable-prefix-caching
                  </label>
                  <label className="flex items-center gap-2 text-sm text-lab-muted">
                    <input
                      type="checkbox"
                      className="accent-lab-accent"
                      checked={mtp}
                      onChange={(e) => setMtp(e.target.checked)}
                    />
                    MTP (--speculative-config method=mtp)
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
                        placeholder="--flag value   (appended after structured fields; duplicates of structured flags are stripped)"
                      />
                    </Field>
                  </div>
                </div>
              </details>
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
      <div
        ref={jobPanelRef}
        id="serve-job-dock"
        className={cn(
          jobRunning &&
            "sticky bottom-3 z-10 rounded-[16px] shadow-[0_-8px_32px_rgba(0,0,0,0.45)]",
        )}
      >
        <Panel
          title="Job"
          className={cn(jobRunning && "border-lab-accent/30 ring-1 ring-[rgba(10,132,255,0.12)]")}
          action={
            <div className="flex items-center gap-2">
              {(jobRunning || jobStatus || logs) && (
                <Btn variant="ghost" size="sm" onClick={clearJobPanel} title="Clear job panel">
                  Dismiss
                </Btn>
              )}
              <Badge
                tone={
                  jobStatus === "done" || jobStatus === "completed"
                    ? "ok"
                    : jobStatus === "error" || jobStatus === "failed"
                      ? "danger"
                      : jobRunning
                        ? "accent"
                        : "muted"
                }
                dot={jobRunning}
              >
                {jobStatus || "idle"}
              </Badge>
            </div>
          }
        >
          <div className="space-y-3 p-4">
            {jobRunning || jobStatus || logs ? (
              <>
                <ProgressBar
                  value={Math.round((jobProgress || 0) * 100)}
                  indeterminate={jobRunning && !(jobProgress > 0)}
                  label={jobMsg || (jobRunning ? "Working…" : "Last job")}
                />
                <LogView
                  text={logs}
                  live={jobRunning}
                  empty="Job output appears here when you start a serve, stop, bench, or agentic run."
                />
              </>
            ) : (
              <p className="text-[12px] leading-relaxed text-lab-muted">
                Idle — start a serve, stop, bench, or agentic run and live logs stream here.
              </p>
            )}
          </div>
        </Panel>
      </div>
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
        <Callout tone="warn" title="Endpoint down">
          Start a model on Serve first. Smoke and perf need a healthy :8000.
        </Callout>
      )}
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Intent tag" htmlFor="perf-intent">
          <select
            id="perf-intent"
            className={inputCls}
            value={intent}
            onChange={(e) => setIntent(e.target.value)}
          >
            <option value="lab_safe">lab_safe</option>
            <option value="workflow_max">workflow_max</option>
            <option value="attach">attach</option>
          </select>
        </Field>
        <Field label="Runner" htmlFor="perf-runner">
          <select
            id="perf-runner"
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
      {err && (
        <Callout tone="danger" title="Perf action failed" onDismiss={() => setErr(null)}>
          {err}
        </Callout>
      )}
      {smokeResult && (
        <pre className="rounded border border-lab-border bg-lab-editor p-2 text-[11px] whitespace-pre-wrap">
          {smokeResult}
        </pre>
      )}
      <div className="flex flex-wrap gap-2">
        <Btn
          disabled={!healthy}
          title={!healthy ? "Start a model first" : undefined}
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
          disabled={!healthy}
          title={!healthy ? "Start a model first" : undefined}
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
          <Callout tone="warn" title="Endpoint down">
            Start a model on Serve first. Agentic benches need a healthy OpenAI-compatible endpoint.
          </Callout>
        )}
        {err && (
          <Callout tone="danger" title="Agentic action failed" onDismiss={() => setErr(null)}>
            {err}
          </Callout>
        )}
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
          title={!healthy ? "Start a model first" : undefined}
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
            <SegmentedControl
              ariaLabel="tool-eval-bench preset"
              value={preset}
              onChange={setPreset}
              options={[
                { id: "short", label: "Short (15)" },
                { id: "full", label: "Full (69)" },
                { id: "hardmode", label: "Hard mode" },
                { id: "coding", label: "Coding cats" },
              ]}
            />
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
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = (soft?: boolean) => {
    if (!soft) setRefreshing(true);
    api
      .runs()
      .then((r) => {
        setRuns(r);
        setErr(null);
      })
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)))
      .finally(() => {
        setLoading(false);
        setRefreshing(false);
      });
  };

  useEffect(() => {
    load(true);
  }, []);

  return (
    <Panel
      title="Run history"
      action={
        <Btn size="sm" variant="secondary" loading={refreshing} onClick={() => load()}>
          Refresh
        </Btn>
      }
    >
      <div className="space-y-3 p-2">
        {err && (
          <Callout tone="danger" title="Couldn’t load runs" onDismiss={() => setErr(null)}>
            {err}
          </Callout>
        )}
        <div className="overflow-x-auto">
          <table className="lab-table">
            <thead>
              <tr>
                <th scope="col">Run</th>
                <th scope="col">Kind</th>
                <th scope="col">Intent</th>
                <th scope="col">Model</th>
                <th scope="col">Created</th>
              </tr>
            </thead>
            <tbody>
              {loading && (
                <tr>
                  <td colSpan={5} className="!p-3">
                    <div className="space-y-2" aria-busy="true">
                      {[0, 1, 2].map((i) => (
                        <Skeleton key={i} className="h-3 w-full" />
                      ))}
                    </div>
                  </td>
                </tr>
              )}
              {!loading &&
                runs.map((r) => {
                  const isTool =
                    r.kind === "agentic_tool_eval" || String(r.kind || "").includes("tool");
                  return (
                    <tr key={r.run_id}>
                      <td className="font-mono text-[11px]">
                        {isTool ? (
                          <a
                            href={`/evals/tool/${r.run_id}`}
                            className="text-lab-accent-bright underline-offset-2 hover:underline"
                          >
                            {r.run_id}
                          </a>
                        ) : (
                          r.run_id
                        )}
                      </td>
                      <td className="font-mono text-[11px] text-lab-muted">{r.kind}</td>
                      <td>{r.intent || "—"}</td>
                      <td className="max-w-[200px] truncate">
                        {r.model_id?.split("/").pop() || "—"}
                      </td>
                      <td className="text-lab-muted">{r.created_at?.slice(0, 19) || "—"}</td>
                    </tr>
                  );
                })}
              {!loading && !runs.length && (
                <tr>
                  <td colSpan={5} className="py-6 text-center text-[12px] text-lab-muted">
                    No runs yet — smoke or perf when the endpoint is healthy.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </Panel>
  );
}
