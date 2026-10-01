"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, type ServeExample, type ServeRecommend } from "@/lib/api";
import {
  Badge,
  Btn,
  Callout,
  CheckboxRow,
  EmptyState,
  Eyebrow,
  Field,
  Input,
  LogView,
  Panel,
  ProgressBar,
  Skeleton,
  SyncRing,
  Tick,
  inputCls,
  CopyButton,
  useCopy,
} from "@/components/ui";
import { useShallow } from "zustand/react/shallow";
import { EngineSelect, EngineSummary, engineHas, useServeEngines } from "@/components/serve/EngineSelect";
import { engineLabel } from "@/lib/engines";
import { serveHealthy, tightestNode, useLabStatusStore, useStale } from "@/lib/lab-status-store";
import { CONFIRM_WINDOW_MS, confirmStep } from "@/lib/confirm-click";
import { LiveAge } from "@/components/status/LiveAge";
import { hermesBases } from "@/components/status/EndpointHero";
import { useJobWatch } from "@/lib/use-job-watch";
import { cn } from "@/lib/utils";

// Serve is for serving: throughput / latency live on /bench, smoke, tool-calling
// suites and the run log on /evals.

/** What each engine calls the context window flag (vLLM: max-model-len). */
const CONTEXT_FLAG: Record<string, string> = {
  sglang: "context-length",
  llamacpp: "ctx-size (-c)",
  tensorfold: "context",
};

/* ---------------------------------------------------------------------------
   Local HUD atoms. Everything chromatic here resolves through lab-* tokens so
   the light "reconstruction plate" and the dark "in simulation" void both come
   out deliberate. Nothing below owns behaviour.
   ------------------------------------------------------------------------- */

/**
 * Unset value. A bare em-dash reads as "broken"; a condensed muted word reads
 * as "nothing here yet, and that is the correct state".
 */
function Unset({ children = "Awaiting" }: { children?: ReactNode }) {
  return <Eyebrow>{children}</Eyebrow>;
}

/**
 * Numbered step head. This is the page's spine: every block on Serve carries
 * one, so model → envelope → presets → flags → launch reads as a single
 * operator sequence instead of a stack of unrelated cards.
 *
 * `n` is decorative — sequence position, or "——" for blocks that sit off the
 * spine. Either way a screen reader announcing "em dash em dash" is noise,
 * so the numeral slot is hidden from the a11y tree; the label carries meaning.
 */
function Seq({
  n,
  label,
  hint,
  action,
}: {
  n: string;
  label: string;
  hint?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
      <div className="flex min-w-0 items-center gap-2.5">
        <span
          aria-hidden
          className="font-mono text-[10px] leading-none tabular-nums text-lab-muted"
        >
          {n}
        </span>
        <span aria-hidden className="h-3 w-px shrink-0 bg-lab-accent" />
        <span className="animus-eyebrow whitespace-nowrap text-lab-text-dim">{label}</span>
        {hint ? (
          <>
            <Tick className="hidden md:block" />
            <span className="hidden min-w-0 truncate text-[11px] leading-none text-lab-muted md:inline">
              {hint}
            </span>
          </>
        ) : null}
      </div>
      {action ? <div className="flex flex-wrap items-center gap-2">{action}</div> : null}
    </div>
  );
}

/** One row of the live-endpoint readout: condensed label, tabular value. */
function Readout({
  label,
  value,
  unset = "Awaiting",
  mono = true,
  children,
}: {
  label: string;
  value?: string | null;
  unset?: string;
  mono?: boolean;
  children?: ReactNode;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-lab-border-subtle py-1.5 last:border-b-0">
      <dt className="animus-eyebrow shrink-0 text-[10px] tracking-[0.14em]">{label}</dt>
      <dd
        className={cn(
          "min-w-0 break-all text-right text-[11px] leading-snug text-lab-text-dim",
          mono && "font-mono tabular-nums",
        )}
      >
        {children ?? (value ? value : <Unset>{unset}</Unset>)}
      </dd>
    </div>
  );
}

/** Telemetry cell for the job dock — condensed label over a tabular value. */
function Telem({
  label,
  children,
  tone,
}: {
  label: string;
  children: ReactNode;
  tone?: "ok" | "warn" | "danger" | "accent";
}) {
  return (
    <div className="min-w-0">
      <div className="animus-eyebrow text-[9px] tracking-[0.18em]">{label}</div>
      <div
        className={cn(
          "mt-1 truncate font-mono text-[12px] leading-none tabular-nums",
          tone === "ok"
            ? "text-lab-ok"
            : tone === "warn"
              ? "text-lab-warn"
              : tone === "danger"
                ? "text-lab-danger"
                : tone === "accent"
                  ? "text-lab-accent-bright"
                  : "text-lab-text-dim",
        )}
      >
        {children}
      </div>
    </div>
  );
}

export default function ServerPage() {
  // No live subscription here: the form must not re-render on every sample. The live
  // endpoint panel subscribes on its own and is the one place that says what serves.
  const engines = useServeEngines();
  const [engineName, setEngineName] = useState("vllm");
  const eng = engines.find((e) => e.name === engineName);
  const engLabel = eng?.label ?? engineName;
  const has = (field: string) => engineHas(eng, field);
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
  const [mtpMoeBackend, setMtpMoeBackend] = useState("");
  const [dockerEnv, setDockerEnv] = useState("");
  const [extra, setExtra] = useState("");
  const [download, setDownload] = useState(false);
  const [rec, setRec] = useState<ServeRecommend | null>(null);
  const [recBusy, setRecBusy] = useState(false);
  const [recError, setRecError] = useState<string | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  const [startBusy, setStartBusy] = useState(false);
  const [advOpen, setAdvOpen] = useState(false);
  const [formFlash, setFormFlash] = useState<string | null>(null);
  const [appliedExample, setAppliedExample] = useState<string | null>(null);
  const jobPanelRef = useRef<HTMLDivElement>(null);

  const jobWatch = useJobWatch();
  const { logs, message: jobMsg, progress: jobProgress } = jobWatch.job;
  const jobStatus = jobWatch.job.status ?? "";
  const jobRunning = jobWatch.running;

  // Static presets: fetched once, not carried on the 1 s live snapshot.
  // null while loading: the section keeps its place (skeleton cards) so the steps below
  // neither jump down nor renumber when the answer lands.
  const [examples, setExamples] = useState<Record<string, ServeExample> | null>(null);
  const [modelHints, setModelHints] = useState<string[]>([]);
  useEffect(() => {
    api
      .serveExamples()
      .then((r) => {
        setExamples(r.examples || {});
        setModelHints(r.presets || []);
      })
      .catch(() => setExamples({}));
  }, []);
  /**
   * Spine numbering. The presets block is conditional, so hardcoded numerals
   * render 01 → 02 → 04 when no presets exist — a skipped step reads as a
   * missing section. Derive the sequence from what is actually on screen.
   */
  const hasPresets = examples == null || Object.keys(examples).length > 0;
  const step = (() => {
    let n = 0;
    const next = () => String(++n).padStart(2, "0");
    return {
      target: next(),
      presets: hasPresets ? next() : "",
      flags: next(),
      launch: next(),
      job: next(),
    };
  })();

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
    jobWatch.track(jobId);
    // Feedback is below the fold on Serve — pull the dock into view on start
    requestAnimationFrame(() => {
      jobPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  }

  // Re-attach Job panel only to a *live* job (not orphaned sqlite rows)
  useEffect(() => {
    let cancelled = false;
    const STALE_MS = 30 * 60 * 1000;
    api
      .jobs()
      .then(async (list) => {
        if (cancelled) return;
        // Serve and stop jobs dock here; agentic suites dock on Evals.
        const active = list.find(
          (j) => (j.status === "running" || j.status === "queued") && (j.kind === "serve" || j.kind === "stop"),
        );
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
    setMtpMoeBackend(String(c.mtp_moe_backend ?? ""));
    setLoadFormat(String(c.load_format || ""));
    setChunkedPrefill(!!c.enable_chunked_prefill);
    setPrefixCaching(!!c.enable_prefix_caching);
    if (c.image) setImage(String(c.image));
    if (c.util != null) setUtil(String(c.util));
    if (c.max_model_len != null) setMaxLen(String(c.max_model_len));
    if (c.port != null) setPort(String(c.port));
  }

  /** Switch engine: its default port replaces the old engine's default (never a port you typed). */
  function selectEngine(name: string) {
    const next = engines.find((e) => e.name === name);
    if (next && (!port.trim() || port === String(eng?.default_port ?? 8000))) setPort(String(next.default_port));
    setEngineName(name);
    setRec(null);
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

  const clearJobPanel = jobWatch.clear;

  function applyRecipeConfig(cfg: Record<string, unknown> | undefined) {
    if (!cfg) return;
    // Merge recipe onto current form without wiping envelope fields unless set
    applyConfig({ ...cfg, model: model || cfg.model });
  }

  function clearForm() {
    setModel("");
    setUtil("");
    setMaxLen("");
    setPort(String(eng?.default_port ?? 8000));
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
    setMtpMoeBackend("");
    setDockerEnv("");
    setExtra("");
    setDownload(false);
    setRec(null);
    setRecError(null);
    setStartError(null);
  }

  async function autoConfigure() {
    if (!model.trim()) {
      setRecError("Enter a model id first (e.g. unsloth/Qwen3.6-35B-A3B-NVFP4)");
      return;
    }
    setRecBusy(true);
    setRecError(null);
    try {
      const r = await api.recommendServe(model.trim(), true, engineName);
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
      engine: engineName,
      port: parseInt(port, 10) || eng?.default_port || 8000,
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
      mtp_moe_backend: mtpMoeBackend.trim(),
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
    } catch (e) {
      setStartError(e instanceof Error ? e.message : String(e));
    }
  }

  const confTone =
    rec?.confidence === "high" ? "ok" : rec?.confidence === "medium" ? "warn" : "muted";

  // Presentation only — mirrors the existing disabled expression exactly so the
  // launch control can explain WHY it is not ready instead of just dimming.
  const hasModel = !!model.trim();
  const startDisabledReason = !hasModel
    ? "Enter a model id first"
    : jobRunning
      ? "Wait for the current job to finish"
      : rec?.serve_blocked
        ? "Weights do not fit this cluster — add nodes or pick a smaller checkpoint"
        : null;

  return (
    <div className="space-y-4 lab-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Serve</h1>
          <p className="page-sub">
            Start a serve container — vLLM, SGLang, llama.cpp or TensorFold. Auto-configure reads the
            live model card and sizes memory; advanced flags stay folded until you need them.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Btn variant="ghost" size="sm" onClick={clearForm}>
            Clear form
          </Btn>
        </div>
      </div>

      <div className="space-y-4">
          <div className="grid gap-3 lg:grid-cols-3">
            <Panel className="lg:col-span-2">
              <div className="space-y-3.5 p-4">
                <Seq
                  n={step.target}
                  label="Target"
                  hint="hugging face id · live card lookup"
                  action={
                    <Badge tone={hasModel ? "accent" : "muted"}>
                      {hasModel ? "model set" : "no target"}
                    </Badge>
                  }
                />
                <div className="animus-rule" aria-hidden />
                {engineName === "vllm" ? (
                  <p className="text-[11px] leading-relaxed text-lab-muted">
                    Fetches the live model card + config from huggingface.co, then researches
                    Unsloth docs, NVIDIA playbooks, and GitHub vLLM recipes even when the card
                    has no links. Scores every{" "}
                    <code className="font-mono text-[10.5px] text-lab-text-dim">vllm serve</code>{" "}
                    recipe, applies checkpoint safety (e.g. strips flashinfer_b12x on mixed FP8
                    MoE), and sizes the hardware envelope automatically.
                  </p>
                ) : (
                  <p className="text-[11px] leading-relaxed text-lab-muted">
                    Fetches the model card + config from huggingface.co, sizes weights, tensor
                    parallel and memory with the placement engine, and shows the exact{" "}
                    {engLabel} command line for every rank before anything launches.
                  </p>
                )}
                <div className="flex flex-wrap items-end gap-3">
                  <div className="w-full sm:w-40">
                    <EngineSelect engines={engines} value={engineName} onChange={selectEngine} disabled={jobRunning} />
                  </div>
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
                </div>
                <EngineSummary engine={eng} />
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
                  <div className="animus-chamfer-sm animus-bracketed relative border border-lab-border bg-lab-editor p-3.5 text-xs space-y-2.5 before:top-[3px]! before:left-[3px]! after:right-[3px]! after:bottom-[3px]!">
                    <div className="animus-eyebrow text-[9px] tracking-[0.2em]">
                      Auto-config result
                    </div>
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
                      className="block break-all font-mono text-[11px] text-lab-accent-bright underline-offset-2 hover:underline"
                      href={rec.card_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {rec.card_url}
                    </a>
                  )}
                  {rec.notes && (
                    <p className="text-[11px] leading-relaxed text-lab-muted">{rec.notes}</p>
                  )}
                  {(rec.processes || []).length > 0 && (
                    <div className="space-y-1.5">
                      <div className="animus-eyebrow text-[9px] tracking-[0.2em]">
                        {rec.processes!.length > 1 ? `Command per rank (${rec.processes!.length})` : "Command"}
                      </div>
                      {rec.processes!.map((p) => (
                        <pre
                          key={String(p.rank)}
                          className="max-h-32 overflow-auto whitespace-pre-wrap break-all rounded-[2px] border border-lab-border-subtle bg-lab-panel/60 p-2 font-mono text-[10.5px] leading-relaxed text-lab-text-dim"
                        >
                          {p.rank != null ? `# rank ${p.rank}${p.node ? ` on ${p.node}` : ""}\n` : ""}
                          {p.argv}
                        </pre>
                      ))}
                    </div>
                  )}
                  {(rec.warnings || []).length > 0 && (
                    <ul className="space-y-1 border-l-2 border-l-lab-warn pl-3 text-[11px] leading-relaxed text-lab-warn">
                      {rec.warnings.map((w, i) => (
                        <li key={i}>{w}</li>
                      ))}
                    </ul>
                  )}
                  {(rec.card_recipes || []).length > 0 && (
                    <details open className="text-lab-muted">
                      <summary className="animus-eyebrow cursor-pointer text-[10px] text-lab-text-dim hover:text-lab-text">
                        Card recipes ({rec.card_recipes!.length}) — click Apply to try another
                      </summary>
                      <ul className="mt-2 space-y-2">
                        {rec.card_recipes!.map((cr, i) => (
                          <li
                            key={i}
                            className={cn(
                              "animus-notch border-l-2 px-2.5 py-2 font-mono text-[11px]",
                              cr.selected
                                ? "border border-l-[color:var(--color-lab-accent)] border-[color:var(--animus-accent-edge)] bg-[color:var(--animus-accent-wash)]"
                                : "border border-l-[color:var(--animus-hairline)] border-lab-border-subtle",
                            )}
                          >
                            <div className="flex flex-wrap items-center gap-2 text-lab-text">
                              <span className="tabular-nums">score {cr.score}</span>
                              {cr.selected && <Badge tone="ok">selected</Badge>}
                              {cr.section && (
                                <span className="font-[family-name:var(--font-display)] text-[10px] uppercase tracking-[0.14em] text-lab-muted">
                                  {cr.section}
                                </span>
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
                            <div className="mt-1.5 whitespace-pre-wrap break-all text-lab-text-dim">
                              {cr.raw}
                            </div>
                            {(cr.reasons || []).length > 0 && (
                              <ul className="mt-1.5 list-disc space-y-0.5 pl-4 font-sans text-[10px] text-lab-muted">
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
                      <p className="mt-2 font-sans text-[10px] leading-relaxed text-lab-muted">
                        &quot;Apply raw recipe&quot; fills form fields from that card snippet only —
                        checkpoint safety / envelope are not re-run. Prefer the selected recipe
                        (already safety-merged) unless you know what you&apos;re doing.
                      </p>
                    </details>
                  )}
                  <details className="text-lab-muted">
                    <summary className="animus-eyebrow cursor-pointer text-[10px] text-lab-text-dim hover:text-lab-text">
                      Why these flags
                    </summary>
                    <ul className="mt-2 list-disc space-y-1 pl-4 text-[11px] leading-relaxed">
                      {(rec.rationale || []).map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                    {(rec.sources || []).length > 0 && (
                      <div className="mt-2.5">
                        <div className="animus-eyebrow text-[9px] tracking-[0.2em]">
                          Fetched from
                        </div>
                        <ul className="mt-1.5 space-y-1">
                          {rec.sources!.map((s, i) => (
                            <li key={i} className="font-mono text-[10.5px] break-all">
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
              </div>
            </Panel>

            <LiveEndpoint onStop={() => void stop()} stopBlocked={startBusy} />
          </div>

          {hasPresets && (
            <Panel>
              <div className="space-y-3.5 p-4">
                <Seq
                  n={step.presets}
                  label="Proven presets"
                  hint="fills the form only — nothing launches"
                  action={
                    appliedExample ? (
                      <Badge tone="accent">applied</Badge>
                    ) : (
                      <Unset>None applied</Unset>
                    )
                  }
                />
                <div className="animus-rule" aria-hidden />
                <p className="text-[11px] leading-relaxed text-lab-muted">
                  Static hardware-proven presets. Prefer Auto-configure for newest HF cards.
                </p>
                {formFlash && (
                  <Callout tone="ok" title="Form updated">
                    {formFlash}
                  </Callout>
                )}
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  {examples == null &&
                    [0, 1].map((i) => <Skeleton key={i} className="h-[54px] w-full" />)}
                  {Object.entries(examples ?? {}).map(([k, ex]) => {
                    const selected =
                      appliedExample === k || appliedExample === (ex.label || ex.model);
                    return (
                      <button
                        key={k}
                        type="button"
                        onClick={() => applyExample(ex, k)}
                        aria-pressed={selected}
                        className={cn(
                          "animus-notch relative border border-l-2 px-3 py-2.5 text-left text-xs transition-colors",
                          // The chamfer clips the global focus ring — carry an inset one.
                          "focus-visible:outline-none! focus-visible:shadow-[inset_0_0_0_2px_var(--color-lab-line)]!",
                          selected
                            ? "border-[color:var(--animus-accent-edge)] border-l-[color:var(--color-lab-accent)] bg-[color:var(--animus-accent-wash)] text-lab-text"
                            : "border-lab-border-subtle border-l-[color:var(--animus-hairline)] text-lab-muted hover:border-lab-border hover:border-l-lab-line hover:bg-lab-hover/60 hover:text-lab-text",
                        )}
                      >
                        <div className="flex items-center gap-2">
                          <span className="truncate font-[family-name:var(--font-display)] text-[12px] font-semibold uppercase tracking-[0.1em] text-lab-text">
                            {ex.label || k}
                          </span>
                          {selected && (
                            <span
                              aria-hidden
                              className="ml-auto h-1.5 w-1.5 shrink-0 rotate-45 bg-lab-accent"
                            />
                          )}
                        </div>
                        {ex.model ? (
                          <div className="mt-1 truncate font-mono text-[10.5px] text-lab-muted">
                            {ex.model}
                          </div>
                        ) : (
                          <div className="mt-1">
                            <Unset>No model pinned</Unset>
                          </div>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            </Panel>
          )}

          <Panel>
            <div className="space-y-4 p-4">
              <Seq
                n={step.flags}
                label="Engine flags"
                hint="envelope first · advanced on demand"
                action={
                  advancedHasValues ? (
                    <Badge tone="accent">advanced set</Badge>
                  ) : (
                    <Unset>Defaults</Unset>
                  )
                }
              />
              <div className="animus-rule" aria-hidden />
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {has("util") && (
                  <Field
                    label={engineName === "sglang" ? "mem-fraction-static" : "gpu-memory-utilization"}
                    htmlFor="serve-util"
                  >
                    <Input
                      id="serve-util"
                      value={util}
                      onChange={(e) => setUtil(e.target.value)}
                      placeholder={engineName === "vllm" ? "0.85" : "engine default"}
                    />
                  </Field>
                )}
                {has("max_model_len") && (
                  <Field label={CONTEXT_FLAG[engineName] ?? "max-model-len"} htmlFor="serve-maxlen">
                    <Input
                      id="serve-maxlen"
                      value={maxLen}
                      onChange={(e) => setMaxLen(e.target.value)}
                      placeholder={engineName === "vllm" ? "262144" : "engine default"}
                    />
                  </Field>
                )}
                <Field label="Port" htmlFor="serve-port">
                  <Input id="serve-port" value={port} onChange={(e) => setPort(e.target.value)} />
                </Field>
                {has("tensor_parallel_size") && (
                  <Field
                    label="tensor-parallel-size"
                    htmlFor="serve-tp"
                    hint={eng?.max_tp ? `≤ ${eng.max_tp} · one rank per Spark` : undefined}
                  >
                    <Input
                      id="serve-tp"
                      value={tpSize}
                      onChange={(e) => setTpSize(e.target.value)}
                      placeholder="1"
                    />
                  </Field>
                )}
              </div>

              <div className="animus-notch border border-l-2 border-lab-border-subtle border-l-[color:var(--animus-hairline)] bg-lab-editor/50 px-2 py-1">
                <CheckboxRow id="serve-download" checked={download} onChange={setDownload}>
                  Download weights first (hf download) before docker start
                </CheckboxRow>
              </div>

              <details
                className="group animus-chamfer-sm border border-lab-border-subtle bg-lab-editor/40 open:bg-lab-editor/60"
                open={advOpen}
                onToggle={(e) => setAdvOpen((e.currentTarget as HTMLDetailsElement).open)}
              >
                <summary className="cursor-pointer list-none px-3.5 py-2.5 marker:content-none [&::-webkit-details-marker]:hidden">
                  <span className="inline-flex items-center gap-2">
                    <span
                      aria-hidden
                      className="font-mono text-[10px] leading-none text-lab-line transition-transform group-open:rotate-90"
                    >
                      ▸
                    </span>
                    <span className="animus-eyebrow text-[10px] text-lab-text-dim">
                      Advanced flags
                    </span>
                    <Tick className="hidden sm:block" />
                    <span className="hidden text-[11px] leading-none text-lab-muted sm:inline">
                      image · quant · tools · MTP · docker env
                    </span>
                  </span>
                </summary>
                <div className="grid gap-3 border-t border-lab-border-subtle p-3.5 sm:grid-cols-2 lg:grid-cols-3">
                  <Field label={`${engLabel} image`} htmlFor="adv-image" hint={eng?.image_env ? `default from ${eng.image_env}` : undefined}>
                    <Input
                      id="adv-image"
                      value={image}
                      onChange={(e) => setImage(e.target.value)}
                      placeholder={eng?.default_image || "vllm/vllm-openai:v0.27.1"}
                    />
                  </Field>
                  {has("quantization") && (
                  <Field label="--quantization" htmlFor="adv-quant">
                    <Input
                      id="adv-quant"
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
                  )}
                  {has("kv_cache_dtype") && (
                  <Field label={engineName === "tensorfold" ? "--kv-dtype" : "--kv-cache-dtype"} htmlFor="adv-kv">
                    <Input
                      id="adv-kv"
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
                  )}
                  {has("moe_backend") && (
                  <Field label="--moe-backend" htmlFor="adv-moe">
                    <Input
                      id="adv-moe"
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
                  )}
                  {has("max_num_seqs") && (
                    <Field label={engineName === "sglang" ? "--max-running-requests" : "--max-num-seqs"} htmlFor="adv-seqs">
                      <Input
                        id="adv-seqs"
                        value={maxNumSeqs}
                        onChange={(e) => setMaxNumSeqs(e.target.value)}
                        placeholder="4"
                      />
                    </Field>
                  )}
                  {has("load_format") && (
                    <Field label="--load-format" htmlFor="adv-loadfmt">
                      <Input
                        id="adv-loadfmt"
                        value={loadFormat}
                        onChange={(e) => setLoadFormat(e.target.value)}
                      />
                    </Field>
                  )}
                  {has("tool_call_parser") && (
                    <Field label="--tool-call-parser" htmlFor="adv-toolparser">
                      <Input
                        id="adv-toolparser"
                        value={toolCallParser}
                        onChange={(e) => setToolCallParser(e.target.value)}
                        placeholder="qwen3_coder"
                      />
                    </Field>
                  )}
                  {has("reasoning_parser") && (
                    <Field label="--reasoning-parser" htmlFor="adv-reasonparser">
                      <Input
                        id="adv-reasonparser"
                        value={reasoningParser}
                        onChange={(e) => setReasoningParser(e.target.value)}
                        placeholder="qwen3"
                      />
                    </Field>
                  )}
                  {has("mtp") && (
                    <Field label="MTP speculative tokens" htmlFor="adv-mtptokens">
                      <Input
                        id="adv-mtptokens"
                        value={mtpTokens}
                        onChange={(e) => setMtpTokens(e.target.value)}
                        disabled={!mtp}
                      />
                    </Field>
                  )}
                  {(has("trust_remote_code") || has("mtp")) && (
                  <div className="space-y-1.5 sm:col-span-2 lg:col-span-3">
                    <div className="animus-eyebrow text-[9px] tracking-[0.2em]">Toggles</div>
                    <div className="animus-rule" aria-hidden />
                    <div className="grid gap-x-4 sm:grid-cols-2 lg:grid-cols-3">
                      {has("trust_remote_code") && (
                        <CheckboxRow
                          id="adv-trc"
                          checked={trustRemoteCode}
                          onChange={setTrustRemoteCode}
                        >
                          <span className="font-mono text-[11px]">--trust-remote-code</span>
                        </CheckboxRow>
                      )}
                      {has("enable_auto_tool_choice") && (
                        <CheckboxRow
                          id="adv-autotool"
                          checked={enableAutoTool}
                          onChange={setEnableAutoTool}
                        >
                          <span className="font-mono text-[11px]">--enable-auto-tool-choice</span>
                        </CheckboxRow>
                      )}
                      {has("enable_chunked_prefill") && (
                        <CheckboxRow
                          id="adv-chunked"
                          checked={chunkedPrefill}
                          onChange={setChunkedPrefill}
                        >
                          <span className="font-mono text-[11px]">--enable-chunked-prefill</span>
                        </CheckboxRow>
                      )}
                      {has("enable_prefix_caching") && (
                        <CheckboxRow
                          id="adv-prefix"
                          checked={prefixCaching}
                          onChange={setPrefixCaching}
                        >
                          <span className="font-mono text-[11px]">--enable-prefix-caching</span>
                        </CheckboxRow>
                      )}
                      {has("mtp") && (
                        <CheckboxRow id="adv-mtp" checked={mtp} onChange={setMtp}>
                          <span className="font-mono text-[11px]">
                            MTP (--speculative-config method=mtp)
                          </span>
                        </CheckboxRow>
                      )}
                    </div>
                  </div>
                  )}
                  <div className="sm:col-span-2 lg:col-span-3">
                    <Field label="Docker env (KEY=VALUE per line)" htmlFor="adv-dockerenv">
                      <textarea
                        id="adv-dockerenv"
                        className={cn(inputCls, "min-h-[72px] font-mono text-xs")}
                        value={dockerEnv}
                        onChange={(e) => setDockerEnv(e.target.value)}
                        placeholder={"CUTE_DSL_ARCH=sm_121a\n# only lines you type are passed"}
                      />
                    </Field>
                  </div>
                  <div className="sm:col-span-2 lg:col-span-3">
                    <Field label={`Extra ${engLabel} flags`} htmlFor="adv-extra">
                      <textarea
                        id="adv-extra"
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

          {/*
            LAUNCH — the one primary CTA on this surface. Ready when a
            model is set and no job is in flight; otherwise deliberately
            not ready, with the reason spelled out both in the title attribute
            and on the plate itself.
          */}
          <section
            className={cn(
              "animus-chamfer animus-bracketed relative border px-4 py-4 transition-colors",
              startDisabledReason
                ? "border-lab-border-subtle bg-lab-panel/60"
                : "border-[color:var(--animus-accent-edge)] bg-[color:var(--animus-accent-wash)]",
            )}
          >
            <Seq n={step.launch} label="Launch" hint={`starts a real ${engLabel} container`} />
            <div className="animus-rule mt-3" aria-hidden />
            <div className="mt-3.5 flex flex-wrap items-center justify-between gap-x-5 gap-y-3">
              <div className="min-w-0 space-y-1.5">
                <div className="flex items-center gap-2">
                  <span
                    aria-hidden
                    className={cn(
                      "h-2 w-2 rotate-45",
                      startDisabledReason
                        ? "bg-transparent shadow-[inset_0_0_0_1px_var(--color-lab-muted)]"
                        : "bg-lab-accent shadow-[0_0_10px_var(--animus-accent-edge)]",
                    )}
                  />
                  <span
                    className={cn(
                      "font-[family-name:var(--font-display)] text-[12px] font-semibold uppercase leading-none tracking-[0.18em]",
                      startDisabledReason ? "text-lab-muted" : "text-lab-accent-bright",
                    )}
                  >
                    {startBusy ? "Starting" : startDisabledReason ? "Not ready" : "Ready"}
                  </span>
                </div>
                <p className="max-w-lg text-[11px] leading-relaxed text-lab-muted">
                  {startDisabledReason
                    ? `${startDisabledReason}.`
                    : "Stops running serve containers first, then boots the configured serve. Live output docks below."}
                </p>
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-[10.5px] tabular-nums text-lab-muted">
                  <span className="truncate">{model.trim() || "no target"}</span>
                  <Tick />
                  <span>{engLabel}</span>
                  <Tick />
                  <span>:{port || eng?.default_port || "8000"}</span>
                </div>
              </div>
              {/* pointer-events-none on a disabled button swallows hover, so the
                  wrapper carries the tooltip too. */}
              <span title={startDisabledReason ?? undefined}>
                <Btn
                  onClick={() => void start()}
                  disabled={!!startDisabledReason}
                  loading={startBusy}
                  title={startDisabledReason ?? undefined}
                  className="h-11 px-7 text-[13px] tracking-[0.2em]"
                >
                  Start serve
                </Btn>
              </span>
            </div>
          </section>
      </div>

      {/* Serve and stop output */}
      <div
        ref={jobPanelRef}
        id="serve-job-dock"
        className={cn(
          // Both themes: elevation comes from the per-theme panel shadow token,
          // never a hardcoded black.
          jobRunning && "sticky bottom-3 z-10 shadow-[var(--animus-panel-shadow-hover)]",
        )}
      >
        <Panel
          className={cn(
            jobRunning &&
              "border-[color:var(--animus-accent-edge)] shadow-[0_0_0_1px_var(--animus-accent-edge)]",
          )}
        >
          <div className="space-y-3.5 p-4">
            <Seq
              n={step.job}
              label="Job dock"
              hint="serve · stop output"
              action={
                <>
                  {jobRunning && (
                    <Btn
                      variant="danger"
                      size="sm"
                      onClick={() => void jobWatch.cancel().catch((e) => setStartError(String(e)))}
                      title="Cancel the running job"
                    >
                      Cancel
                    </Btn>
                  )}
                  {(jobRunning || jobStatus || logs) && (
                    <Btn
                      variant="ghost"
                      size="sm"
                      onClick={clearJobPanel}
                      title="Clear job panel"
                    >
                      Dismiss
                    </Btn>
                  )}
                  <span aria-live="polite" aria-atomic="true" className="inline-flex">
                    <Badge
                      tone={
                        jobWatch.done
                          ? "ok"
                          : jobWatch.failed
                            ? "danger"
                            : jobRunning
                              ? "accent"
                              : "muted"
                      }
                      dot={jobRunning}
                    >
                      {jobStatus || "idle"}
                    </Badge>
                  </span>
                </>
              }
            />
            <div className="animus-rule" aria-hidden />
            {jobRunning || jobStatus || logs ? (
              <>
                <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
                  <Telem
                    label="State"
                    tone={
                      jobWatch.done
                        ? "ok"
                        : jobWatch.failed
                          ? "danger"
                          : jobRunning
                            ? "accent"
                            : undefined
                    }
                  >
                    {jobStatus || "idle"}
                  </Telem>
                  <Telem label="Progress">
                    {jobProgress > 0
                      ? `${Math.round((jobProgress || 0) * 100)}%`
                      : jobRunning
                        ? "—— %"
                        : "0%"}
                  </Telem>
                  <Telem label="Stream" tone={jobRunning ? "accent" : undefined}>
                    {jobRunning ? "live" : "closed"}
                  </Telem>
                  <Telem label="Log bytes">{logs.length.toLocaleString()}</Telem>
                </div>
                <ProgressBar
                  value={Math.round((jobProgress || 0) * 100)}
                  indeterminate={jobRunning && !(jobProgress > 0)}
                  label={jobMsg || (jobRunning ? "Working…" : "Last job")}
                />
                <LogView
                  text={logs}
                  live={jobRunning}
                  empty="Output appears here when you start or stop a serve."
                />
              </>
            ) : (
              <EmptyState title="No job">
                Start or stop a serve — its live log streams here.
              </EmptyState>
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}

/**
 * What is serving right now and how to reach it — its own store subscription, so a
 * live sample re-renders this panel, not the form beside it. Stop lives here, next
 * to what it stops, behind a second click: a stray click on a 15–30 min NVFP4 load
 * costs the operator half an hour.
 */
function LiveEndpoint({ onStop, stopBlocked }: { onStop: () => void; stopBlocked: boolean }) {
  const { loading, healthy, serve, engineError } = useLabStatusStore(
    useShallow((s) => ({
      loading: s.loading,
      healthy: serveHealthy(s.status),
      serve: s.status?.serve ?? null,
      engineError: s.engineError,
    })),
  );
  // Frozen data is never shown as live: the last snapshot dims, with its age.
  const stale = useStale();
  const [armedAt, setArmedAt] = useState<number | null>(null);
  const confirming = armedAt != null;
  useEffect(() => {
    if (armedAt == null) return;
    const t = setTimeout(() => setArmedAt(null), CONFIRM_WINDOW_MS);
    return () => clearTimeout(t);
  }, [armedAt]);

  const containers = serve?.containers ?? [];
  const up = containers.filter((c) => c.status.includes("Up"));
  const modelShort = healthy ? (serve?.model_id || "").split("/").pop() || null : null;
  const tightest = tightestNode(serve?.cluster?.nodes);
  const freeGib = tightest?.available_gib ?? serve?.hardware?.available_gib ?? null;
  const pressure = tightest?.mem_pressure ?? serve?.headroom ?? null;
  const bases = hermesBases(serve?.base_url, "127.0.0.1");
  const env = bases && healthy && serve?.model_id ? `OPENAI_BASE_URL=${bases.local}\nOPENAI_API_KEY=local\nOPENAI_MODEL=${serve.model_id}` : null;
  const [flash, copy] = useCopy();
  const canStop = up.length > 0 && !stopBlocked;

  return (
    <Panel>
      <div className="space-y-3.5 p-4">
        <Seq
          n="——"
          label="Live endpoint"
          action={
            <span className="flex items-center gap-2">
              {stale && <LiveAge />}
              <SyncRing
                state={loading ? null : engineError ? "offline" : stale ? "stale" : healthy ? "serving" : "idle"}
                label={
                  loading
                    ? "Checking endpoint"
                    : engineError
                      ? `Serve-engine not answering (${engineError})`
                      : stale
                        ? "Live data not updating"
                        : healthy
                          ? "Endpoint healthy"
                          : "Endpoint idle"
                }
              />
            </span>
          }
        />
        <div className="animus-rule" aria-hidden />
        {loading ? (
          // about the panel's height while serving, so the steps below do not jump on the first sample
          <div className="min-h-[240px] space-y-3" aria-busy="true" aria-label="Loading live status">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-[80%]" />
            <Skeleton className="h-4 w-[60%]" />
          </div>
        ) : (
          <>
            <div className={cn("space-y-3.5 transition-opacity duration-300", stale && "opacity-60")}>
              <dl>
                <Readout label="Served model" value={healthy ? serve?.model_id : null} unset="No model" />
                <Readout
                  label="Engine"
                  value={
                    healthy && serve?.engine?.name
                      ? [engineLabel(serve.engine.name), serve.engine.version && `v${serve.engine.version}`]
                          .filter(Boolean)
                          .join(" ")
                      : undefined
                  }
                  unset="Unknown"
                />
                <Readout label="OpenAI base URL">
                  {bases ? (
                    <span className="inline-flex items-center gap-1.5">
                      {bases.local}
                      <CopyButton text={bases.local} label="Base URL" />
                    </span>
                  ) : (
                    <Unset>Not set</Unset>
                  )}
                </Readout>
                <Readout
                  label="Memory free"
                  value={
                    freeGib != null
                      ? `${freeGib.toFixed(1)} GiB${tightest && (serve?.cluster?.nodes?.length ?? 0) > 1 ? ` on ${tightest.id}` : ""}${pressure && pressure !== "ok" ? ` · ${pressure}` : ""}`
                      : undefined
                  }
                  unset="Unknown"
                />
              </dl>
              {env && (
                <div className="flex flex-wrap items-center gap-2">
                  <Btn variant="secondary" size="sm" onClick={() => copy("Hermes env", env)} title={env}>
                    Copy Hermes env
                  </Btn>
                  <a href="/connect" className="font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase tracking-[0.14em] text-lab-accent-bright hover:text-lab-accent">
                    More ways to connect →
                  </a>
                  {flash && <Eyebrow className="text-lab-ok">{flash}</Eyebrow>}
                </div>
              )}
              <div className="space-y-1.5">
                <div className="animus-eyebrow text-[9px] tracking-[0.2em]">Containers</div>
                {containers.length === 0 && (
                  <div className="animus-notch border border-l-2 border-lab-border-subtle border-l-[color:var(--animus-hairline)] px-2.5 py-2">
                    <Unset>No serve containers</Unset>
                  </div>
                )}
                {containers.map((c) => (
                  <div
                    key={c.name}
                    className={cn(
                      "animus-notch flex items-center justify-between gap-2 border border-l-2 border-lab-border-subtle px-2.5 py-1.5 text-[11px]",
                      c.status.includes("Up") ? "border-l-lab-ok" : "border-l-[color:var(--animus-hairline)]",
                    )}
                  >
                    <span className="truncate font-mono text-lab-text-dim">{c.name}</span>
                    <Badge tone={c.status.includes("Up") ? "ok" : "muted"}>{c.status}</Badge>
                  </div>
                ))}
              </div>
            </div>
            {up.length > 0 && (
              <div className="flex flex-wrap items-center gap-2 border-t border-[color:var(--animus-hairline)] pt-3">
                <Btn
                  variant={confirming ? "danger" : "secondary"}
                  size="sm"
                  disabled={!canStop}
                  onClick={() => {
                    const now = Date.now();
                    const step = confirmStep(armedAt, now);
                    if (step === "ignore") return; // the second click of a double-click
                    if (step === "arm") {
                      setArmedAt(now);
                      return;
                    }
                    setArmedAt(null);
                    onStop();
                  }}
                  title={stopBlocked ? "Wait for the start request to register" : "Stops every serve container (any engine) on every node (two clicks)"}
                >
                  {confirming ? "Confirm stop" : modelShort ? `Stop ${modelShort}` : "Stop containers"}
                </Btn>
                {confirming && (
                  <span className="text-[11px] text-lab-warn">Click again within 4 s — reloading takes minutes.</span>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </Panel>
  );
}
