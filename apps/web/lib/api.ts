import { getClientToken, tokenQuery } from "./auth-token";
import type {
  StreamPack,
  StreamRunRequest,
  StreamRunRow,
  StreamRunSnapshot,
} from "./stream-run-types";

const BASE = "";

export function lailTokenHeader(): Record<string, string> {
  const t = getClientToken();
  return t ? { "X-Lail-Token": t } : {};
}

/** A non-2xx reply. `message` stays the body text (what pages already show). */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    body: string,
    /** Parsed body when it was JSON, e.g. the controller's {error, message}. */
    readonly json: { error?: string; message?: string } | null,
  ) {
    super(body);
    this.name = "ApiError";
  }
}

/** The controller's JSON error body ({error, message, run_id}) of a failed call, or just its text. */
export function parseApiError(e: unknown): { error?: string; message: string; run_id?: string } {
  const msg = e instanceof Error ? e.message : String(e);
  try {
    const j = JSON.parse(msg) as { error?: string; message?: string; run_id?: string };
    return { error: j.error, message: j.message || msg, run_id: j.run_id };
  } catch {
    return { message: msg };
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...lailTokenHeader(),
      ...(init?.headers || {}),
    },
  });
  if (!r.ok) {
    const body = await r.text();
    let json: ApiError["json"] = null;
    try {
      json = JSON.parse(body);
    } catch {
      /* not JSON */
    }
    throw new ApiError(r.status, body || r.statusText, json);
  }
  return r.json();
}

export type ClusterNode = {
  id: string;
  label: string;
  role?: string;
  local?: boolean;
  online?: boolean;
  /** offline = no ping; unreachable = pings but ssh failed; stray = old serve container, no endpoint */
  state?: "offline" | "unreachable" | "idle" | "loading" | "stray" | "serving" | "serving_worker" | string;
  /** 0 on the serving head and N on headless TP worker N; the endpoint rate is on serve.metrics, not here. */
  tp_rank?: number | null;
  probe_error?: string | null;
  hostname?: string | null;
  lan_ip?: string | null;
  tailscale_ip?: string | null;
  qsfp_ip?: string | null;
  gpu_sku?: string | null;
  temperature_c?: number | null;
  gpu_util_pct?: number | null;
  power_w?: number | null;
  ram_gib?: number | null;
  /** /proc/meminfo MemAvailable, 0.01 GiB precision */
  available_gib?: number | null;
  swap_total_gib?: number | null;
  swap_used_gib?: number | null;
  /** GiB held by GPU compute processes (on GB10 UMA: the engine's slice of system RAM) */
  engine_reserved_gib?: number | null;
  mem_psi_full_avg10?: number | null;
  /** ok | tight | critical — judged against running out of RAM AND swap, not a fixed GiB line */
  mem_pressure?: "ok" | "tight" | "critical" | null;
  cpu?: string | null;
  cpu_util_pct?: number | null;
  soc_temp_c?: number | null;
  nvme_temp_c?: number | null;
  nic_temp_c?: number | null;
  /** server epoch ms of the telemetry (memory / temps / power) on this node */
  sampled_at?: number | null;
  /** server epoch ms of the slow inventory (containers, endpoint, rails) */
  inventory_at?: number | null;
  /** bytes/s per RoCE netdev from the RDMA port counters (null until two reads) */
  rail_rates?: Record<string, { tx_bps: number; rx_bps: number }> | null;
  /** why the ~1 s remote telemetry stream is down, when it is */
  telemetry_error?: string | null;
  endpoint_healthy?: boolean;
  model_id?: string | null;
  models?: Array<{ id?: string }>;
  containers?: Array<{ name: string; status: string; image: string }>;
  tensor_parallel_size?: number | null;
  ray_hint?: boolean;
  qsfp_if?: string | null;
  qsfp_carrier?: number | null;
  qsfp_speed_mbps?: number | null;
  roce_up_ifs?: string[];
  rails?: Array<{ if: string; ip: string; prefix: number; carrier?: number | null; speed_mbps?: number | null }>;
  ping?: { via?: string; ip?: string; ok?: boolean; rtt_ms?: number | null; error?: string | null } | null;
  vllm_url?: string | null;
  ssh_host?: string;
  /** GPU memory (C2 contract; nullable — UMA hosts may not report it). */
  gpu_mem_used_gib?: number | null;
  gpu_mem_total_gib?: number | null;
  memory_used_mib?: number | null;
  memory_total_mib?: number | null;
};

/**
 * Engine telemetry (C2 contract, additive on GET /api/status). Every field is
 * optional/nullable — the UI renders <Nil/> when a value is absent.
 */
export type EngineStatus = {
  /** Engine detected on the endpoint (owned_by / metrics prefix): vllm | sglang | llamacpp | tensorfold. */
  name?: string | null;
  kv_usage_pct?: number | null;
  requests_running?: number | null;
  requests_waiting?: number | null;
  block_size?: number | null;
  num_gpu_blocks?: number | null;
  kv_capacity_tokens?: number | null;
  max_model_len?: number | null;
  version?: string | null;
  prefix_cache_hit_rate?: number | null;
  preemptions_total?: number | null;
  sleep_state?: string | null;
  uptime_s?: number | null;
  flags_fingerprint?: string | null;
  flags?: string[] | null;
};

/** Live endpoint rates over the sampler window (each null when there is no reading). */
export type ServeMetrics = {
  requests_running?: number | null;
  requests_waiting?: number | null;
  /** per-stream decode tok/s over busy time (Σ inter-token latency); 0 = running, no token moved; null = idle */
  decode_tok_per_s?: number | null;
  /** all streams together: Δgenerated tokens / Δwall */
  throughput_tok_per_s?: number | null;
  /** computed prompt tok/s of requests that finished in the window */
  prefill_tok_per_s?: number | null;
  ttft_s?: number | null;
  spec_accept_rate?: number | null;
  spec_tokens_per_step?: number | null;
  spec_accept_rate_lifetime?: number | null;
  rate_window_s?: number | null;
  /** the previous busy period — show only as "last", never as live */
  last_burst?: { decode_tok_per_s?: number | null; tokens?: number; ended_at?: number } | null;
  /** latest non-null prefill_tok_per_s and when (epoch ms) — "last", never live */
  last_prefill?: { tok_per_s: number; at: number } | null;
  /** latest non-null ttft_s and when (epoch ms) — "last", never live */
  last_ttft?: { s: number; at: number } | null;
  /** server epoch ms of the /metrics scrape */
  sampled_at?: number | null;
};

export type ClusterStatus = {
  name?: string;
  updated_from?: string;
  error?: string;
  /** the first inventory has not finished yet (serve-engine just started) */
  pending?: boolean;
  nodes: ClusterNode[];
  fabric?: {
    ok?: boolean;
    note?: string;
    links?: Array<{
      from: string;
      to: string;
      via?: string;
      /** this side's RoCE interface — one link per rail */
      iface?: string | null;
      target_ip?: string;
      ok?: boolean;
      rtt_ms?: number | null;
      from_carrier?: number | null;
      to_carrier?: number | null;
      from_speed_mbps?: number | null;
      to_speed_mbps?: number | null;
      error?: string | null;
    }>;
  };
  summary?: {
    nodes_total?: number;
    nodes_online?: number;
    nodes_serving?: number;
    cluster_reachable?: boolean;
    fabric_ok?: boolean;
    healthy?: boolean;
    multi?: {
      mode?: string;
      model_id?: string | null;
      nodes_serving?: string[];
      tensor_parallel_hint?: number | null;
      fabric_ok?: boolean;
      message?: string;
      models_by_node?: Record<string, string | null | undefined>;
    };
  };
};

export type LabStatus = {
  controller: string;
  defaultBackend: string;
  defaultModel: string;
  openAiBase: string;
  backends: Record<string, { ok: boolean; url: string; error?: string }>;
  serve: {
    healthy?: boolean;
    base_url?: string;
    model_id?: string | null;
    /** Age of the serve-engine sampler snapshot (s). */
    stale_s?: number | null;
    models?: Array<{ id: string }>;
    /** The serving engine's version (vLLM /version, SGLang /server_info, llama.cpp /props). */
    version?: { version?: string | null } | null;
    /** The endpoint block: /metrics counters and live window rates from the serve-engine sampler. */
    metrics?: ServeMetrics | null;
    engine?: EngineStatus | null;
    hardware?: {
      gpu_sku?: string;
      ram_gib?: number;
      available_gib?: number | null;
      swap_total_gib?: number | null;
      swap_used_gib?: number | null;
      engine_reserved_gib?: number | null;
      cpu?: string;
      sampled_at?: number | null;
    };
    containers?: Array<{ name: string; status: string; image: string }>;
    /** worst node mem_pressure: ok | tight | critical */
    headroom?: string;
    error?: string;
    unreachable?: boolean;
    cluster?: ClusterStatus;
    /** server epoch ms when this snapshot was published */
    sampled_at_ms?: number | null;
  } | null;
};

export type ServeExample = {
  label?: string;
  model?: string;
  quantization?: string;
  kv_cache_dtype?: string;
  moe_backend?: string;
  trust_remote_code?: boolean;
  reasoning_parser?: string;
  tool_call_parser?: string;
  enable_auto_tool_choice?: boolean;
  max_num_seqs?: number | string;
  docker_env?: string[];
  extra_flags?: string;
  mtp?: boolean;
  notes?: string;
};

export type Job = {
  job_id: string;
  kind: string;
  status: string;
  progress: number;
  message: string;
  result: Record<string, unknown> | null;
  log_path: string | null;
};

export type RunRow = {
  run_id: string;
  created_at: string;
  kind: string;
  intent: string | null;
  model_id: string | null;
  summary: Record<string, unknown>;
  path: string;
};

export type ToolEvalBoardRow = {
  run_id: string;
  created_at?: string;
  model_id: string;
  model_short: string;
  final_score: number | null;
  rating?: string | null;
  preset?: string | null;
  total_scenarios?: number | null;
  total_points?: number | null;
  max_points?: number | null;
  deployability?: number | null;
  responsiveness?: number | null;
  safety_passed?: boolean;
  safety_warnings?: unknown[];
  categories: Array<{
    id?: string;
    label?: string;
    percent?: number;
    earned?: number;
    max?: number;
    pass?: number;
    partial?: number;
    fail?: number;
  }>;
  engine_image?: string | null;
  engine_version?: string | null;
  quant?: string | null;
  href: string;
};

export type LabArtifactRun = {
  id: string;
  kind: string;
  task_type: string;
  title: string;
  model_id: string;
  created_at: string;
  entry: string;
  tags: string[];
  brief?: string;
  eval_run_id?: string | null;
  /** Capability URL for the run's artifacts (no token needed; iframes cannot send one). */
  artifacts_url: string;
  play_url: string;
  gallery_url?: string;
  public_url?: string | null;
  task_fingerprint?: string;
  hermes?: { session?: string; source?: string } | null;
  share?: { public: boolean; slug: string | null };
  files?: string[];
  siblings?: LabArtifactRun[];
};

/** An engine Serve can launch (GET /api/serve/engines). */
export type ServeEngine = {
  name: string;
  label: string;
  default_port: number;
  default_image: string;
  image_env: string;
  /** Most TP ranks it runs here (null = one per node, no cap). */
  max_tp: number | null;
  /** Serve fields this engine translates into its own flags. */
  fields: string[];
  notes: string;
};

export type ServeRecommend = {
  model: string;
  mode: string;
  confidence: string;
  label?: string | null;
  notes?: string | null;
  card_url?: string | null;
  from_website?: boolean;
  hf_token_ok?: boolean;
  serve_blocked?: boolean;
  config: Record<string, unknown>;
  rationale: string[];
  warnings: string[];
  detected: Record<string, unknown>;
  topology?: {
    nodes?: number;
    nodes_used?: number;
    fabric_ok?: boolean;
    head_ip?: string | null;
    worker_ips?: string[];
    tensor_parallel_size?: number;
    pipeline_parallel_size?: number;
    weights_gib?: number | null;
    per_node_weights_gib?: number | null;
    util_computed?: number | null;
    fits?: boolean;
    node_ram_gib?: number | null;
    overlay?: string | null;
  };
  sources?: Array<{ kind: string; ref: string; notes?: string }>;
  /** Non-vLLM engines: the exact command line per process (rank null = single node). */
  argv?: string;
  processes?: Array<{ rank: number | null; node?: string; argv: string }>;
  card_recipes?: Array<{
    score: number;
    section?: string;
    raw: string;
    selected?: boolean;
    reasons?: string[];
    config?: Record<string, unknown>;
  }>;
};

export const api = {
  labStatus: (signal?: AbortSignal) => req<LabStatus>("/api/lab-status", { signal }),
  configure: {
    get: () => req<Settings>("/api/configure"),
    put: (body: Partial<Settings>) =>
      req<Settings>("/api/configure", { method: "PUT", body: JSON.stringify(body) }),
  },
  usage: () => req<UsageSummary>("/api/usage"),
  startServe: (body: Record<string, unknown>) =>
    req<{ job_id: string }>("/api/serve/start", { method: "POST", body: JSON.stringify(body) }),
  serveExamples: () =>
    req<{ examples: Record<string, ServeExample>; presets: string[] }>("/api/serve/examples"),
  stopServe: () => req<{ job_id: string }>("/api/serve/stop", { method: "POST" }),
  serveEngines: () => req<{ engines: ServeEngine[] }>("/api/serve/engines"),
  recommendServe: (model: string, fetchRemote = true, engine = "vllm") =>
    req<ServeRecommend>(
      `/api/serve/recommend?model=${encodeURIComponent(model)}&fetch_remote=${fetchRemote}&backend=${encodeURIComponent(engine)}`,
    ),
  job: (id: string) => req<Job>(`/api/jobs/${id}`),
  cancelJob: (id: string) =>
    req<{ job_id: string; status?: string }>(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }),
  jobs: () =>
    req<
      Array<{
        job_id: string;
        kind: string;
        status: string;
        progress: number;
        message: string;
        created_at?: string;
        updated_at?: string;
      }>
    >("/api/jobs"),
  smoke: () => req<{ ok: boolean; content: string }>("/api/smoke", { method: "POST" }),
  benchAgentic: (body: {
    suite?: "golden" | "tool_eval";
    preset?: "short" | "full" | "hardmode" | "coding";
    seed?: number;
    model?: string;
    base_url?: string;
    intent?: string;
    context_pressure?: number | null;
  }) =>
    req<{ job_id: string }>("/api/bench/agentic", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  toolEvalStatus: () =>
    req<{
      available: boolean;
      path?: string | null;
      via?: string;
      version?: string | null;
      install?: string;
      repo?: string;
    }>("/api/bench/tool-eval-status"),
  runs: (opts?: { kind?: string; limit?: number }) => {
    const q = new URLSearchParams();
    if (opts?.kind) q.set("kind", opts.kind);
    if (opts?.limit) q.set("limit", String(opts.limit));
    const s = q.toString();
    return req<RunRow[]>(`/api/runs${s ? `?${s}` : ""}`);
  },
  runsCount: (kind?: string) =>
    req<{ count: number }>(`/api/runs/count${kind ? `?kind=${encodeURIComponent(kind)}` : ""}`),
  run: (runId: string) =>
    req<{
      index: RunRow;
      envelope: Record<string, unknown> | null;
      tool_eval_raw?: Record<string, unknown> | null;
    }>(`/api/runs/${encodeURIComponent(runId)}`),
  toolEvalBoard: (limit = 40) =>
    req<{ runs: ToolEvalBoardRow[]; count: number }>(
      `/api/runs/tool-eval/board?limit=${limit}`,
    ),
  toolEvalCompare: (ids: string[]) =>
    req<{
      runs: ToolEvalBoardRow[];
      winner_run_id: string;
      winner_model?: string;
      metrics: Array<{ metric: string; values: Record<string, unknown>; delta_best_vs_rest?: number }>;
      categories: Array<{ id: string; label: string; values: Record<string, number | null | undefined> }>;
    }>(`/api/runs/tool-eval/compare?ids=${encodeURIComponent(ids.join(","))}`),
  lab: {
    list: (opts?: { limit?: number; task_type?: string; model?: string; fingerprint?: string }) => {
      const q = new URLSearchParams();
      if (opts?.limit) q.set("limit", String(opts.limit));
      if (opts?.task_type) q.set("task_type", opts.task_type);
      if (opts?.model) q.set("model", opts.model);
      if (opts?.fingerprint) q.set("fingerprint", opts.fingerprint);
      const s = q.toString();
      return req<{ runs: LabArtifactRun[]; count: number }>(`/api/lab/runs${s ? `?${s}` : ""}`);
    },
    get: (id: string) =>
      req<LabArtifactRun & { files: string[]; siblings?: LabArtifactRun[] }>(
        `/api/lab/runs/${encodeURIComponent(id)}`,
      ),
    compare: (ids: string[]) =>
      req<{
        runs: LabArtifactRun[];
        task_fingerprint: string | null;
        same_brief: boolean;
        brief: string | null;
      }>(`/api/lab/compare?ids=${encodeURIComponent(ids.join(","))}`),
    share: (id: string, makePublic = true) =>
      req<LabArtifactRun>(`/api/lab/runs/${encodeURIComponent(id)}/share`, {
        method: "POST",
        body: JSON.stringify({ public: makePublic }),
      }),
  },
  /* Streams engine (controller) — PLAN.md A2. Events stream from
     GET /api/streams/runs/:id/events (see lib/use-stream-run.ts). */
  streamPacks: () => req<StreamPack[]>("/api/streams/packs"),
  startStreamRun: (body: StreamRunRequest) =>
    req<{ run_id: string }>("/api/streams/runs", { method: "POST", body: JSON.stringify(body) }),
  stopStreamRun: (runId: string) =>
    req<{ ok: boolean }>(`/api/streams/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" }),
  streamRunSnapshot: (runId: string) =>
    req<StreamRunSnapshot>(`/api/streams/runs/${encodeURIComponent(runId)}`),
  listStreamRuns: () => req<StreamRunRow[]>("/api/streams/runs"),
};

export function streamRunEventsUrl(runId: string): string {
  return tokenQuery(`/api/streams/runs/${encodeURIComponent(runId)}/events`);
}

const TERMINAL_JOB = new Set(["completed", "failed", "cancelled", "done", "error"]);

/**
 * Follow a serve-engine job: logs + status over SSE until a terminal state.
 *
 * The SSE replays the log from byte 0 on every (re)connection; only bytes not yet
 * delivered reach `onLog`. After 3 consecutive stream errors the EventSource is
 * closed and the job row is polled with backoff (2 s → 15 s): a terminal row
 * finishes the watch, a live row re-opens the stream, and only a 404 — never a
 * transient failure (controller restart, 401 while a token is re-pasted) — is
 * reported as failed.
 */
export function watchJob(
  jobId: string,
  onLog: (chunk: string) => void,
  onStatus: (s: { status: string; progress: number; message: string }) => void,
  onResult?: (r: unknown) => void,
): () => void {
  let es: EventSource | null = null;
  let closed = false;
  let errorTicks = 0;
  let delivered = 0;
  let seen = 0;
  let poll: ReturnType<typeof setTimeout> | null = null;
  let backoff = 2000;
  let last = { status: "running", progress: 0, message: "" };

  const finish = (payload?: unknown) => {
    if (closed) return;
    closed = true;
    es?.close();
    es = null;
    if (poll) clearTimeout(poll);
    poll = null;
    if (payload !== undefined) onResult?.(payload);
  };

  const status = (s: { status: string; progress: number; message: string }) => {
    last = s;
    onStatus(s);
  };

  const checkJob = () => {
    poll = null;
    void api
      .job(jobId)
      .then((j) => {
        if (closed) return;
        status({ status: j.status, progress: j.progress ?? 0, message: j.message || j.status });
        if (TERMINAL_JOB.has(j.status)) finish(j);
        else {
          backoff = 2000;
          open(); // the job API answers again: resume the stream (already-seen log bytes are skipped)
        }
      })
      .catch((e: unknown) => {
        if (closed) return;
        if (e instanceof ApiError && e.status === 404) {
          status({ status: "failed", progress: last.progress, message: "job not found — the serve-engine no longer knows it" });
          finish(null);
          return;
        }
        onStatus({ ...last, message: "reconnecting…" });
        poll = setTimeout(checkJob, backoff);
        backoff = Math.min(backoff * 2, 15_000);
      });
  };

  const open = () => {
    const src = new EventSource(tokenQuery(`/api/jobs/${jobId}/logs`));
    es = src;
    seen = 0;
    src.onopen = () => {
      seen = 0;
    };
    src.addEventListener("log", (e) => {
      const chunk = (e as MessageEvent).data as string;
      const skip = Math.max(0, Math.min(chunk.length, delivered - seen));
      seen += chunk.length;
      if (chunk.length > skip) {
        delivered += chunk.length - skip;
        onLog(chunk.slice(skip));
      }
    });
    src.addEventListener("status", (e) => {
      errorTicks = 0;
      try {
        status(JSON.parse((e as MessageEvent).data));
      } catch {
        /* */
      }
    });
    src.addEventListener("result", (e) => {
      try {
        finish(JSON.parse((e as MessageEvent).data));
      } catch {
        finish(null);
      }
    });
    src.onerror = () => {
      if (closed || es !== src) return;
      errorTicks += 1;
      // EventSource retries on its own; after a few consecutive failures stop it and ask the job API.
      if (errorTicks < 3) return;
      errorTicks = 0;
      src.close();
      es = null;
      checkJob();
    };
  };

  open();
  return () => finish();
}

export type Settings = {
  defaultBackend: string;
  defaultModel: string;
  backends: Record<string, { url: string; enabled: boolean; label: string }>;
};

export type UsageSummary = {
  lifetimeTokens: number;
  lifetimePrompt: number;
  lifetimeCompletion: number;
  heatmap: Array<{ date: string; tokens: number }>;
  daily: Array<{ date: string; prompt: number; completion: number }>;
  mix: { prompt: number; completion: number };
  topModels: Array<{ model: string; tokens: number; calls: number }>;
};
