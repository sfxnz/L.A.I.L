export * from "./stats";

export type BackendKind = "vllm" | "llamacpp";

export type LabSettings = {
  defaultBackend: BackendKind;
  defaultModel: string;
  backends: Record<BackendKind, { url: string; enabled: boolean; label: string }>;
  hfToken?: string;
  contextBudgetChars?: number;
  contextMaxFileChars?: number;
  contextMaxSearchHits?: number;
};

export type ContextMention =
  | { type: "file"; path: string }
  | { type: "folder"; path: string }
  | { type: "search"; query: string };

export type EditorSelection = {
  path: string;
  startLine: number;
  endLine: number;
  text: string;
};

export type EditorSnapshot = {
  openFiles: Array<{ path: string; content?: string }>;
  activePath?: string | null;
  selection?: EditorSelection | null;
  mentions: ContextMention[];
};

export type Workspace = {
  id: string;
  name: string;
  rootPath: string;
  pinned: boolean;
  createdAt: string;
  updatedAt: string;
};

export type Session = {
  id: string;
  title: string;
  workspaceId: string | null;
  createdAt: string;
  updatedAt: string;
  pinned: boolean;
};

export type ChatMessage = {
  id: string;
  sessionId: string;
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  createdAt: string;
  meta?: Record<string, unknown>;
};

export type AgentMode = "plan" | "ask" | "agent";

export type PatchOp = "replace" | "create" | "delete";

export type PatchStatus = "pending" | "accepted" | "rejected" | "failed";

export type Patch = {
  id: string;
  runId: string;
  sessionId: string;
  path: string;
  oldString: string;
  newString: string;
  op: PatchOp;
  status: PatchStatus;
  reason?: string;
  createdAt: string;
  resolvedAt?: string;
};

export type AgentRunStatus = "running" | "done" | "error" | "cancelled";

export type AgentEvent =
  | { type: "thought"; runId: string; text: string }
  | { type: "token"; runId: string; text: string; channel?: "assistant" | "thought" }
  | { type: "status"; runId: string; text: string }
  | { type: "tool_start"; runId: string; tool: string; args: Record<string, unknown> }
  | { type: "tool_end"; runId: string; tool: string; summary: string; output?: string }
  | { type: "patch_proposed"; runId: string; patch: Patch }
  | { type: "patch_updated"; runId: string; patch: Patch }
  | { type: "shell_approval_required"; runId: string; approvalId: string; command: string }
  | { type: "file_write"; runId: string; path: string; bytes: number } // after accept only
  | { type: "assistant"; runId: string; text: string; delta?: boolean }
  | { type: "done"; runId: string; usage?: { prompt: number; completion: number } }
  | { type: "cancelled"; runId: string }
  | { type: "error"; runId: string; message: string }
  | { type: "context_truncated"; runId: string; dropped: string[] };

export type DownloadEvent =
  | { type: "download_progress"; jobId: string; progress: number; message: string }
  | { type: "download_done"; jobId: string; model: string }
  | { type: "download_error"; jobId: string; message: string };

export type WsEnvelope =
  | { channel: string; event: AgentEvent | DownloadEvent | { type: string; [k: string]: unknown } };

export type UsageSummary = {
  lifetimeTokens: number;
  lifetimePrompt: number;
  lifetimeCompletion: number;
  heatmap: Array<{ date: string; tokens: number }>;
  daily: Array<{ date: string; prompt: number; completion: number }>;
  mix: { prompt: number; completion: number };
  topModels: Array<{ model: string; tokens: number; calls: number }>;
};

export type ModelCard = {
  id: string;
  name: string;
  author?: string;
  downloads?: number;
  likes?: number;
  tags?: string[];
  pipeline_tag?: string;
  library_name?: string;
  license?: string;
  description?: string;
  sizeHint?: string;
  quantizations?: string[];
  hardwareFit?: "excellent" | "good" | "tight" | "unknown";
  local?: boolean;
  backends?: BackendKind[];
};

export type TreeNode = {
  name: string;
  path: string;
  type: "file" | "dir";
  children?: TreeNode[];
};

// ── Streams engine (controller fan-out, SSE multiplex) ──────────────

export type StreamRunMode = "load" | "bench-decode" | "bench-prefill";
export type StreamThinking = "auto" | "on" | "off";
export type StreamArrival = "burst" | "staggered" | "poisson";
export type StreamRunStatus = "running" | "done" | "cancelled" | "error";
export type StrandState = "waiting" | "prefill" | "decode" | "done" | "error" | "cancelled";

export type StreamRunRequest = {
  mode: StreamRunMode;
  /** Configured backend URL or `http://127.0.0.1:<port>`; default = default backend. */
  base_url?: string;
  /** Served model id; default = first id from `GET {base_url}/v1/models`. */
  model?: string;
  pack: string;
  /** load: strands (1–32). */
  n?: number;
  /** bench-decode: concurrency levels, run ascending as sequential waves. */
  levels?: number[];
  /** bench-prefill: prompt sizes in tokens, one stream each with max_tokens 1. */
  sizes?: number[];
  /**
   * Bench modes: repeated samples. bench-decode: minimum strands per level — a level
   * runs ⌈samples ÷ c⌉ waves (default 3: ×1 three times, ×2 twice, ×4+ once).
   * bench-prefill: requests per size (default 2). Headline numbers are medians with min–max.
   */
  samples?: number;
  max_tokens?: number;
  /** min_tokens = max_tokens + ignore_eos (forced on for bench-decode). */
  fill_to_max?: boolean;
  thinking?: StreamThinking;
  temperature?: number;
  arrival?: StreamArrival;
  arrival_ms?: number;
};

export type StreamPackPrompt = { title: string; text: string };
export type StreamPack = {
  id: string;
  label: string;
  prompts: StreamPackPrompt[];
  default_max_tokens: number;
};

export type StreamPromptRef = {
  i: number;
  title: string;
  text: string;
  /** Source family (for `mixed`, the underlying pack). */
  pack: string;
  /** Bench modes: index into `levels` / `sizes` this strand belongs to. */
  level?: number;
  /** Bench modes: repeat (wave) of its level, 0-based. */
  wave?: number;
};

export type StreamHelloEvent = {
  type: "hello";
  run_id: string;
  mode: StreamRunMode;
  model: string;
  base_url: string;
  /** Total strands in the run (Σ levels for bench-decode, sizes.length for bench-prefill). */
  n: number;
  levels?: number[];
  sizes?: number[];
  max_tokens: number;
  /** `max_model_len` read from `GET {base_url}/v1/models`; null when the server does not report it. */
  max_model_len: number | null;
  /** Bench modes: see `StreamRunRequest.samples`. */
  samples?: number;
  /** Serve-engine `engine.flags_fingerprint` at run start (the serve config measured); null when unknown. */
  serve_fingerprint?: string | null;
  started_at: string;
  prompts: StreamPromptRef[];
};

/** Coalesced per strand every ≤80 ms. `chunks` = upstream chunks folded into `text`, `tokens` = tokens they carried. */
export type StreamDeltaEvent = {
  type: "delta";
  i: number;
  text: string;
  reasoning: boolean;
  chunks: number;
  /** Exact from per-chunk `usage.completion_tokens` deltas (1 per chunk when the server sends none). */
  tokens: number;
};

export type StreamStrandEvent = {
  type: "strand";
  i: number;
  state: StrandState;
  ttft_ms?: number;
  /** Tokens so far: the latest `usage.completion_tokens` (chunk count only when the server sends no usage). */
  tokens?: number;
  /**
   * Live (every 250 ms tick while decoding, so a stalled strand decays to 0): tokens that
   * arrived in the last `RATE_WINDOW_MS` ÷ that window. Terminal: decode rate,
   * (completion_tokens − first-chunk tokens) / (t_last − t_first) = 1 / TPOT.
   */
  tok_s?: number;
  /** Highest live window rate once a full window has been decoding. */
  peak_tok_s?: number;
  finish_reason?: string;
  error?: string;
  /**
   * Decode steps: the gap (ms) before each output chunk and the tokens it carried. One chunk
   * is one scheduler step; with speculative/MTP decoding it carries several tokens, so
   * per-token ITL is gap ÷ tokens (`perTokenLatencies`); a long gap is a stall regardless.
   * Live events carry only the steps since the previous event (`steps_append`); terminal
   * events and snapshots carry the last ≤100.
   */
  step_ms?: number[];
  step_tokens?: number[];
  steps_append?: boolean;
};

/**
 * 4 Hz. `tok_s` = tokens that arrived in the sliding `RATE_WINDOW_MS` window ÷ the window
 * (the first second of a run divides by 1 s, not less, so it ramps instead of spiking).
 * Requests ask for `continuous_usage_stats`, so every vLLM chunk carries the exact
 * cumulative `usage.completion_tokens` and its token delta is counted — no tokens-per-chunk
 * estimate (MTP chunks carry 1–4 tokens).
 */
export type StreamAggEvent = {
  type: "agg";
  t_ms: number;
  tok_s: number;
  /** Highest `tok_s` once a full window has been decoding (never a part-window spike). */
  peak_tok_s: number;
  tokens: number;
  running: number;
  waiting: number;
  done: number;
  ttft_p50_ms?: number;
  ttft_p95_ms?: number;
  /** False when some chunk arrived without `usage` (counted as 1 token): live numbers are then chunk counts. */
  tokens_exact: boolean;
};

/** vLLM `/metrics` deltas over one bench level — the server's view, a cross-check without client overhead. */
export type ServerLevelMetrics = {
  /** Δspec_decode_num_accepted_tokens ÷ Δspec_decode_num_draft_tokens; null without speculative decoding. */
  spec_acceptance: number | null;
  /** Tokens per decode step: 1 + Δaccepted ÷ Δdrafts. */
  spec_tokens_per_step: number | null;
  /** Δtime_to_first_token_seconds_sum ÷ Δcount, ms. */
  ttft_mean_ms: number | null;
  /** Δrequest_prefill_kv_computed_tokens_sum ÷ Δrequest_prefill_time_seconds_sum (prefix-cache hits excluded). */
  prefill_tok_s: number | null;
  /** Δnum_preemptions. */
  preemptions: number | null;
};

export type BenchRange = [number, number];

/** Bench modes only, after each wave (or immediately for a skipped size). */
export type StreamLevelEvent = {
  type: "level";
  index: number;
  concurrency?: number;
  size?: number;
  /**
   * Wall-clock per wave: Σ completion_tokens / (max t_last − min t_start) over ok strands
   * (includes TTFT and stragglers' tails). Median over the level's waves.
   */
  aggregate_tok_s: number | null;
  /**
   * Decode span per wave: Σ (completion_tokens − first-chunk tokens) / (max t_last − min t_first) —
   * the time-average of the live `agg.tok_s`, so the gauge's needle and settled mark agree.
   * Median over the level's waves.
   */
  aggregate_steady_tok_s: number | null;
  /** Median per-strand decode rate over every ok strand of the level (= 1 / TPOT). */
  per_stream_median_tok_s: number | null;
  /** Waves (decode) or requests (prefill) behind the medians. */
  samples?: number;
  /** min–max of the per-wave `aggregate_tok_s` (decode, ≥ 2 waves). */
  aggregate_range?: BenchRange | null;
  /** min–max of the per-strand decode rates (decode) or prefill rates (prefill), ≥ 2 samples. */
  per_stream_range?: BenchRange | null;
  /** Most requests running/waiting on the server that were not this run's, sampled 1 Hz; > 0 = contended level. */
  foreign_max?: number | null;
  server?: ServerLevelMetrics | null;
  ttft_p50_ms: number | null;
  /** p95/p99 only from `TAIL_MIN_SAMPLES` strands up; null below. */
  ttft_p95_ms: number | null;
  ttft_p99_ms: number | null;
  tpot_ms: number | null;
  ok: number;
  requests: number;
  errors: string[];
  /** Prefill sizes only: median prompt tokens and prompt_tokens / TTFT. */
  prompt_tokens?: number | null;
  prefill_tok_s?: number | null;
  skipped?: string;
};

export type StreamRunSummary = {
  status: StreamRunStatus;
  mode: StreamRunMode;
  model: string;
  duration_ms: number;
  /** Σ completion tokens over every strand, cancelled ones included (`usage` where present, chunk count otherwise). */
  tokens: number;
  /** Peak of the live window rate (see `StreamAggEvent`). */
  peak_tok_s: number;
  /**
   * load: over ok strands, wall-clock (`StreamLevelEvent.aggregate_tok_s` definition, "goodput").
   * Bench modes: the headline — peak level aggregate. For a `cancelled` run these are partial:
   * the window actually run, TTFT over strands that reached first token, `ok` = strands that produced output.
   */
  aggregate_tok_s: number | null;
  /** load: decode span over all strands (the time-average of the live `tok_s`). Bench: the peak level's. */
  aggregate_steady_tok_s: number | null;
  /** load: median per-strand decode rate. Bench: the ×1 level's (the headline). */
  per_stream_median_tok_s: number | null;
  ttft_p50_ms: number | null;
  ttft_p95_ms: number | null;
  ok: number;
  requests: number;
  errors: string[];
  /** Bench modes. */
  headline?: BenchHeadline;
  /** Bench modes: node temperature / power / energy over the run (no series). */
  hardware?: Omit<BenchHardware, "series"> | null;
  error?: string;
};

export type StreamDoneEvent = {
  type: "done";
  run_id: string;
  summary: StreamRunSummary;
  /** Serve-engine run id after `POST /api/runs/import` (bench modes); null when not saved. */
  saved_run_id?: string | null;
};

export type StreamErrorEvent = { type: "error"; message: string };

export type StreamRunEvent =
  | StreamHelloEvent
  | StreamDeltaEvent
  | StreamStrandEvent
  | StreamAggEvent
  | StreamLevelEvent
  | StreamDoneEvent
  | StreamErrorEvent;

export type StreamStrandSnapshot = Omit<StreamStrandEvent, "type"> & {
  /** Accumulated text, tail-capped at 8k chars each. */
  text: string;
  reasoning_text: string;
  chunks: number;
  /** Tokens that were reasoning (see `StreamDeltaEvent.tokens`). */
  reasoning_tokens: number;
};

/** `GET /api/streams/runs/:id` — everything a reconnecting client needs. */
export type StreamRunSnapshot = {
  hello: Omit<StreamHelloEvent, "type">;
  status: StreamRunStatus;
  strands: StreamStrandSnapshot[];
  /** Last ≤240 `agg` samples (60 s at 4 Hz). */
  agg: Array<Omit<StreamAggEvent, "type">>;
  /** `level` events emitted so far (bench modes). */
  levels: Array<Omit<StreamLevelEvent, "type">>;
  done: Omit<StreamDoneEvent, "type"> | null;
  error: string | null;
};

/** `GET /api/streams/runs` — recent runs, newest first. */
export type StreamRunRow = {
  run_id: string;
  mode: StreamRunMode;
  model: string;
  base_url: string;
  pack: string;
  n: number;
  status: StreamRunStatus;
  started_at: string;
  finished_at: string | null;
  summary: StreamRunSummary | null;
  saved_run_id: string | null;
};

// Bench envelope — compatible with the serve-engine run index (`POST /api/runs/import`).

/** Definitions: see the matching `StreamLevelEvent` fields. */
export type BenchArm = {
  concurrency?: number;
  size?: number;
  ok: number;
  requests: number;
  ttft_s: { p50: number | null; p95: number | null; p99: number | null };
  aggregate_tok_per_s: number | null;
  aggregate_steady_tok_per_s?: number | null;
  decode_tok_per_s_median: number | null;
  tpot_s: number | null;
  errors: string[];
  prompt_tokens?: number | null;
  prefill_tok_per_s?: number | null;
  skipped?: string;
  samples?: number;
  aggregate_range?: BenchRange | null;
  per_stream_range?: BenchRange | null;
  foreign_max?: number | null;
  server?: ServerLevelMetrics | null;
};

export type BenchPerRequest = {
  i: number;
  ttft_s: number | null;
  decode_s: number | null;
  completion_tokens: number | null;
  prompt_tokens: number | null;
  tok_per_s: number | null;
  finish_reason: string | null;
  error: string | null;
  /** ms since the request was sent, one entry per token-bearing output chunk (decode step). */
  token_times_ms: number[];
  /** Cumulative completion tokens at each `token_times_ms` entry. */
  token_counts: number[];
  /** Counts fell back to chunk count (no `usage` frame). */
  estimated?: boolean;
  /** Request belongs to this repeat (wave) of its level. */
  wave?: number;
};

export type BenchHeadline = {
  decode_tok_per_s_median_c1: number | null;
  aggregate_peak_tok_per_s: number | null;
  aggregate_peak_concurrency: number | null;
  /** Prefill tok/s at the largest completed size. */
  prefill_tok_per_s_sustained?: number | null;
  ttft_p50_s_c1: number | null;
};

/** Per node over the run window, from the serve-engine status snapshot sampled every 2 s. */
export type BenchHardwareNode = {
  id: string;
  samples: number;
  temp_max_c: number | null;
  temp_mean_c: number | null;
  power_mean_w: number | null;
  /** Lowest available memory seen (GiB). */
  available_min_gib: number | null;
};

export type BenchHardware = {
  /** [t_ms since run start, node id, temp °C, power W, available GiB] */
  series: Array<[number, string, number | null, number | null, number | null]>;
  nodes: BenchHardwareNode[];
  /** ∫ Σ node power dt over the sampled window (trapezoid); null without ≥ 2 power samples. */
  energy_j: number | null;
  /** energy_j ÷ Σ completion tokens of the run. */
  energy_j_per_token: number | null;
};

/** What makes two bench runs comparable ("vs previous", the ghost): same model, pack, max_tokens and serve fingerprint. */
export type BenchSummary = BenchHeadline & {
  pack: string;
  max_tokens: number;
  serve_fingerprint: string | null;
  energy_j_per_token?: number | null;
};

export type BenchEnvelope = {
  kind: "decode" | "prefill";
  model: string;
  workload: {
    pack: string;
    levels?: number[];
    sizes?: number[];
    samples: number;
    max_tokens: number;
    thinking: StreamThinking;
    temperature: number;
    fill_to_max: boolean;
    base_url: string;
    serve_fingerprint: string | null;
  };
  metrics: {
    arms: BenchArm[];
    full_arms: Array<BenchArm & { per_request: BenchPerRequest[] }>;
    headline: BenchHeadline;
    hardware?: BenchHardware | null;
  };
  summary: BenchSummary;
  source: "controller-streams";
};
