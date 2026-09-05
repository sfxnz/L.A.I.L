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
  started_at: string;
  prompts: StreamPromptRef[];
};

/** Coalesced per strand every ≤80 ms. `chunks` = upstream chunks folded into `text`. */
export type StreamDeltaEvent = {
  type: "delta";
  i: number;
  text: string;
  reasoning: boolean;
  chunks: number;
};

export type StreamStrandEvent = {
  type: "strand";
  i: number;
  state: StrandState;
  ttft_ms?: number;
  /** Live: estimated tokens (chunks × `tokens_per_chunk`). Terminal: `usage.completion_tokens` when present. */
  tokens?: number;
  /** Live: estimated tokens in the strand's last 1 s. Terminal: completion_tokens / (t_last − t_first). */
  tok_s?: number;
  peak_tok_s?: number;
  finish_reason?: string;
  error?: string;
  /** Last ≤100 inter-token gaps (ms). */
  itl_ms?: number[];
};

/**
 * 4 Hz. `tok_s` = estimated tokens in the sliding 1 s window (live estimate; final numbers
 * come from `usage`). vLLM emits one chunk per scheduler step, and with speculative/MTP
 * decoding a chunk carries several tokens, so chunks are scaled by `tokens_per_chunk`:
 * 1 until a strand of this base_url+model has finished with a `usage` frame, then
 * Σ completion_tokens / Σ chunks over finished strands (remembered across runs).
 */
export type StreamAggEvent = {
  type: "agg";
  t_ms: number;
  tok_s: number;
  peak_tok_s: number;
  tokens: number;
  running: number;
  waiting: number;
  done: number;
  ttft_p50_ms?: number;
  ttft_p95_ms?: number;
  tokens_per_chunk: number;
};

/** Bench modes only, after each wave (or immediately for a skipped size). */
export type StreamLevelEvent = {
  type: "level";
  index: number;
  concurrency?: number;
  size?: number;
  aggregate_tok_s: number | null;
  per_stream_median_tok_s: number | null;
  ttft_p50_ms: number | null;
  ttft_p95_ms: number | null;
  ttft_p99_ms: number | null;
  tpot_ms: number | null;
  ok: number;
  requests: number;
  errors: string[];
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
  /** Peak of the live 1 s-window estimate (see `StreamAggEvent`). */
  peak_tok_s: number;
  aggregate_tok_s: number | null;
  per_stream_median_tok_s: number | null;
  ttft_p50_ms: number | null;
  ttft_p95_ms: number | null;
  ok: number;
  requests: number;
  errors: string[];
  /** Bench modes. */
  headline?: BenchHeadline;
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

export type BenchArm = {
  concurrency?: number;
  size?: number;
  ok: number;
  requests: number;
  ttft_s: { p50: number | null; p95: number | null; p99: number | null };
  aggregate_tok_per_s: number | null;
  decode_tok_per_s_median: number | null;
  tpot_s: number | null;
  errors: string[];
  prompt_tokens?: number | null;
  prefill_tok_per_s?: number | null;
  skipped?: string;
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
  /** ms since the request was sent, one entry per output chunk. */
  token_times_ms: number[];
  /** Counts fell back to chunk count (no `usage` frame). */
  estimated?: boolean;
};

export type BenchHeadline = {
  decode_tok_per_s_median_c1: number | null;
  aggregate_peak_tok_per_s: number | null;
  aggregate_peak_concurrency: number | null;
  /** Prefill tok/s at the largest completed size. */
  prefill_tok_per_s_sustained?: number | null;
  ttft_p50_s_c1: number | null;
};

export type BenchEnvelope = {
  kind: "decode" | "prefill";
  model: string;
  workload: {
    pack: string;
    levels?: number[];
    sizes?: number[];
    max_tokens: number;
    thinking: StreamThinking;
    temperature: number;
    fill_to_max: boolean;
    base_url: string;
  };
  metrics: {
    arms: BenchArm[];
    full_arms: Array<BenchArm & { per_request: BenchPerRequest[] }>;
    headline: BenchHeadline;
  };
  summary: BenchHeadline;
  source: "controller-streams";
};
