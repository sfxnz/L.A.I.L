const BASE = "";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  return r.json();
}

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
    models?: Array<{ id: string }>;
    hardware?: {
      gpu_sku?: string;
      ram_gib?: number;
      available_gib?: number | null;
      cpu?: string;
    };
    containers?: Array<{ name: string; status: string; image: string }>;
    headroom?: string;
    error?: string;
    unreachable?: boolean;
    presets?: string[];
    serve_examples?: Record<string, ServeExample>;
    tool_eval?: {
      available: boolean;
      path?: string | null;
      via?: string;
      version?: string | null;
      install?: string;
      repo?: string;
    };
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
  play_url: string;
  preview_url?: string | null;
  gallery_url?: string;
  public_url?: string | null;
  task_fingerprint?: string;
  hermes?: { session?: string; source?: string } | null;
  share?: { public: boolean; slug: string | null };
  files?: string[];
  siblings?: LabArtifactRun[];
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
  config: Record<string, unknown>;
  rationale: string[];
  warnings: string[];
  detected: Record<string, unknown>;
  sources?: Array<{ kind: string; ref: string; notes?: string }>;
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
  health: () => req<{ status: string }>("/api/health"),
  labStatus: () => req<LabStatus>("/api/lab-status"),
  status: () => req<Record<string, unknown>>("/api/status"),
  bootstrap: () => req<{ workspace: Workspace; settings: Settings }>("/api/bootstrap"),
  configure: {
    get: () => req<Settings>("/api/configure"),
    put: (body: Partial<Settings>) =>
      req<Settings>("/api/configure", { method: "PUT", body: JSON.stringify(body) }),
  },
  workspaces: {
    list: () => req<Workspace[]>("/api/workspaces"),
    create: (name: string, rootPath?: string) =>
      req<Workspace>("/api/workspaces", {
        method: "POST",
        body: JSON.stringify({ name, rootPath }),
      }),
    tree: (id: string) => req<TreeNode[]>(`/api/workspaces/${id}/tree`),
    patch: (id: string, body: Partial<Workspace>) =>
      req<Workspace>(`/api/workspaces/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
    readFile: (id: string, path: string) =>
      req<{ path: string; content: string; size: number }>(
        `/api/workspaces/${id}/file?path=${encodeURIComponent(path)}`,
      ),
    writeFile: (id: string, path: string, content: string) =>
      req<{ ok: boolean }>(`/api/workspaces/${id}/file`, {
        method: "PUT",
        body: JSON.stringify({ path, content }),
      }),
  },
  sessions: {
    list: () => req<Session[]>("/api/sessions"),
    create: (title?: string, workspaceId?: string) =>
      req<Session>("/api/sessions", {
        method: "POST",
        body: JSON.stringify({ title, workspaceId }),
      }),
    get: (id: string) =>
      req<{ session: Session; messages: Message[] }>(`/api/sessions/${id}`),
    patch: (id: string, body: Partial<Session>) =>
      req<Session>(`/api/sessions/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  },
  agentRun: (
    sessionId: string,
    message: string,
    workspaceId?: string,
    mode: AgentMode = "agent",
    editorSnapshot?: EditorSnapshot,
  ) =>
    req<{ runId: string }>("/api/agent/run", {
      method: "POST",
      body: JSON.stringify({ sessionId, message, workspaceId, mode, editorSnapshot }),
    }),
  cancelAgentRun: (runId: string) =>
    req<{ ok: boolean }>(`/api/agent/runs/${runId}/cancel`, { method: "POST" }),
  shellApproval: (runId: string, approvalId: string, decision: "allow" | "deny") =>
    req<{ ok: boolean }>(`/api/agent/runs/${runId}/shell-approvals/${approvalId}`, {
      method: "POST",
      body: JSON.stringify({ decision }),
    }),
  patches: {
    list: (q: { sessionId?: string; runId?: string; status?: string } = {}) => {
      const sp = new URLSearchParams();
      if (q.sessionId) sp.set("sessionId", q.sessionId);
      if (q.runId) sp.set("runId", q.runId);
      if (q.status) sp.set("status", q.status);
      return req<Patch[]>(`/api/patches?${sp}`);
    },
    accept: (id: string) =>
      req<Patch>(`/api/patches/${id}/accept`, { method: "POST" }),
    reject: (id: string) =>
      req<Patch>(`/api/patches/${id}/reject`, { method: "POST" }),
    acceptAll: (body: { sessionId?: string; runId?: string }) =>
      req<Patch[]>("/api/patches/accept-all", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  },
  usage: () => req<UsageSummary>("/api/usage"),
  models: {
    local: () => req<{ local: ModelCard[] }>("/api/models"),
    search: (q: string) => req<{ results: ModelCard[]; error?: string }>(`/api/models/search?q=${encodeURIComponent(q)}`),
    pull: (model: string, backend: "hf" | "vllm" | "llamacpp" = "hf") =>
      req<{ jobId: string }>("/api/models/pull", {
        method: "POST",
        body: JSON.stringify({ model, backend }),
      }),
  },
  startServe: (body: Record<string, unknown>) =>
    req<{ job_id: string }>("/api/serve/start", { method: "POST", body: JSON.stringify(body) }),
  stopServe: () => req<{ job_id: string }>("/api/serve/stop", { method: "POST" }),
  agentRestore: () => req<{ job_id: string }>("/api/serve/agent-restore", { method: "POST" }),
  recommendServe: (model: string, mode: string, fetchRemote = true) =>
    req<ServeRecommend>(
      `/api/serve/recommend?model=${encodeURIComponent(model)}&mode=${encodeURIComponent(mode)}&fetch_remote=${fetchRemote}`,
    ),
  job: (id: string) => req<Job>(`/api/jobs/${id}`),
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
  benchPerf: (body: Record<string, unknown>) =>
    req<{ job_id: string }>("/api/bench/perf", { method: "POST", body: JSON.stringify(body) }),
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
    import: (body: {
      title: string;
      from: string;
      task_type?: string;
      model_id?: string;
      entry?: string;
      tags?: string[];
      brief?: string;
      eval_run_id?: string;
      share_public?: boolean;
    }) =>
      req<LabArtifactRun>("/api/lab/runs/import", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    share: (id: string, makePublic = true) =>
      req<LabArtifactRun>(`/api/lab/runs/${encodeURIComponent(id)}/share`, {
        method: "POST",
        body: JSON.stringify({ public: makePublic }),
      }),
  },
};

export function watchJob(
  jobId: string,
  onLog: (chunk: string) => void,
  onStatus: (s: { status: string; progress: number; message: string }) => void,
  onResult?: (r: unknown) => void,
): () => void {
  const es = new EventSource(`/api/jobs/${jobId}/logs`);
  es.addEventListener("log", (e) => onLog((e as MessageEvent).data));
  es.addEventListener("status", (e) => {
    try {
      onStatus(JSON.parse((e as MessageEvent).data));
    } catch {
      /* */
    }
  });
  es.addEventListener("result", (e) => {
    try {
      onResult?.(JSON.parse((e as MessageEvent).data));
    } catch {
      /* */
    }
    es.close();
  });
  return () => es.close();
}

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
  pinned: boolean;
  createdAt: string;
  updatedAt: string;
};

export type Message = {
  id: string;
  sessionId: string;
  role: string;
  content: string;
  createdAt: string;
  meta?: Record<string, unknown>;
};

export type TreeNode = {
  name: string;
  path: string;
  type: "file" | "dir";
  children?: TreeNode[];
};

export type Settings = {
  defaultBackend: string;
  defaultModel: string;
  backends: Record<string, { url: string; enabled: boolean; label: string }>;
  hfToken?: string;
  contextBudgetChars?: number;
  contextMaxFileChars?: number;
  contextMaxSearchHits?: number;
};

/** Mirrors @lail/shared ContextMention */
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

/** Mirrors @lail/shared EditorSnapshot — client → server context pack */
export type EditorSnapshot = {
  openFiles: Array<{ path: string; content?: string }>;
  activePath?: string | null;
  selection?: EditorSelection | null;
  mentions: ContextMention[];
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

export type ModelCard = {
  id: string;
  name: string;
  author?: string;
  downloads?: number;
  likes?: number;
  tags?: string[];
  license?: string;
  sizeHint?: string;
  quantizations?: string[];
  hardwareFit?: string;
  local?: boolean;
  backends?: string[];
  pipeline_tag?: string;
};

export type AgentMode = "plan" | "ask" | "agent";

export type Patch = {
  id: string;
  runId: string;
  sessionId: string;
  path: string;
  oldString: string;
  newString: string;
  op: "replace" | "create" | "delete";
  status: "pending" | "accepted" | "rejected" | "failed";
  reason?: string;
  createdAt: string;
  resolvedAt?: string;
};
