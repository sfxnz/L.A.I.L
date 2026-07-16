export type BackendKind = "vllm" | "llamacpp";

export type LabSettings = {
  defaultBackend: BackendKind;
  defaultModel: string;
  backends: Record<BackendKind, { url: string; enabled: boolean; label: string }>;
  hfToken?: string;
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

export type AgentEvent =
  | { type: "thought"; runId: string; text: string }
  | { type: "status"; runId: string; text: string }
  | { type: "tool_start"; runId: string; tool: string; args: Record<string, unknown> }
  | { type: "tool_end"; runId: string; tool: string; summary: string; output?: string }
  | { type: "file_write"; runId: string; path: string; bytes: number }
  | { type: "assistant"; runId: string; text: string; delta?: boolean }
  | { type: "done"; runId: string; usage?: { prompt: number; completion: number } }
  | { type: "error"; runId: string; message: string };

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
