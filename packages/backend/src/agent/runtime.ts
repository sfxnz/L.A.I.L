import { randomUUID } from "crypto";
import type { AgentMode, EditorSnapshot } from "@lail/shared";
import { getDb } from "../db/schema";
import { wsHub } from "../ws/hub";
import { addMessage } from "../controller/sessions";
import { getWorkspace } from "../controller/workspaces";
import { getSettings, openAiBase, resolveModelId } from "../controller/settings";
import { recordUsage } from "../controller/usage";
import { runTool, toolDefinitions } from "../tools";
import { buildContextPack, loadHistory } from "./context";
import { systemPrompt } from "./prompts";
import { isToolAllowed, classifyShell } from "./tool-policy";
import { approvalHub } from "./approvals";
import { patchStore } from "./patch-store";

const DEFAULT_MAX_STEPS = 32;

export type StartRunOpts = {
  sessionId: string;
  message: string;
  workspaceId: string;
  mode: AgentMode;
  editorSnapshot?: EditorSnapshot;
  /** test seam */
  fetchImpl?: typeof fetch;
  maxSteps?: number;
};

type RunHandle = {
  abort: AbortController;
  cancelled: boolean;
};

type ChatMsg = {
  role: string;
  content?: string | null;
  tool_calls?: Array<{
    id: string;
    type: string;
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
  name?: string;
};

type LlmMessage = {
  content?: string | null;
  reasoning?: string | null;
  tool_calls?: Array<{
    id: string;
    type: string;
    function: { name: string; arguments: string };
  }>;
};

type LlmResult = {
  message: LlmMessage | null;
  usage: { prompt_tokens?: number; completion_tokens?: number };
};

const runs = new Map<string, RunHandle>();

function now() {
  return new Date().toISOString();
}

function publish(runId: string, event: Record<string, unknown>) {
  // Single channel only — clients default to "*" and dual-publish was doubling the UI stream
  wsHub.publish(`agent:${runId}`, { runId, ...event });
}

function extractText(msg: LlmMessage | null | undefined): string {
  if (!msg) return "";
  if (msg.content && String(msg.content).trim()) return String(msg.content);
  if (msg.reasoning && String(msg.reasoning).trim()) return String(msg.reasoning);
  return "";
}

function updateRun(
  runId: string,
  patch: {
    status?: string;
    error?: string | null;
    promptTokens?: number;
    completionTokens?: number;
  },
) {
  const ts = now();
  const sets: string[] = ["updated_at = ?"];
  const params: Array<string | number | null> = [ts];
  if (patch.status !== undefined) {
    sets.push("status = ?");
    params.push(patch.status);
  }
  if (patch.error !== undefined) {
    sets.push("error = ?");
    params.push(patch.error);
  }
  if (patch.promptTokens !== undefined) {
    sets.push("prompt_tokens = ?");
    params.push(patch.promptTokens);
  }
  if (patch.completionTokens !== undefined) {
    sets.push("completion_tokens = ?");
    params.push(patch.completionTokens);
  }
  params.push(runId);
  getDb()
    .query(`UPDATE agent_runs SET ${sets.join(", ")} WHERE id = ?`)
    .run(...params);
}

function insertRun(opts: {
  runId: string;
  sessionId: string;
  workspaceId: string;
  mode: AgentMode;
  message: string;
}) {
  const ts = now();
  getDb()
    .query(
      `INSERT INTO agent_runs
       (id, session_id, workspace_id, mode, status, message, error, prompt_tokens, completion_tokens, created_at, updated_at)
       VALUES (?, ?, ?, ?, 'running', ?, NULL, 0, 0, ?, ?)`,
    )
    .run(opts.runId, opts.sessionId, opts.workspaceId, opts.mode, opts.message, ts, ts);
}

export function startAgentRun(opts: StartRunOpts): { runId: string } {
  const runId = randomUUID();
  const abort = new AbortController();
  runs.set(runId, { abort, cancelled: false });

  insertRun({
    runId,
    sessionId: opts.sessionId,
    workspaceId: opts.workspaceId,
    mode: opts.mode,
    message: opts.message,
  });

  // Fire-and-forget
  void agentLoop(runId, opts).catch((e) => {
    const message = e instanceof Error ? e.message : String(e);
    publish(runId, { type: "error", message });
    updateRun(runId, { status: "error", error: message });
    runs.delete(runId);
  });

  return { runId };
}

export function cancelAgentRun(runId: string): boolean {
  const h = runs.get(runId);
  if (!h) {
    // Mark DB cancelled if still running
    const row = getDb().query("SELECT status FROM agent_runs WHERE id = ?").get(runId) as
      | { status: string }
      | null;
    if (row?.status === "running") {
      updateRun(runId, { status: "cancelled" });
      publish(runId, { type: "cancelled" });
      return true;
    }
    return false;
  }
  h.cancelled = true;
  try {
    h.abort.abort();
  } catch {
    /* ignore */
  }
  return true;
}

export function getAgentRun(
  runId: string,
): { id: string; status: string; mode: string } | null {
  const row = getDb()
    .query("SELECT id, status, mode FROM agent_runs WHERE id = ?")
    .get(runId) as { id: string; status: string; mode: string } | null;
  return row ? { id: row.id, status: row.status, mode: row.mode } : null;
}

function isCancelled(runId: string): boolean {
  return runs.get(runId)?.cancelled === true;
}

/** Exit the run as cancelled (caller should return immediately after). */
function finishCancelled(
  runId: string,
  promptTokens: number,
  completionTokens: number,
) {
  publish(runId, { type: "cancelled" });
  updateRun(runId, {
    status: "cancelled",
    promptTokens,
    completionTokens,
  });
  runs.delete(runId);
}

/**
 * Wait for shell approval, but exit early if the run is cancelled mid-wait.
 * Resolves "cancelled" when cancel wins; otherwise "allow" | "deny".
 */
function waitApprovalOrCancel(
  runId: string,
  approvalId: string,
): Promise<"allow" | "deny" | "cancelled"> {
  if (isCancelled(runId)) return Promise.resolve("cancelled");
  const handle = runs.get(runId);
  return new Promise((resolve) => {
    let settled = false;
    const finish = (v: "allow" | "deny" | "cancelled") => {
      if (settled) return;
      settled = true;
      handle?.abort.signal.removeEventListener("abort", onAbort);
      resolve(v);
    };
    const onAbort = () => {
      // Unblock approvalHub so the entry does not sit until timeout
      try {
        approvalHub.decide(approvalId, "deny");
      } catch {
        /* ignore */
      }
      finish("cancelled");
    };
    if (handle?.abort.signal.aborted) {
      onAbort();
      return;
    }
    handle?.abort.signal.addEventListener("abort", onAbort);
    approvalHub.wait(approvalId).then((d) => {
      if (isCancelled(runId)) finish("cancelled");
      else finish(d);
    });
  });
}

async function agentLoop(runId: string, opts: StartRunOpts) {
  const { sessionId, workspaceId, mode, message } = opts;
  const maxSteps = opts.maxSteps ?? DEFAULT_MAX_STEPS;
  const fetchFn = opts.fetchImpl ?? fetch;
  const handle = runs.get(runId);

  const ws = getWorkspace(workspaceId);
  if (!ws) {
    const err = "Workspace not found";
    publish(runId, { type: "error", message: err });
    updateRun(runId, { status: "error", error: err });
    runs.delete(runId);
    return;
  }

  let model: string;
  try {
    model = await resolveModelId();
  } catch (e) {
    const err = e instanceof Error ? e.message : String(e);
    publish(runId, { type: "error", message: err });
    updateRun(runId, { status: "error", error: err });
    runs.delete(runId);
    return;
  }

  const base = openAiBase();

  publish(runId, {
    type: "thought",
    text: `Using model \`${model}\` · workspace ${ws.rootPath} · mode=${mode}`,
  });
  publish(runId, { type: "status", text: "Working on your request…" });

  const settings = getSettings();
  const pack = await buildContextPack({
    rootPath: ws.rootPath,
    snapshot: opts.editorSnapshot ?? { openFiles: [], mentions: [] },
    budgetChars: settings.contextBudgetChars ?? 32_000,
    maxFileChars: settings.contextMaxFileChars,
    maxSearchHits: settings.contextMaxSearchHits,
    message: opts.message,
  });

  if (pack.truncated) {
    publish(runId, {
      type: "status",
      text: `Context truncated to budget (${pack.droppedLabels.length} lower-priority chunks dropped)`,
    });
    publish(runId, {
      type: "context_truncated",
      dropped: pack.droppedLabels,
    });
  }

  const history = await loadHistory(sessionId);
  const messages: ChatMsg[] = [
    {
      role: "system",
      content: systemPrompt(mode, ws.rootPath) + "\n\n" + pack.systemExtra,
    },
  ];
  if (pack.contextMessage) {
    messages.push(pack.contextMessage);
  }
  messages.push(...history);

  // Ensure latest user message is present (facade should have addMessage'd it)
  if (!messages.some((m) => m.role === "user" && m.content === message)) {
    messages.push({ role: "user", content: message });
  }

  const allowedTools = toolDefinitions.filter((t) => isToolAllowed(mode, t.function.name));

  let promptTokens = 0;
  let completionTokens = 0;
  let finalText = "";

  try {
    for (let step = 0; step < maxSteps; step++) {
      if (isCancelled(runId)) {
        finishCancelled(runId, promptTokens, completionTokens);
        return;
      }

      publish(runId, { type: "status", text: `Thinking (step ${step + 1})…` });

      let result: LlmResult;
      try {
        result = await callLlm({
          fetchFn,
          base,
          model,
          messages,
          tools: allowedTools,
          signal: handle?.abort.signal,
        });
      } catch (e) {
        if (isCancelled(runId) || (e instanceof Error && e.name === "AbortError")) {
          finishCancelled(runId, promptTokens, completionTokens);
          return;
        }
        throw e;
      }

      promptTokens += result.usage.prompt_tokens || 0;
      completionTokens += result.usage.completion_tokens || 0;

      // Honor cancel that arrived during LLM call (e.g. fetch mock ignores AbortSignal)
      // before treating response as final done or continuing with tools.
      if (isCancelled(runId)) {
        finishCancelled(runId, promptTokens, completionTokens);
        return;
      }

      const msg = result.message;
      if (!msg) break;

      if (msg.tool_calls?.length) {
        const thought = extractText(msg);
        messages.push({
          role: "assistant",
          content: thought || "",
          tool_calls: msg.tool_calls,
        });
        if (thought) {
          publish(runId, { type: "thought", text: thought });
        }

        for (const tc of msg.tool_calls) {
          if (isCancelled(runId)) {
            finishCancelled(runId, promptTokens, completionTokens);
            return;
          }

          let args: Record<string, unknown> = {};
          try {
            args = JSON.parse(tc.function.arguments || "{}");
          } catch {
            args = {};
          }
          const toolName = tc.function.name;

          publish(runId, { type: "tool_start", tool: toolName, args });

          let output = "";
          let summary = "";

          if (!isToolAllowed(mode, toolName)) {
            output = `Tool \`${toolName}\` is not available in ${mode} mode`;
            summary = "not available in this mode";
            publish(runId, {
              type: "tool_end",
              tool: toolName,
              summary,
              output: output.slice(0, 4000),
            });
            messages.push({
              role: "tool",
              tool_call_id: tc.id,
              name: toolName,
              content: output,
            });
            continue;
          }

          if (toolName === "run_shell") {
            const command = String(args.command || "");
            const classification = classifyShell(command);

            if (classification === "deny") {
              output = "Command blocked by safety policy";
              summary = "Blocked";
              publish(runId, {
                type: "tool_end",
                tool: toolName,
                summary,
                output,
              });
              messages.push({
                role: "tool",
                tool_call_id: tc.id,
                name: toolName,
                content: output,
              });
              continue;
            }

            if (classification === "approve") {
              const approvalId = randomUUID();
              publish(runId, {
                type: "shell_approval_required",
                approvalId,
                command,
              });
              const decision = await waitApprovalOrCancel(runId, approvalId);
              if (decision === "cancelled" || isCancelled(runId)) {
                finishCancelled(runId, promptTokens, completionTokens);
                return;
              }
              if (decision === "deny") {
                output = "Shell command denied by user";
                summary = "Denied";
                publish(runId, {
                  type: "tool_end",
                  tool: toolName,
                  summary,
                  output,
                });
                messages.push({
                  role: "tool",
                  tool_call_id: tc.id,
                  name: toolName,
                  content: output,
                });
                continue;
              }
            }
          }

          const toolResult = await runTool(workspaceId, toolName, args);

          if (toolResult.patchProposal) {
            const patch = patchStore.propose({
              runId,
              sessionId,
              path: toolResult.patchProposal.path,
              oldString: toolResult.patchProposal.oldString,
              newString: toolResult.patchProposal.newString,
              op: toolResult.patchProposal.op,
            });
            publish(runId, { type: "patch_proposed", patch });
            summary = `Proposed patch for ${patch.path}`;
            output = toolResult.output;
            publish(runId, {
              type: "tool_end",
              tool: toolName,
              summary,
              output: output.slice(0, 4000),
            });
          } else {
            summary = toolResult.summary;
            output = toolResult.output;
            publish(runId, {
              type: "tool_end",
              tool: toolName,
              summary,
              output: output.slice(0, 4000),
            });
          }

          messages.push({
            role: "tool",
            tool_call_id: tc.id,
            name: toolName,
            content: output.slice(0, 20_000),
          });
        }
        continue;
      }

      finalText = extractText(msg);
      // Publish token chunks for UI feel when non-stream JSON was used
      if (finalText) {
        const chunkSize = 24;
        for (let i = 0; i < finalText.length; i += chunkSize) {
          publish(runId, {
            type: "token",
            text: finalText.slice(i, i + chunkSize),
            channel: "assistant",
          });
        }
      }
      break;
    }

    // Final guard: do not publish done if cancel won the race after last step
    if (isCancelled(runId)) {
      finishCancelled(runId, promptTokens, completionTokens);
      return;
    }

    if (!finalText) finalText = "Done.";
    publish(runId, { type: "assistant", text: finalText });
    addMessage(sessionId, "assistant", finalText, {
      runId,
      usage: { prompt: promptTokens, completion: completionTokens },
    });
    recordUsage({
      model,
      prompt: promptTokens,
      completion: completionTokens || Math.ceil(finalText.length / 4),
      sessionId,
      source: "agent",
    });
    updateRun(runId, {
      status: "done",
      promptTokens,
      completionTokens,
    });
    publish(runId, {
      type: "done",
      usage: { prompt: promptTokens, completion: completionTokens },
    });
  } catch (e) {
    const err = e instanceof Error ? e.message : String(e);
    publish(runId, { type: "error", message: err });
    updateRun(runId, {
      status: "error",
      error: err,
      promptTokens,
      completionTokens,
    });
    try {
      addMessage(sessionId, "assistant", `Error: ${err}`);
    } catch {
      /* ignore */
    }
  } finally {
    runs.delete(runId);
  }
}

async function callLlm(opts: {
  fetchFn: typeof fetch;
  base: string;
  model: string;
  messages: ChatMsg[];
  tools: typeof toolDefinitions;
  signal?: AbortSignal;
}): Promise<LlmResult> {
  const { fetchFn, base, model, messages, tools, signal } = opts;

  const payload: Record<string, unknown> = {
    model,
    messages,
    stream: true,
    temperature: 0.3,
    // Laguna / Qwen-style reasoning: on by default for Composer so agentic
    // answers can use interleaved thinking. Per-request off still works via
    // models that honor chat_template_kwargs from the client; we force on here.
    chat_template_kwargs: { enable_thinking: true },
  };
  if (tools.length) {
    payload.tools = tools;
    payload.tool_choice = "auto";
  }

  let res: Response;
  try {
    res = await fetchFn(`${base}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(payload),
      signal: signal ?? AbortSignal.timeout(180_000),
    });
  } catch (e) {
    // Fallback: non-stream request (better mock/test reliability)
    if (signal?.aborted) throw e;
    res = await fetchFn(`${base}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...payload,
        stream: false,
      }),
      signal: signal ?? AbortSignal.timeout(180_000),
    });
  }

  if (!res.ok) {
    // Retry without tools on 400/404
    const t = await res.text();
    if (res.status === 400 || res.status === 404) {
      const plain = await fetchFn(`${base}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          messages: messages.map(({ role, content }) => ({ role, content })),
          stream: false,
          temperature: 0.3,
          chat_template_kwargs: { enable_thinking: true },
        }),
        signal: signal ?? AbortSignal.timeout(180_000),
      });
      if (plain.ok) {
        return await parseCompletionResponse(plain);
      }
    }
    throw new Error(`LLM error ${res.status}: ${t.slice(0, 400)}`);
  }

  return await parseCompletionResponse(res);
}

/**
 * Parse either SSE stream (stream:true) or a single JSON completion body.
 * Tests may return Response.json(...) even when stream:true was requested.
 */
async function parseCompletionResponse(res: Response): Promise<LlmResult> {
  const ct = (res.headers.get("content-type") || "").toLowerCase();

  // Non-stream JSON
  if (ct.includes("application/json")) {
    return parseJsonCompletion(await res.json());
  }

  // Try body as text — SSE or bare JSON
  const text = await res.text();
  const trimmed = text.trim();
  if (trimmed.startsWith("{")) {
    try {
      return parseJsonCompletion(JSON.parse(trimmed));
    } catch {
      /* fall through to SSE */
    }
  }

  // SSE parse
  return parseSseCompletion(trimmed);
}

function parseJsonCompletion(json: unknown): LlmResult {
  const j = json as {
    choices?: Array<{
      message?: LlmMessage;
      delta?: LlmMessage;
      finish_reason?: string;
    }>;
    usage?: { prompt_tokens?: number; completion_tokens?: number };
  };
  const choice = j.choices?.[0];
  const message = choice?.message ?? choice?.delta ?? null;
  return {
    message,
    usage: j.usage || {},
  };
}

function parseSseCompletion(raw: string): LlmResult {
  let content = "";
  let reasoning = "";
  const toolCalls = new Map<
    number,
    { id: string; type: string; function: { name: string; arguments: string } }
  >();
  let usage: { prompt_tokens?: number; completion_tokens?: number } = {};

  const lines = raw.split(/\r?\n/);
  for (const line of lines) {
    if (!line.startsWith("data:")) continue;
    const data = line.slice(5).trim();
    if (!data || data === "[DONE]") continue;
    let chunk: {
      choices?: Array<{
        delta?: {
          content?: string | null;
          reasoning?: string | null;
          tool_calls?: Array<{
            index?: number;
            id?: string;
            type?: string;
            function?: { name?: string; arguments?: string };
          }>;
        };
        message?: LlmMessage;
      }>;
      usage?: { prompt_tokens?: number; completion_tokens?: number };
    };
    try {
      chunk = JSON.parse(data);
    } catch {
      continue;
    }
    if (chunk.usage) usage = chunk.usage;
    const choice = chunk.choices?.[0];
    if (choice?.message) {
      // Some servers send full message in a final SSE event
      return {
        message: choice.message,
        usage: chunk.usage || usage,
      };
    }
    const delta = choice?.delta;
    if (!delta) continue;
    if (delta.content) content += delta.content;
    if (delta.reasoning) reasoning += delta.reasoning;
    if (delta.tool_calls) {
      for (const tc of delta.tool_calls) {
        const idx = tc.index ?? 0;
        const cur = toolCalls.get(idx) || {
          id: tc.id || `call_${idx}`,
          type: tc.type || "function",
          function: { name: "", arguments: "" },
        };
        if (tc.id) cur.id = tc.id;
        if (tc.type) cur.type = tc.type;
        if (tc.function?.name) cur.function.name += tc.function.name;
        if (tc.function?.arguments) cur.function.arguments += tc.function.arguments;
        toolCalls.set(idx, cur);
      }
    }
  }

  const tool_calls = [...toolCalls.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, v]) => v);

  return {
    message: {
      content: content || null,
      reasoning: reasoning || null,
      tool_calls: tool_calls.length ? tool_calls : undefined,
    },
    usage,
  };
}
