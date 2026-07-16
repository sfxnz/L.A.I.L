import { randomUUID } from "crypto";
import { wsHub } from "../ws/hub";
import { addMessage, getSession, listMessages, updateSession } from "./sessions";
import { getWorkspace } from "./workspaces";
import { getSettings, openAiBase, resolveModelId } from "./settings";
import { recordUsage } from "./usage";
import { runTool, toolDefinitions } from "../tools";

const MAX_STEPS = 12;

function publish(runId: string, event: Record<string, unknown>) {
  wsHub.publish(`agent:${runId}`, { runId, ...event });
  wsHub.publish("agent", { runId, ...event });
}

export async function runAgent(opts: {
  sessionId: string;
  message: string;
  workspaceId?: string | null;
}): Promise<{ runId: string }> {
  const runId = randomUUID();
  const session = getSession(opts.sessionId);
  if (!session) throw new Error("Session not found");

  const workspaceId = opts.workspaceId || session.workspaceId;
  if (!workspaceId) {
    throw Object.assign(new Error("Session has no workspace linked"), {
      code: "NO_WORKSPACE",
      recovery: "Link a workspace before running the Composer agent.",
    });
  }
  const ws = getWorkspace(workspaceId);
  if (!ws) {
    throw Object.assign(new Error("Workspace not found"), {
      code: "WORKSPACE_NOT_FOUND",
    });
  }

  if (session.workspaceId !== workspaceId) {
    updateSession(opts.sessionId, { workspaceId });
  }

  addMessage(opts.sessionId, "user", opts.message);
  if (session.title === "New session") {
    updateSession(opts.sessionId, { title: opts.message.slice(0, 60) });
  }

  // Fire and forget agent loop
  (async () => {
    try {
      await agentLoop(runId, opts.sessionId, workspaceId, opts.message, ws.rootPath);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      publish(runId, { type: "error", message });
      addMessage(opts.sessionId, "assistant", `Error: ${message}`);
    }
  })();

  return { runId };
}

async function agentLoop(
  runId: string,
  sessionId: string,
  workspaceId: string,
  userMessage: string,
  rootPath: string,
) {
  const settings = getSettings();
  const base = openAiBase();
  let model: string;
  try {
    model = await resolveModelId();
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    publish(runId, { type: "error", message });
    addMessage(sessionId, "assistant", `Error: ${message}`);
    return;
  }

  publish(runId, {
    type: "thought",
    text: `Using model \`${model}\` · workspace ${rootPath}`,
  });
  publish(runId, { type: "status", text: "Working on your request…" });

  type Msg = {
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

  // Drop prior backend errors so they don't poison the model (e.g. old "model default" 404s)
  const history = listMessages(sessionId)
    .filter((m) => {
      if (m.role !== "user" && m.role !== "assistant") return false;
      if (m.role === "assistant" && /^Error:\s*LLM error/i.test(m.content)) return false;
      if (m.role === "assistant" && /model `default` does not exist/i.test(m.content)) return false;
      return true;
    })
    .map((m) => ({
      role: m.role,
      content: m.content,
    }));

  const messages: Msg[] = [
    {
      role: "system",
      content: `You are Composer, the L.A.I.L (Local AI Lab) coding agent.
You work ONLY inside the workspace root: ${rootPath}
You can read/write files, list dirs, grep, and run shell commands (cwd=workspace).
Style: think briefly, use tools, report progress. Prefer creating clear markdown docs when asked.
When fixing session/workspace issues, validate paths exist and stay under the workspace root.
After finishing, give a short summary of what you did.
If the user asks for a Local AI Survival Guide, write local-ai-survival-guide.md with practical local-LLM guidance (vLLM, llama.cpp, quantizations, hardware fit).
The backend model is already configured — never claim the model is "default" or missing unless a tool just failed.`,
    },
    ...history.slice(-20),
  ];

  // Ensure latest user message is present
  if (!messages.some((m) => m.role === "user" && m.content === userMessage)) {
    messages.push({ role: "user", content: userMessage });
  }

  let promptTokens = 0;
  let completionTokens = 0;
  let finalText = "";

  async function callLlm(payload: Record<string, unknown>): Promise<Response> {
    return fetch(`${base}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...payload,
        // Qwen3.6: keep answers out of the reasoning channel
        chat_template_kwargs: { enable_thinking: false },
      }),
      signal: AbortSignal.timeout(180_000),
    });
  }

  function extractText(msg: {
    content?: string | null;
    reasoning?: string | null;
  } | null | undefined): string {
    if (!msg) return "";
    if (msg.content && String(msg.content).trim()) return String(msg.content);
    if (msg.reasoning && String(msg.reasoning).trim()) return String(msg.reasoning);
    return "";
  }

  for (let step = 0; step < MAX_STEPS; step++) {
    const body = {
      model,
      messages,
      tools: toolDefinitions,
      tool_choice: "auto" as const,
      stream: false,
      temperature: 0.3,
    };

    let res: Response;
    try {
      res = await callLlm(body);
    } catch {
      // Offline fallback: deterministic tool-using demo for survival guide / file ops
      const fb = await offlineFallback(runId, workspaceId, userMessage);
      finalText = fb;
      break;
    }

    if (!res.ok) {
      const t = await res.text();
      // Model name wrong — re-resolve from /v1/models and retry once
      if (res.status === 404 && /model/i.test(t)) {
        try {
          model = await resolveModelId();
          publish(runId, { type: "status", text: `Retrying with model \`${model}\`…` });
          res = await callLlm({ ...body, model });
        } catch {
          /* keep original failure */
        }
      }
      if (!res.ok) {
        const errBody = res === undefined ? t : await res.clone().text().catch(() => t);
        // try without tools
        if (res.status === 400 || res.status === 404) {
          const plain = await callLlm({
            model,
            messages: messages.map(({ role, content }) => ({ role, content })),
            stream: false,
          });
          if (plain.ok) {
            const pj = (await plain.json()) as {
              choices?: Array<{ message?: { content?: string | null; reasoning?: string | null } }>;
              usage?: { prompt_tokens?: number; completion_tokens?: number };
            };
            finalText = extractText(pj.choices?.[0]?.message);
            promptTokens += pj.usage?.prompt_tokens || 0;
            completionTokens += pj.usage?.completion_tokens || 0;
            if (/survival|guide|create.*md/i.test(userMessage)) {
              await execNamed(runId, workspaceId, "write_file", {
                path: "local-ai-survival-guide.md",
                content: survivalGuideMarkdown(),
              });
            }
            break;
          }
        }
        throw new Error(`LLM error ${res.status}: ${(errBody || t).slice(0, 400)}`);
      }
    }

    const json = (await res.json()) as {
      choices?: Array<{
        message?: {
          content?: string | null;
          reasoning?: string | null;
          tool_calls?: Array<{
            id: string;
            type: string;
            function: { name: string; arguments: string };
          }>;
        };
        finish_reason?: string;
      }>;
      usage?: { prompt_tokens?: number; completion_tokens?: number };
    };

    promptTokens += json.usage?.prompt_tokens || 0;
    completionTokens += json.usage?.completion_tokens || 0;

    const msg = json.choices?.[0]?.message;
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
        let args: Record<string, unknown> = {};
        try {
          args = JSON.parse(tc.function.arguments || "{}");
        } catch {
          args = {};
        }
        publish(runId, {
          type: "tool_start",
          tool: tc.function.name,
          args,
        });
        const result = await runTool(workspaceId, tc.function.name, args);
        publish(runId, {
          type: "tool_end",
          tool: tc.function.name,
          summary: result.summary,
          output: result.output.slice(0, 4000),
        });
        if (result.fileWrite) {
          publish(runId, {
            type: "file_write",
            path: result.fileWrite.path,
            bytes: result.fileWrite.bytes,
          });
          publish(runId, {
            type: "status",
            text: `Working on ${result.fileWrite.path}…`,
          });
        }
        messages.push({
          role: "tool",
          tool_call_id: tc.id,
          name: tc.function.name,
          content: result.output.slice(0, 20_000),
        });
      }
      continue;
    }

    finalText = extractText(msg);
    break;
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
  publish(runId, {
    type: "done",
    usage: { prompt: promptTokens, completion: completionTokens },
  });
}

async function execNamed(
  runId: string,
  workspaceId: string,
  tool: string,
  args: Record<string, unknown>,
) {
  publish(runId, { type: "tool_start", tool, args });
  const result = await runTool(workspaceId, tool, args);
  publish(runId, {
    type: "tool_end",
    tool,
    summary: result.summary,
    output: result.output.slice(0, 4000),
  });
  if (result.fileWrite) {
    publish(runId, {
      type: "file_write",
      path: result.fileWrite.path,
      bytes: result.fileWrite.bytes,
    });
  }
  return result;
}

async function offlineFallback(runId: string, workspaceId: string, userMessage: string) {
  publish(runId, {
    type: "thought",
    text: "LLM backend unreachable — running local tool path for file work.",
  });
  publish(runId, { type: "status", text: "Working offline with tools…" });
  await execNamed(runId, workspaceId, "list_dir", { path: "." });
  if (/survival|guide|markdown|\.md/i.test(userMessage) || /create|write|generate/i.test(userMessage)) {
    await execNamed(runId, workspaceId, "write_file", {
      path: "local-ai-survival-guide.md",
      content: survivalGuideMarkdown(),
    });
    return "Created `local-ai-survival-guide.md` in the workspace (offline tool path — start vLLM or llama.cpp for full model reasoning).";
  }
  await execNamed(runId, workspaceId, "run_shell", { command: "ls -la" });
  return "Completed basic workspace inspection. Configure vLLM or llama.cpp under Configure for full Composer reasoning.";
}

function survivalGuideMarkdown() {
  return `# Local AI Survival Guide

Generated by **L.A.I.L Composer**.

## Stack choices (this lab)

| Backend | Best for | Notes |
|---------|----------|-------|
| **vLLM** | Throughput on NVIDIA / DGX Spark | OpenAI \`/v1\`, continuous batching, tool parsers — use **Server** tab |
| **llama.cpp** | GGUF on CPU / Apple Silicon / edge | Lightweight OpenAI-compatible server (default \`:8080\`) |

No Ollama — only vLLM and llama.cpp.

## Quantization cheatsheet

- **Q4_K_M / Q5_K_M (GGUF)** — llama.cpp sweet spot
- **FP8 / NVFP4** — high quality on modern NVIDIA (e.g. DGX Spark) via vLLM
- **AWQ / GPTQ** — common for vLLM
- Bigger isn't always better: match **active params** + **KV cache** to free RAM/VRAM

## Hardware fit rules of thumb

1. Leave headroom for the OS and KV cache (context length × concurrency).
2. On unified memory (Apple / Spark UMA), watch **available GiB** not just "GPU util".
3. Lab-safe bench util ≈ **0.4**; agent/workflow max ≈ **0.7–0.85** when dedicated.

## L.A.I.L workflow

1. **Configure** → vLLM (\`:8000\`) or llama.cpp (\`:8080\`) + default model  
2. **Models** → search HF / download weights into \`data/models\`  
3. **Server** → start vLLM Lab Safe or Workflow Max  
4. **Workbench** → Composer agent for multi-step file work  
5. **Usage** → track lifetime tokens locally  

## Quick commands

\`\`\`bash
# vLLM (example) — or use Server → Start serve in the UI
# docker / spark_lab as configured on this host

# llama.cpp server (example)
./llama-server -m model.gguf --port 8080

# Point L.A.I.L
export LAIL_DEFAULT_BACKEND=vllm
export LAIL_VLLM_URL=http://127.0.0.1:8000
# or: export LAIL_DEFAULT_BACKEND=llamacpp
#     export LAIL_LLAMACPP_URL=http://127.0.0.1:8080
bun run dev
\`\`\`

## Safety

- Agent tools are sandboxed to the workspace root  
- Prefer explicit quant + max-model-len over "max everything"  
- Never publish throughput numbers if smoke tests fail  

Stay local. Stay curious.
`;
}
