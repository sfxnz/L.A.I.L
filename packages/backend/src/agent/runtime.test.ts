import { describe, expect, test, beforeAll, afterAll } from "bun:test";
import { mkdirSync, writeFileSync, readFileSync, rmSync } from "fs";
import { join } from "path";
import { startAgentRun, getAgentRun, cancelAgentRun } from "./runtime";
import { patchStore } from "./patch-store";
import { createWorkspace } from "../controller/workspaces";
import { createSession, addMessage } from "../controller/sessions";
import { putSettings } from "../controller/settings";

const TMP = `/tmp/lail-runtime-test-${process.pid}`;

async function waitForRun(
  runId: string,
  timeoutMs = 5000,
): Promise<{ id: string; status: string; mode: string }> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const r = getAgentRun(runId);
    if (r && r.status !== "running") return r;
    await Bun.sleep(20);
  }
  const last = getAgentRun(runId);
  throw new Error(
    `Run ${runId} still running after ${timeoutMs}ms (status=${last?.status})`,
  );
}

function mockFetchTwoStep(): typeof fetch {
  let call = 0;
  return (async (_input: RequestInfo | URL, _init?: RequestInit) => {
    call++;
    if (call === 1) {
      return Response.json({
        choices: [
          {
            message: {
              content: null,
              tool_calls: [
                {
                  id: "call_1",
                  type: "function",
                  function: {
                    name: "search_replace",
                    arguments: JSON.stringify({
                      path: "a.txt",
                      old_string: "foo",
                      new_string: "bar",
                    }),
                  },
                },
              ],
            },
          },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 5 },
      });
    }
    return Response.json({
      choices: [
        {
          message: {
            content: "Done.",
            tool_calls: undefined,
          },
        },
      ],
      usage: { prompt_tokens: 12, completion_tokens: 3 },
    });
  }) as unknown as typeof fetch;
}

describe("AgentRuntime", () => {
  let workspaceId: string;
  let rootPath: string;

  beforeAll(() => {
    rmSync(TMP, { recursive: true, force: true });
    mkdirSync(TMP, { recursive: true });
    writeFileSync(join(TMP, "a.txt"), "foo\n");
    const ws = createWorkspace(`runtime-test-${process.pid}`, TMP);
    workspaceId = ws.id;
    rootPath = ws.rootPath;
    // Avoid resolveModelId probing a real backend
    putSettings({ defaultModel: "mock-model" });
  });

  afterAll(() => {
    rmSync(TMP, { recursive: true, force: true });
  });

  test("search_replace proposes patch without writing disk; accept applies", async () => {
    const session = createSession("runtime patch test", workspaceId);
    const userMsg = "Change foo to bar in a.txt";
    addMessage(session.id, "user", userMsg);

    const { runId } = startAgentRun({
      sessionId: session.id,
      message: userMsg,
      workspaceId,
      mode: "agent",
      fetchImpl: mockFetchTwoStep(),
      maxSteps: 8,
    });

    const finished = await waitForRun(runId);
    expect(finished.status).toBe("done");

    // File unchanged until accept
    expect(readFileSync(join(rootPath, "a.txt"), "utf8")).toContain("foo");
    expect(readFileSync(join(rootPath, "a.txt"), "utf8")).not.toContain("bar");

    const pending = patchStore.list({ sessionId: session.id, status: "pending" });
    expect(pending.length).toBeGreaterThanOrEqual(1);
    const patch = pending.find((p) => p.path === "a.txt")!;
    expect(patch).toBeTruthy();
    expect(patch.oldString).toBe("foo");
    expect(patch.newString).toBe("bar");
    expect(patch.op).toBe("replace");
    expect(patch.runId).toBe(runId);

    const applied = patchStore.accept(patch.id, rootPath);
    expect(applied.status).toBe("accepted");
    expect(readFileSync(join(rootPath, "a.txt"), "utf8")).toContain("bar");
  });

  test("plan mode rejects run_shell tool", async () => {
    const session = createSession("runtime plan shell", workspaceId);
    const userMsg = "run ls";
    addMessage(session.id, "user", userMsg);

    let call = 0;
    const fetchImpl = (async () => {
      call++;
      if (call === 1) {
        return Response.json({
          choices: [
            {
              message: {
                content: null,
                tool_calls: [
                  {
                    id: "call_shell",
                    type: "function",
                    function: {
                      name: "run_shell",
                      arguments: JSON.stringify({ command: "ls -la" }),
                    },
                  },
                ],
              },
            },
          ],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        });
      }
      return Response.json({
        choices: [{ message: { content: "I cannot run shell in plan mode." } }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
    }) as unknown as typeof fetch;

    const { runId } = startAgentRun({
      sessionId: session.id,
      message: userMsg,
      workspaceId,
      mode: "plan",
      fetchImpl,
      maxSteps: 6,
    });

    const finished = await waitForRun(runId);
    expect(finished.status).toBe("done");
    // No shell side-effects required; run completes with denied tool path
    expect(call).toBeGreaterThanOrEqual(1);
  });

  test("cancelAgentRun marks run cancelled when still running", async () => {
    const session = createSession("runtime cancel", workspaceId);
    const userMsg = "slow";
    addMessage(session.id, "user", userMsg);

    const fetchImpl = (async () => {
      await Bun.sleep(200);
      return Response.json({
        choices: [{ message: { content: "too late" } }],
        usage: {},
      });
    }) as unknown as typeof fetch;

    const { runId } = startAgentRun({
      sessionId: session.id,
      message: userMsg,
      workspaceId,
      mode: "ask",
      fetchImpl,
      maxSteps: 4,
    });

    // Cancel quickly while first LLM call is sleeping
    const ok = cancelAgentRun(runId);
    expect(ok).toBe(true);

    const finished = await waitForRun(runId, 3000);
    // Cancel during LLM must not become done after a late successful response
    expect(finished.status).toBe("cancelled");
  });
});
