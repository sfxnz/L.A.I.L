import type { AgentMode } from "@lail/shared";
import { startAgentRun, cancelAgentRun, getAgentRun } from "../agent/runtime";
import { addMessage, getSession, updateSession } from "./sessions";
import { getWorkspace } from "./workspaces";

export async function runAgent(opts: {
  sessionId: string;
  message: string;
  workspaceId?: string | null;
  mode?: AgentMode;
}): Promise<{ runId: string }> {
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

  return startAgentRun({
    sessionId: opts.sessionId,
    message: opts.message,
    workspaceId,
    mode: opts.mode ?? "agent",
  });
}

export { cancelAgentRun, getAgentRun };
