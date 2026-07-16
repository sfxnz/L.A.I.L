import type { Patch, PatchStatus } from "@lail/shared";
import { patchStore } from "../agent/patch-store";
import { getWorkspace } from "./workspaces";
import { getSession } from "./sessions";
import { wsHub } from "../ws/hub";

export function listPatches(q: {
  sessionId?: string;
  runId?: string;
  status?: string;
}): Patch[] {
  return patchStore.list({
    sessionId: q.sessionId,
    runId: q.runId,
    status: q.status as PatchStatus | undefined,
  });
}

function workspaceRootForSession(sessionId: string): string {
  const session = getSession(sessionId);
  if (!session?.workspaceId) throw new Error("No workspace");
  const ws = getWorkspace(session.workspaceId);
  if (!ws) throw new Error("Workspace missing");
  return ws.rootPath;
}

function publishPatchUpdate(patch: Patch) {
  wsHub.publish(`agent:${patch.runId}`, {
    runId: patch.runId,
    type: "patch_updated",
    patch,
  });
  if (patch.status === "accepted") {
    wsHub.publish(`agent:${patch.runId}`, {
      runId: patch.runId,
      type: "file_write",
      path: patch.path,
      bytes: Buffer.byteLength(patch.newString),
    });
  }
}

export function acceptPatch(id: string): Patch {
  const patch = patchStore.get(id);
  if (!patch) throw Object.assign(new Error("not found"), { code: "NOT_FOUND" });
  const root = workspaceRootForSession(patch.sessionId);
  const updated = patchStore.accept(id, root);
  publishPatchUpdate(updated);
  return updated;
}

export function rejectPatch(id: string): Patch {
  const patch = patchStore.get(id);
  if (!patch) throw Object.assign(new Error("not found"), { code: "NOT_FOUND" });
  const updated = patchStore.reject(id);
  publishPatchUpdate(updated);
  return updated;
}

export function acceptAllPatches(body: {
  sessionId?: string;
  runId?: string;
}): Patch[] {
  let sessionId = body.sessionId;
  if (!sessionId && body.runId) {
    const patches = patchStore.list({ runId: body.runId });
    sessionId = patches[0]?.sessionId;
  }
  if (!sessionId) {
    throw Object.assign(new Error("sessionId or runId required"), {
      code: "BAD_REQUEST",
    });
  }
  const root = workspaceRootForSession(sessionId);
  const updated = patchStore.acceptAll(
    { sessionId: body.sessionId, runId: body.runId },
    root,
  );
  for (const p of updated) {
    publishPatchUpdate(p);
  }
  return updated;
}
