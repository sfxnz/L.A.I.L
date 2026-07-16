import { randomUUID } from "crypto";
import { existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from "fs";
import { dirname, join } from "path";
import type { Patch, PatchOp, PatchStatus } from "@lail/shared";
import { getDb } from "../db/schema";
import { assertWorkspaceRelativePath } from "./tool-policy";

export function applySearchReplaceToContent(
  content: string,
  oldString: string,
  newString: string,
): { ok: true; content: string } | { ok: false; reason: "no_match" | "ambiguous" } {
  if (oldString === "") {
    return { ok: false, reason: "no_match" };
  }
  let count = 0;
  let idx = 0;
  while (true) {
    const found = content.indexOf(oldString, idx);
    if (found === -1) break;
    count++;
    idx = found + oldString.length;
    if (count > 1) return { ok: false, reason: "ambiguous" };
  }
  if (count === 0) return { ok: false, reason: "no_match" };
  return { ok: true, content: content.replace(oldString, newString) };
}

function rowToPatch(r: Record<string, unknown>): Patch {
  return {
    id: String(r.id),
    runId: String(r.run_id),
    sessionId: String(r.session_id),
    path: String(r.path),
    oldString: String(r.old_string),
    newString: String(r.new_string),
    op: String(r.op) as PatchOp,
    status: String(r.status) as PatchStatus,
    reason: r.reason != null ? String(r.reason) : undefined,
    createdAt: String(r.created_at),
    resolvedAt: r.resolved_at != null ? String(r.resolved_at) : undefined,
  };
}

export type ProposeInput = {
  runId: string;
  sessionId: string;
  path: string;
  oldString: string;
  newString: string;
  op: PatchOp;
  workspaceRoot?: string; // unused at propose; validate path only
};

export function createPatchStore() {
  return {
    propose(input: ProposeInput): Patch {
      const path = assertWorkspaceRelativePath(input.path);
      const id = randomUUID();
      const ts = new Date().toISOString();
      getDb()
        .query(
          `INSERT INTO patches
           (id, run_id, session_id, path, old_string, new_string, op, status, reason, created_at, resolved_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL, ?, NULL)`,
        )
        .run(
          id,
          input.runId,
          input.sessionId,
          path,
          input.oldString,
          input.newString,
          input.op,
          ts,
        );
      return this.get(id)!;
    },

    get(id: string): Patch | null {
      const row = getDb().query("SELECT * FROM patches WHERE id = ?").get(id) as
        | Record<string, unknown>
        | null;
      return row ? rowToPatch(row) : null;
    },

    list(opts: { sessionId?: string; runId?: string; status?: PatchStatus }): Patch[] {
      let sql = "SELECT * FROM patches WHERE 1=1";
      const params: string[] = [];
      if (opts.sessionId) {
        sql += " AND session_id = ?";
        params.push(opts.sessionId);
      }
      if (opts.runId) {
        sql += " AND run_id = ?";
        params.push(opts.runId);
      }
      if (opts.status) {
        sql += " AND status = ?";
        params.push(opts.status);
      }
      sql += " ORDER BY created_at ASC";
      const rows = getDb().query(sql).all(...params) as Record<string, unknown>[];
      return rows.map(rowToPatch);
    },

    reject(id: string): Patch {
      const ts = new Date().toISOString();
      getDb()
        .query(
          `UPDATE patches SET status = 'rejected', resolved_at = ? WHERE id = ? AND status = 'pending'`,
        )
        .run(ts, id);
      return this.get(id)!;
    },

    accept(id: string, workspaceRoot: string): Patch {
      const patch = this.get(id);
      if (!patch) throw new Error("Patch not found");
      if (patch.status !== "pending") return patch;

      const abs = join(workspaceRoot, patch.path);
      try {
        if (patch.op === "create") {
          if (existsSync(abs)) {
            return this.fail(id, "exists");
          }
          mkdirSync(dirname(abs), { recursive: true });
          writeFileSync(abs, patch.newString, "utf8");
        } else if (patch.op === "delete") {
          if (!existsSync(abs)) return this.fail(id, "no_match");
          unlinkSync(abs);
        } else {
          // replace
          if (!existsSync(abs)) return this.fail(id, "no_match");
          const cur = readFileSync(abs, "utf8");
          const result = applySearchReplaceToContent(cur, patch.oldString, patch.newString);
          if (!result.ok) return this.fail(id, result.reason);
          writeFileSync(abs, result.content, "utf8");
        }
      } catch (e) {
        return this.fail(id, e instanceof Error ? e.message : "apply_error");
      }

      const ts = new Date().toISOString();
      getDb()
        .query(`UPDATE patches SET status = 'accepted', resolved_at = ? WHERE id = ?`)
        .run(ts, id);
      return this.get(id)!;
    },

    fail(id: string, reason: string): Patch {
      const ts = new Date().toISOString();
      getDb()
        .query(
          `UPDATE patches SET status = 'failed', reason = ?, resolved_at = ? WHERE id = ?`,
        )
        .run(reason, ts, id);
      return this.get(id)!;
    },

    acceptAll(opts: { sessionId?: string; runId?: string }, workspaceRoot: string): Patch[] {
      const pending = this.list({ ...opts, status: "pending" });
      return pending.map((p) => this.accept(p.id, workspaceRoot));
    },
  };
}

export const patchStore = createPatchStore();
