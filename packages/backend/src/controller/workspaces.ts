import { randomUUID } from "crypto";
import { existsSync, mkdirSync, readdirSync, statSync } from "fs";
import { join, resolve, relative, sep } from "path";
import type { TreeNode, Workspace } from "@lail/shared";
import { getDb } from "../db/schema";
import { config } from "../config";

function now() {
  return new Date().toISOString();
}

function rowToWs(r: Record<string, unknown>): Workspace {
  return {
    id: String(r.id),
    name: String(r.name),
    rootPath: String(r.root_path),
    pinned: Boolean(r.pinned),
    createdAt: String(r.created_at),
    updatedAt: String(r.updated_at),
  };
}

export function ensureDefaultWorkspace(): Workspace {
  mkdirSync(config.workspacesDir, { recursive: true });
  const demo = resolve(config.workspacesDir, "demo");
  mkdirSync(demo, { recursive: true });
  const existing = listWorkspaces().find((w) => w.rootPath === demo);
  if (existing) return existing;
  return createWorkspace("demo", demo);
}

export function listWorkspaces(): Workspace[] {
  const rows = getDb()
    .query("SELECT * FROM workspaces ORDER BY pinned DESC, updated_at DESC")
    .all() as Record<string, unknown>[];
  return rows.map(rowToWs);
}

export function getWorkspace(id: string): Workspace | null {
  const row = getDb().query("SELECT * FROM workspaces WHERE id = ?").get(id) as
    | Record<string, unknown>
    | null;
  return row ? rowToWs(row) : null;
}

export function createWorkspace(name: string, rootPath?: string): Workspace {
  const id = randomUUID();
  const path = resolve(rootPath || join(config.workspacesDir, name.replace(/[^\w.-]+/g, "-")));
  mkdirSync(path, { recursive: true });
  const ts = now();
  getDb()
    .query(
      `INSERT INTO workspaces (id, name, root_path, pinned, created_at, updated_at)
       VALUES (?, ?, ?, 0, ?, ?)`,
    )
    .run(id, name, path, ts, ts);
  return getWorkspace(id)!;
}

export function updateWorkspace(
  id: string,
  patch: Partial<{ name: string; rootPath: string; pinned: boolean }>,
): Workspace | null {
  const cur = getWorkspace(id);
  if (!cur) return null;
  const name = patch.name ?? cur.name;
  const rootPath = patch.rootPath ? resolve(patch.rootPath) : cur.rootPath;
  const pinned = patch.pinned ?? cur.pinned;
  if (patch.rootPath && !existsSync(rootPath)) {
    throw Object.assign(new Error(`Workspace path does not exist: ${rootPath}`), {
      code: "WORKSPACE_PATH_MISSING",
      recovery: "Create the directory or pick an existing path.",
    });
  }
  getDb()
    .query(
      `UPDATE workspaces SET name = ?, root_path = ?, pinned = ?, updated_at = ? WHERE id = ?`,
    )
    .run(name, rootPath, pinned ? 1 : 0, now(), id);
  return getWorkspace(id);
}

/** Resolve path under workspace root; throw if escape. */
export function resolveInWorkspace(workspaceId: string, relPath: string): string {
  const ws = getWorkspace(workspaceId);
  if (!ws) {
    throw Object.assign(new Error(`Unknown workspace: ${workspaceId}`), {
      code: "WORKSPACE_NOT_FOUND",
    });
  }
  if (!existsSync(ws.rootPath)) {
    throw Object.assign(new Error(`Workspace root missing: ${ws.rootPath}`), {
      code: "WORKSPACE_PATH_MISSING",
      recovery: "Re-link the workspace path under Configure or Projects.",
      workspaceId,
      rootPath: ws.rootPath,
    });
  }
  const root = resolve(ws.rootPath);
  const target = resolve(root, relPath || ".");
  const rel = relative(root, target);
  if (rel.startsWith("..") || rel.includes(`..${sep}`) || (rel !== "" && resolve(root, rel) !== target)) {
    throw Object.assign(new Error("Path escapes workspace"), { code: "PATH_ESCAPE" });
  }
  if (target !== root && !target.startsWith(root + sep)) {
    throw Object.assign(new Error("Path escapes workspace"), { code: "PATH_ESCAPE" });
  }
  return target;
}

export function buildTree(workspaceId: string, maxDepth = 4): TreeNode[] {
  const ws = getWorkspace(workspaceId);
  if (!ws || !existsSync(ws.rootPath)) return [];
  return walk(ws.rootPath, ws.rootPath, 0, maxDepth);
}

function walk(abs: string, root: string, depth: number, maxDepth: number): TreeNode[] {
  if (depth >= maxDepth) return [];
  let entries: string[];
  try {
    entries = readdirSync(abs);
  } catch {
    return [];
  }
  const nodes: TreeNode[] = [];
  for (const name of entries.sort()) {
    if (name === "node_modules" || name === ".git" || name === ".next") continue;
    const full = join(abs, name);
    let st;
    try {
      st = statSync(full);
    } catch {
      continue;
    }
    const rel = relative(root, full) || name;
    if (st.isDirectory()) {
      nodes.push({
        name,
        path: rel,
        type: "dir",
        children: walk(full, root, depth + 1, maxDepth),
      });
    } else {
      nodes.push({ name, path: rel, type: "file" });
    }
  }
  return nodes;
}
