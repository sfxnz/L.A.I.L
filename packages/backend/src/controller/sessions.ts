import { randomUUID } from "crypto";
import type { ChatMessage, Session } from "@lail/shared";
import { getDb } from "../db/schema";

function now() {
  return new Date().toISOString();
}

function rowToSession(r: Record<string, unknown>): Session {
  return {
    id: String(r.id),
    title: String(r.title),
    workspaceId: r.workspace_id ? String(r.workspace_id) : null,
    pinned: Boolean(r.pinned),
    createdAt: String(r.created_at),
    updatedAt: String(r.updated_at),
  };
}

export function listSessions(): Session[] {
  return (
    getDb()
      .query("SELECT * FROM sessions ORDER BY pinned DESC, updated_at DESC")
      .all() as Record<string, unknown>[]
  ).map(rowToSession);
}

export function getSession(id: string): Session | null {
  const row = getDb().query("SELECT * FROM sessions WHERE id = ?").get(id) as
    | Record<string, unknown>
    | null;
  return row ? rowToSession(row) : null;
}

export function createSession(title = "New session", workspaceId: string | null = null): Session {
  const id = randomUUID();
  const ts = now();
  getDb()
    .query(
      `INSERT INTO sessions (id, title, workspace_id, pinned, created_at, updated_at)
       VALUES (?, ?, ?, 0, ?, ?)`,
    )
    .run(id, title, workspaceId, ts, ts);
  return getSession(id)!;
}

export function updateSession(
  id: string,
  patch: Partial<{ title: string; workspaceId: string | null; pinned: boolean }>,
): Session | null {
  const cur = getSession(id);
  if (!cur) return null;
  getDb()
    .query(
      `UPDATE sessions SET title = ?, workspace_id = ?, pinned = ?, updated_at = ? WHERE id = ?`,
    )
    .run(
      patch.title ?? cur.title,
      patch.workspaceId !== undefined ? patch.workspaceId : cur.workspaceId,
      (patch.pinned ?? cur.pinned) ? 1 : 0,
      now(),
      id,
    );
  return getSession(id);
}

export function listMessages(sessionId: string): ChatMessage[] {
  const rows = getDb()
    .query("SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC")
    .all(sessionId) as Record<string, unknown>[];
  return rows.map((r) => ({
    id: String(r.id),
    sessionId: String(r.session_id),
    role: r.role as ChatMessage["role"],
    content: String(r.content),
    createdAt: String(r.created_at),
    meta: r.meta ? JSON.parse(String(r.meta)) : undefined,
  }));
}

export function addMessage(
  sessionId: string,
  role: ChatMessage["role"],
  content: string,
  meta?: Record<string, unknown>,
): ChatMessage {
  const id = randomUUID();
  const ts = now();
  getDb()
    .query(
      `INSERT INTO messages (id, session_id, role, content, meta, created_at) VALUES (?, ?, ?, ?, ?, ?)`,
    )
    .run(id, sessionId, role, content, meta ? JSON.stringify(meta) : null, ts);
  getDb().query(`UPDATE sessions SET updated_at = ? WHERE id = ?`).run(ts, sessionId);
  return { id, sessionId, role, content, createdAt: ts, meta };
}
