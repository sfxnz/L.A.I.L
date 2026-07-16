import { Database } from "bun:sqlite";
import { mkdirSync } from "fs";
import { dirname } from "path";
import { config } from "../config";

let db: Database | null = null;

export function getDb(): Database {
  if (db) return db;
  mkdirSync(dirname(config.dbPath), { recursive: true });
  db = new Database(config.dbPath, { create: true });
  db.exec("PRAGMA journal_mode = WAL;");
  migrate(db);
  return db;
}

function migrate(database: Database) {
  database.exec(`
    CREATE TABLE IF NOT EXISTS settings (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS workspaces (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      root_path TEXT NOT NULL UNIQUE,
      pinned INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      workspace_id TEXT,
      pinned INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
    );

    CREATE TABLE IF NOT EXISTS messages (
      id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      meta TEXT,
      created_at TEXT NOT NULL,
      FOREIGN KEY (session_id) REFERENCES sessions(id)
    );

    CREATE TABLE IF NOT EXISTS usage_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts TEXT NOT NULL,
      model TEXT NOT NULL,
      prompt_tokens INTEGER NOT NULL DEFAULT 0,
      completion_tokens INTEGER NOT NULL DEFAULT 0,
      session_id TEXT,
      source TEXT NOT NULL DEFAULT 'proxy'
    );

    CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_events(ts);
    CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

    CREATE TABLE IF NOT EXISTS agent_runs (
      id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      workspace_id TEXT NOT NULL,
      mode TEXT NOT NULL,
      status TEXT NOT NULL,
      message TEXT NOT NULL,
      error TEXT,
      prompt_tokens INTEGER NOT NULL DEFAULT 0,
      completion_tokens INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS patches (
      id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      session_id TEXT NOT NULL,
      path TEXT NOT NULL,
      old_string TEXT NOT NULL,
      new_string TEXT NOT NULL,
      op TEXT NOT NULL,
      status TEXT NOT NULL,
      reason TEXT,
      created_at TEXT NOT NULL,
      resolved_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_patches_session_status ON patches(session_id, status);
    CREATE INDEX IF NOT EXISTS idx_patches_run ON patches(run_id);
    CREATE INDEX IF NOT EXISTS idx_agent_runs_session ON agent_runs(session_id);
  `);
}
