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

/**
 * Older databases also hold the retired agent/workbench tables (workspaces,
 * sessions, messages, agent_runs, patches). They are left on disk untouched;
 * nothing creates, reads or writes them any more.
 */
function migrate(database: Database) {
  database.exec(`
    CREATE TABLE IF NOT EXISTS settings (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
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
  `);
}
