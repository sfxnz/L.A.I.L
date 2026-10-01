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
 * sessions, messages, agent_runs, patches) and usage_events (controller-side
 * metering that saw only proxied, non-streamed calls). They are left on disk
 * untouched; nothing creates, reads or writes them any more.
 */
function migrate(database: Database) {
  database.exec(`
    CREATE TABLE IF NOT EXISTS settings (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    -- Engine-metered token usage (controller/usage.ts): per-minute deltas of the
    -- backend's Prometheus counters, and the last reading each delta is taken from.
    CREATE TABLE IF NOT EXISTS usage_minutes (
      minute TEXT NOT NULL,
      model TEXT NOT NULL,
      prompt_tokens INTEGER NOT NULL DEFAULT 0,
      completion_tokens INTEGER NOT NULL DEFAULT 0,
      requests INTEGER NOT NULL DEFAULT 0,
      PRIMARY KEY (minute, model)
    );

    CREATE TABLE IF NOT EXISTS usage_counters (
      backend TEXT NOT NULL,
      model TEXT NOT NULL,
      prompt REAL NOT NULL,
      completion REAL NOT NULL,
      requests REAL NOT NULL,
      start_time REAL,
      PRIMARY KEY (backend, model)
    );
  `);
}
