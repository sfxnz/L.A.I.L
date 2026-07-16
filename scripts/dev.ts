#!/usr/bin/env bun
/**
 * One-command L.A.I.L dev: serve-engine (Python) + controller (Bun) + web (Next).
 */
import { spawn, type Subprocess } from "bun";
import { existsSync, mkdirSync } from "fs";
import { resolve, join } from "path";

const root = resolve(import.meta.dir, "..");
const host = process.env.LAIL_HOST || "0.0.0.0";
const apiPort = process.env.LAIL_API_PORT || "8787";
const webPort = process.env.LAIL_WEB_PORT || "3000";
const servePort = process.env.LAIL_SERVE_ENGINE_PORT || "8765";
const dataDir = resolve(process.env.LAIL_DATA_DIR || join(root, "data"));

mkdirSync(dataDir, { recursive: true });
mkdirSync(join(root, "workspaces/demo"), { recursive: true });

process.env.LAIL_ROOT = root;
process.env.LOCAL_AI_LAB_ROOT = root;
process.env.LAIL_DATA_DIR = dataDir;
process.env.LAIL_SERVE_ENGINE_URL = `http://127.0.0.1:${servePort}`;
process.env.LAB_API_PORT = servePort;

const children: Subprocess[] = [];

function run(name: string, cmd: string[], cwd: string, env: Record<string, string> = {}) {
  console.log(`→ starting ${name}: ${cmd.join(" ")}`);
  const proc = spawn({
    cmd,
    cwd,
    env: { ...process.env, ...env },
    stdout: "inherit",
    stderr: "inherit",
    stdin: "inherit",
  });
  children.push(proc);
  return proc;
}

function shutdown() {
  console.log("\nShutting down L.A.I.L…");
  for (const c of children) {
    try {
      c.kill();
    } catch {
      /* */
    }
  }
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

// Python serve-engine
const seRoot = join(root, "packages/serve-engine");
const venvPython = join(root, ".venv/bin/python");
const python = existsSync(venvPython) ? venvPython : "python3";

run(
  "serve-engine",
  [
    python,
    "-m",
    "uvicorn",
    "app.main:app",
    "--host",
    host,
    "--port",
    servePort,
  ],
  seRoot,
  {
    PYTHONPATH: seRoot,
    LOCAL_AI_LAB_ROOT: root,
    LAIL_DATA_DIR: dataDir,
    LAB_API_PORT: servePort,
  },
);

// Bun controller
run(
  "controller",
  ["bun", "run", "src/index.ts"],
  join(root, "packages/backend"),
  {
    LAIL_HOST: host,
    LAIL_API_PORT: apiPort,
    LAIL_SERVE_ENGINE_URL: `http://127.0.0.1:${servePort}`,
    LAIL_ROOT: root,
    LAIL_DATA_DIR: dataDir,
  },
);

// Next web
run(
  "web",
  ["bun", "run", "dev", "--", "-H", host, "-p", webPort],
  join(root, "apps/web"),
  {
    PORT: webPort,
    NEXT_PUBLIC_LAIL_API: `http://127.0.0.1:${apiPort}`,
    NEXT_PUBLIC_LAIL_WS: `ws://127.0.0.1:${apiPort}/ws`,
  },
);

console.log(`
╔══════════════════════════════════════════════╗
║  L.A.I.L — Local AI Lab                      ║
║  Web:          http://127.0.0.1:${webPort}            ║
║  Controller:   http://127.0.0.1:${apiPort}            ║
║  Serve-engine: http://127.0.0.1:${servePort}            ║
╚══════════════════════════════════════════════╝
`);

// Keep alive
await Promise.race(children.map((c) => c.exited));
shutdown();
