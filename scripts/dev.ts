#!/usr/bin/env bun
/**
 * One-command L.A.I.L: serve-engine (Python) + controller (Bun) + web (Next).
 *
 * `bun run dev` serves the web app with `next dev` (hot reload, React dev build).
 * `bun run start:prod` (LAIL_WEB_MODE=prod) builds it once and serves it with
 * `next start`: the minified production build, for the console left open all day.
 * The production build goes to apps/web/.next-prod, so it never collides with a
 * `next dev` running on the same tree.
 */
import { spawn, type Subprocess } from "bun";
import { existsSync, mkdirSync } from "fs";
import { resolve, join } from "path";

const root = resolve(import.meta.dir, "..");
const host = process.env.LAIL_HOST || "127.0.0.1";
const apiPort = process.env.LAIL_API_PORT || "8787";
const webPort = process.env.LAIL_WEB_PORT || "3000";
const servePort = process.env.LAIL_SERVE_ENGINE_PORT || "8765";
const dataDir = resolve(process.env.LAIL_DATA_DIR || join(root, "data"));
const webProd = process.env.LAIL_WEB_MODE === "prod";

mkdirSync(dataDir, { recursive: true });

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

// Web env — the rewrite target (Next → controller on this host) is read at build
// time for `next start`, so the production build gets the same env as the server.
// The browser only ever talks to Next (same origin), so nothing here is baked for it.
const webDir = join(root, "apps/web");
const webEnv: Record<string, string> = {
  PORT: webPort,
  LAIL_API_URL: `http://127.0.0.1:${apiPort}`,
  NEXT_PUBLIC_LAIL_API: `http://127.0.0.1:${apiPort}`,
  LAIL_TOKEN: process.env.LAIL_TOKEN || "",
  ...(webProd ? { LAIL_NEXT_DIST: process.env.LAIL_NEXT_DIST || ".next-prod" } : {}),
};

if (webProd) {
  console.log("→ building web for production (next build)…");
  const build = Bun.spawnSync({
    cmd: ["bun", "run", "build"],
    cwd: webDir,
    env: { ...process.env, ...webEnv },
    stdout: "inherit",
    stderr: "inherit",
  });
  if (build.exitCode !== 0) {
    console.error("web build failed — nothing started");
    process.exit(build.exitCode ?? 1);
  }
}

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
    LAB_HOST: host,
    LAIL_HOST: host,
    LAIL_TOKEN: process.env.LAIL_TOKEN || "",
    LAIL_INSECURE_BIND: process.env.LAIL_INSECURE_BIND || "",
    // uv tool install puts CLI here — needed for tool-eval-bench jobs
    PATH: `${process.env.HOME || ""}/.local/bin:${process.env.PATH || ""}`,
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
    LAIL_TOKEN: process.env.LAIL_TOKEN || "",
    LAIL_INSECURE_BIND: process.env.LAIL_INSECURE_BIND || "",
    LAIL_CORS_ORIGINS: process.env.LAIL_CORS_ORIGINS || "",
    LAIL_SERVE_ENGINE_URL: `http://127.0.0.1:${servePort}`,
    LAIL_ROOT: root,
    LAIL_DATA_DIR: dataDir,
    LAIL_SHARE_PUBLIC_BASE: process.env.LAIL_SHARE_PUBLIC_BASE || "",
  },
);

// Artifacts-only share server (loopback) — Funnel this, never :3000
const sharePort = process.env.LAIL_SHARE_PORT || "8791";
run(
  "lab-public-share",
  ["bun", "run", "scripts/lab-public-server.ts"],
  root,
  {
    LAIL_ROOT: root,
    LAIL_DATA_DIR: dataDir,
    LAIL_SHARE_HOST: "127.0.0.1",
    LAIL_SHARE_PORT: sharePort,
    LAIL_LAB_PUBLIC_DIR: join(dataDir, "lab-public"),
  },
);

// Next web
run("web", ["bun", "run", webProd ? "start" : "dev", "--", "-H", host, "-p", webPort], webDir, webEnv);

console.log(`
╔══════════════════════════════════════════════╗
║  L.A.I.L — Local AI Lab                      ║
║  Web:          http://127.0.0.1:${webPort} (${webProd ? "prod" : "dev"})     ║
║  Controller:   http://127.0.0.1:${apiPort}            ║
║  Serve-engine: http://127.0.0.1:${servePort}            ║
║  Public share: http://127.0.0.1:${sharePort} (artifacts) ║
╚══════════════════════════════════════════════╝
`);

// Keep alive
await Promise.race(children.map((c) => c.exited));
shutdown();
