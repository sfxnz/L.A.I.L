import { createHash, randomBytes } from "crypto";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  statSync,
  writeFileSync,
} from "fs";
import { basename, dirname, extname, join, relative, resolve, sep } from "path";
import { config } from "../config";

export type LabRunMeta = {
  id: string;
  kind: "hermes_task";
  task_type: string;
  title: string;
  model_id: string;
  serve?: Record<string, unknown>;
  eval_run_id?: string | null;
  hermes?: { session?: string; source?: string } | null;
  created_at: string;
  entry: string;
  share: { public: boolean; slug: string | null };
  tags: string[];
  brief?: string;
  task_fingerprint?: string;
};

export type LabRunSummary = LabRunMeta & {
  dir: string;
  preview_url: string | null;
  play_url: string;
};

function labRoot(): string {
  const d = join(config.dataDir, "lab-runs");
  if (!existsSync(d)) mkdirSync(d, { recursive: true });
  return d;
}

function newId(): string {
  const ts = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
  const suffix = randomBytes(3).toString("hex");
  return `${ts}_${suffix}`;
}

function safeResolveUnder(root: string, rel: string): string {
  const abs = resolve(root, rel);
  const rootN = resolve(root) + sep;
  if (abs !== resolve(root) && !abs.startsWith(rootN)) {
    throw Object.assign(new Error("path escapes lab run root"), { code: "bad_path" });
  }
  return abs;
}

export function listLabRuns(limit = 50): LabRunSummary[] {
  const root = labRoot();
  const dirs = readdirSync(root, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name)
    .sort()
    .reverse();

  const out: LabRunSummary[] = [];
  for (const id of dirs) {
    if (out.length >= limit) break;
    const meta = readMeta(id);
    if (!meta) continue;
    out.push(toSummary(meta));
  }
  return out;
}

export function getLabRun(id: string): LabRunSummary | null {
  const meta = readMeta(id);
  return meta ? toSummary(meta) : null;
}

function readMeta(id: string): LabRunMeta | null {
  if (!/^[A-Za-z0-9._-]+$/.test(id)) return null;
  const path = join(labRoot(), id, "meta.json");
  if (!existsSync(path)) return null;
  try {
    return JSON.parse(readFileSync(path, "utf8")) as LabRunMeta;
  } catch {
    return null;
  }
}

function toSummary(meta: LabRunMeta): LabRunSummary {
  const dir = join(labRoot(), meta.id);
  const previewPath = join(dir, "preview.png");
  return {
    ...meta,
    dir,
    preview_url: existsSync(previewPath) ? `/api/lab/runs/${meta.id}/files/preview.png` : null,
    play_url: `/api/lab/runs/${meta.id}/play`,
  };
}

export type ImportLabInput = {
  title: string;
  task_type?: string;
  model_id?: string;
  entry?: string;
  from: string; // absolute or repo-relative file or directory
  tags?: string[];
  brief?: string;
  eval_run_id?: string;
  hermes?: { session?: string; source?: string };
  serve?: Record<string, unknown>;
  share_public?: boolean;
};

export function importLabRun(input: ImportLabInput): LabRunSummary {
  const fromAbs = resolve(config.root, input.from);
  if (!existsSync(fromAbs)) {
    throw Object.assign(new Error(`source not found: ${fromAbs}`), { code: "not_found" });
  }

  const id = newId();
  const runDir = join(labRoot(), id);
  const artDir = join(runDir, "artifacts");
  mkdirSync(artDir, { recursive: true });

  const st = statSync(fromAbs);
  let entry = input.entry || "";

  if (st.isFile()) {
    const name = basename(fromAbs);
    copyFileSync(fromAbs, join(artDir, name));
    entry = entry || name;
  } else {
    copyTree(fromAbs, artDir);
    if (!entry) {
      if (existsSync(join(artDir, "index.html"))) entry = "index.html";
      else {
        const html = readdirSync(artDir).find((f) => f.endsWith(".html"));
        entry = html || "index.html";
      }
    }
  }

  const brief = input.brief || "";
  if (brief) writeFileSync(join(runDir, "brief.md"), brief, "utf8");

  const task_type = input.task_type || guessTaskType(entry);
  const fingerprint = createHash("sha256")
    .update(`${task_type}\n${brief}\n${input.title}`)
    .digest("hex")
    .slice(0, 16);

  const meta: LabRunMeta = {
    id,
    kind: "hermes_task",
    task_type,
    title: input.title,
    model_id: input.model_id || "unknown",
    serve: input.serve || {},
    eval_run_id: input.eval_run_id || null,
    hermes: input.hermes || null,
    created_at: new Date().toISOString(),
    entry: entry.replace(/^\/+/, ""),
    share: { public: !!input.share_public, slug: null },
    tags: input.tags || [],
    brief: brief || undefined,
    task_fingerprint: fingerprint,
  };
  writeFileSync(join(runDir, "meta.json"), JSON.stringify(meta, null, 2), "utf8");
  return toSummary(meta);
}

function guessTaskType(entry: string): string {
  const e = entry.toLowerCase();
  if (e.endsWith(".html")) return "html-game";
  if (e.endsWith(".svg")) return "svg";
  return "artifact";
}

function copyTree(src: string, dst: string) {
  mkdirSync(dst, { recursive: true });
  for (const name of readdirSync(src)) {
    if (name === ".git" || name === "node_modules") continue;
    const s = join(src, name);
    const d = join(dst, name);
    const st = statSync(s);
    if (st.isDirectory()) copyTree(s, d);
    else if (st.isFile() && st.size < 20_000_000) copyFileSync(s, d);
  }
}

export function resolveLabFile(id: string, relPath: string): { abs: string; contentType: string } {
  const meta = readMeta(id);
  if (!meta) throw Object.assign(new Error("not found"), { code: "not_found" });
  const runDir = join(labRoot(), id);
  // Allow files under run root (artifacts/, preview.png, brief.md)
  const clean = relPath.replace(/^\/+/, "").replace(/\\/g, "/");
  if (clean.includes("..")) throw Object.assign(new Error("bad path"), { code: "bad_path" });
  const abs = safeResolveUnder(runDir, clean);
  if (!existsSync(abs) || !statSync(abs).isFile()) {
    throw Object.assign(new Error("file not found"), { code: "not_found" });
  }
  return { abs, contentType: mimeFor(abs) };
}

export function listLabFiles(id: string): string[] {
  const meta = readMeta(id);
  if (!meta) return [];
  const art = join(labRoot(), id, "artifacts");
  if (!existsSync(art)) return [];
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const name of readdirSync(dir)) {
      const p = join(dir, name);
      const st = statSync(p);
      if (st.isDirectory()) walk(p);
      else out.push(relative(art, p).split(sep).join("/"));
    }
  };
  walk(art);
  return out.sort();
}

function mimeFor(path: string): string {
  const e = extname(path).toLowerCase();
  const map: Record<string, string> = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".wasm": "application/wasm",
  };
  return map[e] || "application/octet-stream";
}

/** Seed gallery from demo HTML if empty */
export function ensureDemoLabRuns(): void {
  if (listLabRuns(1).length > 0) return;
  const demo = join(config.workspacesDir, "demo", "geometry-dash-like.html");
  if (!existsSync(demo)) return;
  try {
    importLabRun({
      title: "Geometry Dash–like runner",
      task_type: "html-game",
      model_id: "demo/seed",
      from: demo,
      tags: ["html", "game", "self-contained", "seed"],
      brief:
        "Self-contained HTML game (seed). Replace with Hermes-built runs via POST /api/lab/runs/import.",
      hermes: { source: "seed" },
    });
  } catch {
    /* ignore seed failures */
  }
}
