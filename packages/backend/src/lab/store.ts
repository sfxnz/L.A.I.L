import { createHash, randomBytes } from "crypto";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
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
  public_url: string | null;
  gallery_url: string;
};

function labRoot(): string {
  const d = join(config.dataDir, "lab-runs");
  if (!existsSync(d)) mkdirSync(d, { recursive: true });
  return d;
}

function publicRoot(): string {
  const d = join(config.dataDir, "lab-public");
  if (!existsSync(d)) mkdirSync(d, { recursive: true });
  return d;
}

function newId(): string {
  const ts = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
  const suffix = randomBytes(3).toString("hex");
  return `${ts}_${suffix}`;
}

function newSlug(): string {
  return randomBytes(6).toString("hex");
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

export function listLabRunsByFingerprint(fp: string, limit = 20): LabRunSummary[] {
  if (!fp) return [];
  return listLabRuns(200)
    .filter((r) => r.task_fingerprint === fp)
    .slice(0, limit);
}

export function compareLabRuns(ids: string[]): {
  runs: LabRunSummary[];
  task_fingerprint: string | null;
  same_brief: boolean;
  brief: string | null;
} {
  const runs = ids
    .map((id) => getLabRun(id))
    .filter((r): r is LabRunSummary => !!r);
  const fps = new Set(runs.map((r) => r.task_fingerprint).filter(Boolean));
  const same = fps.size <= 1;
  const brief = runs.find((r) => r.brief)?.brief || null;
  return {
    runs,
    task_fingerprint: fps.size === 1 ? ([...fps][0] as string) : null,
    same_brief: same,
    brief,
  };
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

function writeMeta(meta: LabRunMeta): void {
  writeFileSync(join(labRoot(), meta.id, "meta.json"), JSON.stringify(meta, null, 2), "utf8");
}

function toSummary(meta: LabRunMeta): LabRunSummary {
  const dir = join(labRoot(), meta.id);
  const previewPath = join(dir, "preview.png");
  const slug = meta.share?.slug || null;
  let public_url: string | null = null;
  if (meta.share?.public && slug) {
    const taskSeg = String(meta.task_type || "artifact")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "")
      .slice(0, 48) || "artifact";
    // Prefer Wesche-style static site (GitHub Pages) when configured
    if (config.shareSiteBase) {
      public_url = `${config.shareSiteBase}/dgx/${taskSeg}/${slug}/index.html`;
    } else if (config.sharePublicBase) {
      public_url = `${config.sharePublicBase}/s/${slug}/index.html`;
    } else {
      public_url = `/api/lab/p/${slug}/index.html`;
    }
  }
  return {
    ...meta,
    dir,
    preview_url: existsSync(previewPath) ? `/api/lab/runs/${meta.id}/files/preview.png` : null,
    play_url: `/api/lab/runs/${meta.id}/play`,
    gallery_url: `/lab/${meta.id}`,
    public_url,
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

  scanArtifactsForSecrets(artDir);

  const task_type = input.task_type || guessTaskType(entry);
  const fingerprint = createHash("sha256")
    .update(brief.trim() ? `${task_type}\n${brief.trim()}` : `${task_type}\n${input.title}`)
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
    share: { public: false, slug: null },
    tags: input.tags || [],
    brief: brief || undefined,
    task_fingerprint: fingerprint,
  };
  writeMeta(meta);

  let summary = toSummary(meta);
  if (input.share_public) {
    summary = publishLabRun(meta.id, true);
  }
  return summary;
}

/** Enable/disable public static share (artifacts only). */
export function publishLabRun(id: string, makePublic = true): LabRunSummary {
  const meta = readMeta(id);
  if (!meta) throw Object.assign(new Error("not found"), { code: "not_found" });

  const artDir = join(labRoot(), id, "artifacts");
  scanArtifactsForSecrets(artDir);

  if (makePublic) {
    const slug = meta.share?.slug || newSlug();
    const pubDir = join(publicRoot(), slug);
    // wipe + recopy
    rmrf(pubDir);
    mkdirSync(pubDir, { recursive: true });
    copyTree(artDir, pubDir);
    // index fallback if entry isn't index.html
    const entry = meta.entry || "index.html";
    if (entry !== "index.html" && existsSync(join(pubDir, entry))) {
      // Always materialize entry as index.html for clean /api/lab/p/<slug>/ URLs
      copyFileSync(join(pubDir, entry), join(pubDir, "index.html"));
    }
    writeFileSync(
      join(pubDir, "share.json"),
      JSON.stringify(
        {
          // Minimal metadata for ops — never served over play URL
          title: meta.title,
          model_id: meta.model_id,
          task_type: meta.task_type,
          created_at: meta.created_at,
        },
        null,
        2,
      ),
      "utf8",
    );
    meta.share = { public: true, slug };
  } else {
    if (meta.share?.slug) {
      rmrf(join(publicRoot(), meta.share.slug));
    }
    meta.share = { public: false, slug: null };
  }
  writeMeta(meta);
  return toSummary(meta);
}

export function getPublicBySlug(slug: string): {
  slug: string;
  meta: { title?: string; model_id?: string; task_type?: string; run_id?: string } | null;
  dir: string;
} | null {
  if (!/^[a-f0-9]{8,32}$/i.test(slug)) return null;
  const dir = join(publicRoot(), slug);
  if (!existsSync(dir)) return null;
  let meta = null;
  const sharePath = join(dir, "share.json");
  if (existsSync(sharePath)) {
    try {
      meta = JSON.parse(readFileSync(sharePath, "utf8"));
    } catch {
      meta = null;
    }
  }
  return { slug, meta, dir };
}

export function resolvePublicFile(slug: string, relPath: string): { abs: string; contentType: string } {
  const pub = getPublicBySlug(slug);
  if (!pub) throw Object.assign(new Error("not found"), { code: "not_found" });
  let clean = (relPath || "index.html").replace(/^\/+/, "").replace(/\\/g, "/");
  if (!clean || clean.endsWith("/")) clean = `${clean}index.html`;
  if (clean.includes("..") || clean.includes("\0")) {
    throw Object.assign(new Error("bad path"), { code: "bad_path" });
  }
  // Never expose internal metadata over the play URL
  const base = clean.split("/").pop() || clean;
  if (base === "share.json" || base === "meta.json" || base.startsWith(".")) {
    throw Object.assign(new Error("not found"), { code: "not_found" });
  }
  // Allowlist extensions for public play (no server configs)
  const ext = extname(clean).toLowerCase() || ".html";
  const allowed = new Set([
    ".html",
    ".htm",
    ".js",
    ".mjs",
    ".css",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".wasm",
    ".woff",
    ".woff2",
    ".ttf",
    ".mp3",
    ".ogg",
    ".wav",
    ".json", // game data only; share.json blocked above
    ".txt",
    ".map",
  ]);
  if (!allowed.has(ext)) {
    throw Object.assign(new Error("type not allowed"), { code: "forbidden_type" });
  }
  const abs = safeResolveUnder(pub.dir, clean);
  if (!existsSync(abs) || !statSync(abs).isFile()) {
    throw Object.assign(new Error("file not found"), { code: "not_found" });
  }
  return { abs, contentType: mimeFor(abs) };
}

/** Safe response headers for untrusted model-generated HTML/JS. */
export function publicPlayHeaders(contentType: string): Record<string, string> {
  return {
    "Content-Type": contentType,
    "Cache-Control": "public, max-age=120",
    "X-Robots-Tag": "noindex, nofollow",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    // Lock down what the game page can do. Model HTML is untrusted.
    "Content-Security-Policy": [
      "default-src 'none'",
      "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: blob:",
      "font-src 'self' data:",
      "media-src 'self' blob:",
      "connect-src 'self'",
      "form-action 'none'",
      "frame-ancestors 'self'",
      "base-uri 'none'",
      "object-src 'none'",
    ].join("; "),
    // Do not let other sites embed as a tracking pixel / data siphon easily
    "X-Frame-Options": "SAMEORIGIN",
  };
}

const SECRET_PATTERNS = [
  /api[_-]?key\s*[:=]\s*['"][^'"]+['"]/i,
  /sk-[a-zA-Z0-9]{20,}/,
  /BEGIN (RSA |OPENSSH )?PRIVATE KEY/,
  /BRIDGE_API_KEY\s*=/,
  /HF_TOKEN\s*=/,
  /password\s*[:=]\s*['"][^'"]{8,}['"]/i,
];

function scanArtifactsForSecrets(artDir: string): void {
  if (!existsSync(artDir)) return;
  const walk = (dir: string) => {
    for (const name of readdirSync(dir)) {
      const p = join(dir, name);
      const st = statSync(p);
      if (st.isDirectory()) walk(p);
      else if (st.isFile() && st.size < 2_000_000) {
        const ext = extname(name).toLowerCase();
        if (![".html", ".js", ".ts", ".json", ".txt", ".md", ".env", ".css"].includes(ext) && name !== ".env")
          continue;
        const text = readFileSync(p, "utf8");
        for (const re of SECRET_PATTERNS) {
          if (re.test(text)) {
            throw Object.assign(
              new Error(`refusing publish: possible secret in ${relative(artDir, p)}`),
              { code: "secret_detected" },
            );
          }
        }
      }
    }
  };
  walk(artDir);
}

function rmrf(path: string) {
  if (!existsSync(path)) return;
  rmSync(path, { recursive: true, force: true });
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
        "Self-contained HTML game (seed). Replace with Hermes-built runs via lail_lab_publish.",
      hermes: { source: "seed" },
    });
  } catch {
    /* ignore seed failures */
  }
}
