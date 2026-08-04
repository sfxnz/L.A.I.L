#!/usr/bin/env bun
/**
 * Internet-facing static share server — ONLY data/lab-public artifacts.
 * Never mounts L.A.I.L admin, Hermes, or serve-engine.
 *
 * Bind: 127.0.0.1:8791 (loopback). Expose with:
 *   bun run scripts/lab-share-funnel.sh
 */
import { existsSync, readFileSync, statSync } from "fs";
import { join, resolve } from "path";

const root = resolve(process.env.LAIL_ROOT || join(import.meta.dir, ".."));
const pubRoot = resolve(process.env.LAIL_LAB_PUBLIC_DIR || join(root, "data/lab-public"));
const host = process.env.LAIL_SHARE_HOST || "127.0.0.1";
const port = Number(process.env.LAIL_SHARE_PORT || 8791);

const ALLOWED_EXT = new Set([
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
  ".json",
  ".txt",
  ".map",
]);

function mime(path: string): string {
  const e = path.includes(".") ? path.slice(path.lastIndexOf(".")).toLowerCase() : "";
  const map: Record<string, string> = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".json": "application/json",
    ".txt": "text/plain; charset=utf-8",
    ".wasm": "application/wasm",
  };
  return map[e] || "application/octet-stream";
}

function headers(ct: string): HeadersInit {
  return {
    "Content-Type": ct,
    "Cache-Control": "public, max-age=120",
    "X-Robots-Tag": "noindex, nofollow",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": [
      "default-src 'none'",
      "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: blob:",
      "font-src 'self' data:",
      "media-src 'self' blob:",
      "connect-src 'self'",
      "form-action 'none'",
      "frame-ancestors *",
      "base-uri 'none'",
      "object-src 'none'",
    ].join("; "),
  };
}

function safeJoin(slug: string, rel: string): string | null {
  if (!/^[a-f0-9]{8,32}$/i.test(slug)) return null;
  let clean = (rel || "index.html").replace(/^\/+/, "").replace(/\\/g, "/");
  if (!clean || clean.endsWith("/")) clean += "index.html";
  if (clean.includes("..") || clean.includes("\0")) return null;
  const base = clean.split("/").pop() || "";
  if (base === "share.json" || base === "meta.json" || base.startsWith(".")) return null;
  const ext = base.includes(".") ? base.slice(base.lastIndexOf(".")).toLowerCase() : ".html";
  if (!ALLOWED_EXT.has(ext)) return null;
  const abs = resolve(pubRoot, slug, clean);
  const rootN = resolve(pubRoot, slug) + "/";
  if (abs !== resolve(pubRoot, slug) && !abs.startsWith(rootN)) return null;
  if (!existsSync(abs) || !statSync(abs).isFile()) return null;
  return abs;
}

function parsePath(pathname: string): { slug: string; rel: string } | null {
  // /s/<slug>/...  or  /p/<slug>/...  or  /api/lab/p/<slug>/...
  const patterns = [/^\/s\/([^/]+)\/?(.*)$/, /^\/p\/([^/]+)\/?(.*)$/, /^\/api\/lab\/p\/([^/]+)\/?(.*)$/];
  for (const re of patterns) {
    const m = pathname.match(re);
    if (m) return { slug: m[1], rel: m[2] || "index.html" };
  }
  return null;
}

const server = Bun.serve({
  hostname: host,
  port,
  fetch(req) {
    const url = new URL(req.url);
    if (url.pathname === "/" || url.pathname === "/health") {
      return Response.json({
        ok: true,
        service: "lail-lab-public-share",
        note: "artifacts only — not L.A.I.L admin",
      });
    }
    const parsed = parsePath(url.pathname);
    if (!parsed) return new Response("Not found", { status: 404 });
    const abs = safeJoin(parsed.slug, parsed.rel || "index.html");
    if (!abs) return new Response("Not found", { status: 404 });
    const body = readFileSync(abs);
    return new Response(body, { headers: headers(mime(abs)) });
  },
});

console.log(
  `lab-public-share listening http://${host}:${port} root=${pubRoot} (loopback-only recommended)`,
);
console.log(`  health: http://${host}:${port}/health`);
console.log(`  play:   http://${host}:${port}/s/<slug>/index.html`);
