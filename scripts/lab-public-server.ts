#!/usr/bin/env bun
/**
 * Internet-facing static share server — ONLY data/lab-public artifacts.
 * Never mounts L.A.I.L admin, Hermes, or serve-engine.
 *
 * MUST bind 127.0.0.1 only. Expose with: bun run lab:funnel
 * (Tailscale Funnel → this process only, not :3000/:8787)
 */
import { existsSync, readFileSync, statSync } from "fs";
import { join, resolve } from "path";

const root = resolve(process.env.LAIL_ROOT || join(import.meta.dir, ".."));
const pubRoot = resolve(process.env.LAIL_LAB_PUBLIC_DIR || join(root, "data/lab-public"));
// Hard rule: never listen on all interfaces — Funnel reaches us via loopback.
const host = "127.0.0.1";
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
  // game data only — share.json / meta.json blocked by name below
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
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
    "Cross-Origin-Resource-Policy": "cross-origin",
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
      "upgrade-insecure-requests",
    ].join("; "),
  };
}

function safeJoin(slug: string, rel: string): string | null {
  if (!/^[a-f0-9]{8,32}$/i.test(slug)) return null;
  let clean = (rel || "index.html").replace(/^\/+/, "").replace(/\\/g, "/");
  if (!clean || clean.endsWith("/")) clean += "index.html";
  if (clean.includes("..") || clean.includes("\0") || clean.includes("%")) return null;
  const base = clean.split("/").pop() || "";
  if (
    base === "share.json" ||
    base === "meta.json" ||
    base.startsWith(".") ||
    base.toLowerCase().includes("env")
  ) {
    return null;
  }
  const ext = base.includes(".") ? base.slice(base.lastIndexOf(".")).toLowerCase() : ".html";
  if (!ALLOWED_EXT.has(ext)) return null;
  const slugRoot = resolve(pubRoot, slug);
  const abs = resolve(slugRoot, clean);
  const rootN = slugRoot.endsWith("/") ? slugRoot : slugRoot + "/";
  if (abs !== slugRoot && !abs.startsWith(rootN)) return null;
  if (!existsSync(abs) || !statSync(abs).isFile()) return null;
  // cap file size (20MB)
  if (statSync(abs).size > 20_000_000) return null;
  return abs;
}

function parsePath(pathname: string): { slug: string; rel: string } | null {
  const patterns = [
    /^\/s\/([^/]+)\/?(.*)$/,
    /^\/p\/([^/]+)\/?(.*)$/,
    /^\/api\/lab\/p\/([^/]+)\/?(.*)$/,
  ];
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
    if (req.method !== "GET" && req.method !== "HEAD") {
      return new Response("Method not allowed", { status: 405 });
    }
    const url = new URL(req.url);
    if (url.pathname === "/health") {
      return Response.json({ ok: true, service: "lail-lab-public-share" });
    }
    if (url.pathname === "/") {
      return new Response("L.A.I.L public artifact share", {
        status: 200,
        headers: { "Content-Type": "text/plain; charset=utf-8" },
      });
    }
    const parsed = parsePath(url.pathname);
    if (!parsed) return new Response("Not found", { status: 404 });
    const abs = safeJoin(parsed.slug, parsed.rel || "index.html");
    if (!abs) return new Response("Not found", { status: 404 });
    const body = readFileSync(abs);
    return new Response(req.method === "HEAD" ? null : body, {
      headers: headers(mime(abs)),
    });
  },
});

console.log(
  `lab-public-share listening http://${host}:${port} root=${pubRoot} (loopback ONLY)`,
);
console.log(`  play: http://${host}:${port}/s/<slug>/index.html`);
// keep process alive reference
void server;
