#!/usr/bin/env bun
/**
 * Build a Wesche-style static site from L.A.I.L lab-public / lab-runs.
 *
 * Output: site/dist/  (GitHub Pages ready)
 *   index.html
 *   dgx/<task>/<slug>/index.html
 *   catalog.json
 *
 * Usage:
 *   bun run scripts/lab-site-build.ts
 *   bun run scripts/lab-site-build.ts --slug 8b811f17ebef
 */
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
import { basename, join, relative, resolve } from "path";

const root = resolve(import.meta.dir, "..");
const dataDir = resolve(process.env.LAIL_DATA_DIR || join(root, "data"));
const labRuns = join(dataDir, "lab-runs");
const labPublic = join(dataDir, "lab-public");
const outDir = resolve(process.env.LAIL_SITE_OUT || join(root, "site/dist"));
const siteBase = (process.env.LAIL_SITE_BASE || "").replace(/\/$/, "");

const onlySlug = (() => {
  const i = process.argv.indexOf("--slug");
  return i >= 0 ? process.argv[i + 1] : "";
})();

type Entry = {
  slug: string;
  title: string;
  model_id: string;
  task_type: string;
  created_at: string;
  brief?: string;
  tags: string[];
  path: string; // site-relative path to index.html
  url: string; // absolute if SITE_BASE else path
};

function rmrf(p: string) {
  if (existsSync(p)) rmSync(p, { recursive: true, force: true });
}

function copyTree(src: string, dst: string) {
  mkdirSync(dst, { recursive: true });
  for (const name of readdirSync(src)) {
    if (name === "share.json" || name === "meta.json" || name.startsWith(".")) continue;
    const s = join(src, name);
    const d = join(dst, name);
    const st = statSync(s);
    if (st.isDirectory()) copyTree(s, d);
    else if (st.isFile() && st.size < 20_000_000) copyFileSync(s, d);
  }
}

function slugify(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 48) || "item";
}

function loadMetaForSlug(slug: string): Record<string, unknown> | null {
  // Prefer lab-runs meta that points at this public slug
  if (!existsSync(labRuns)) return null;
  for (const id of readdirSync(labRuns)) {
    const metaPath = join(labRuns, id, "meta.json");
    if (!existsSync(metaPath)) continue;
    try {
      const m = JSON.parse(readFileSync(metaPath, "utf8"));
      if (m?.share?.slug === slug) return m;
    } catch {
      /* */
    }
  }
  const sharePath = join(labPublic, slug, "share.json");
  if (existsSync(sharePath)) {
    try {
      return JSON.parse(readFileSync(sharePath, "utf8"));
    } catch {
      /* */
    }
  }
  return null;
}

function build(): Entry[] {
  if (!existsSync(labPublic)) {
    console.error("No data/lab-public yet — publish a run first");
    process.exit(1);
  }

  // Fresh site (or keep and merge if --slug only)
  if (!onlySlug) {
    rmrf(outDir);
  }
  mkdirSync(outDir, { recursive: true });
  mkdirSync(join(outDir, "dgx"), { recursive: true });

  const slugs = readdirSync(labPublic).filter((s) => {
    if (onlySlug) return s === onlySlug;
    return /^[a-f0-9]{8,32}$/i.test(s) && statSync(join(labPublic, s)).isDirectory();
  });

  const entries: Entry[] = [];

  for (const slug of slugs) {
    const src = join(labPublic, slug);
    if (!existsSync(join(src, "index.html"))) {
      // try any html
      const html = readdirSync(src).find((f) => f.endsWith(".html"));
      if (html) copyFileSync(join(src, html), join(src, "index.html"));
      else continue;
    }

    const meta = loadMetaForSlug(slug) || {};
    const title = String(meta.title || `Lab ${slug.slice(0, 8)}`);
    const task = String(meta.task_type || "artifact");
    const model = String(meta.model_id || "unknown");
    const created = String(meta.created_at || new Date().toISOString());
    const tags = Array.isArray(meta.tags) ? meta.tags.map(String) : [];
    const brief = meta.brief ? String(meta.brief) : undefined;

    const folder = join("dgx", slugify(task), slug);
    const dest = join(outDir, folder);
    rmrf(dest);
    copyTree(src, dest);

    // Ensure index.html
    if (!existsSync(join(dest, "index.html"))) {
      const html = readdirSync(dest).find((f) => f.endsWith(".html"));
      if (html) copyFileSync(join(dest, html), join(dest, "index.html"));
    }

    const path = `${folder}/index.html`.replace(/\\/g, "/");
    const url = siteBase ? `${siteBase}/${path}` : `/${path}`;
    entries.push({
      slug,
      title,
      model_id: model,
      task_type: task,
      created_at: created,
      brief,
      tags,
      path,
      url,
    });
  }

  // Merge with existing catalog if partial build
  let all = entries;
  const catalogPath = join(outDir, "catalog.json");
  if (onlySlug && existsSync(catalogPath)) {
    try {
      const prev = JSON.parse(readFileSync(catalogPath, "utf8")) as Entry[];
      const map = new Map(prev.map((e) => [e.slug, e]));
      for (const e of entries) map.set(e.slug, e);
      all = [...map.values()];
    } catch {
      /* */
    }
  }

  all.sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
  writeFileSync(catalogPath, JSON.stringify(all, null, 2), "utf8");
  writeFileSync(join(outDir, "index.html"), renderIndex(all), "utf8");
  writeFileSync(
    join(outDir, "CNAME.example"),
    "# Rename to CNAME and put your domain, e.g.\n# lab.sfxnz.com\n",
    "utf8",
  );
  writeFileSync(
    join(outDir, ".nojekyll"),
    "",
    "utf8",
  );

  console.log(JSON.stringify({ ok: true, outDir, count: all.length, siteBase: siteBase || null, entries: all }, null, 2));
  return all;
}

function esc(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderIndex(entries: Entry[]): string {
  const cards = entries
    .map((e) => {
      const model = esc((e.model_id || "").split("/").pop() || e.model_id);
      return `
      <a class="card" href="${esc(e.path)}">
        <div class="frame"><iframe src="${esc(e.path)}" loading="lazy" sandbox="allow-scripts" tabindex="-1"></iframe></div>
        <div class="meta">
          <h2>${esc(e.title)}</h2>
          <p class="sub">${model} · ${esc(e.task_type)}</p>
          <p class="date">${esc(e.created_at.slice(0, 10))}</p>
        </div>
      </a>`;
    })
    .join("\n");

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DGX Lab — local model builds</title>
  <meta name="description" content="Self-contained HTML games and visuals built with local models on DGX Spark." />
  <meta name="twitter:card" content="summary_large_image" />
  <style>
    :root { color-scheme: dark; --bg:#0a0a0b; --card:#141416; --border:#2a2a2e; --text:#f5f5f7; --muted:#8e8e93; --accent:#0a84ff; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background:var(--bg); color:var(--text); }
    header { max-width:1100px; margin:0 auto; padding:2.5rem 1.25rem 1rem; }
    header h1 { font-size:1.5rem; letter-spacing:-0.03em; margin:0 0 .35rem; }
    header p { color:var(--muted); margin:0; max-width:40rem; line-height:1.45; font-size:.95rem; }
    main { max-width:1100px; margin:0 auto; padding:1rem 1.25rem 3rem; display:grid; gap:1rem; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); }
    .card { display:flex; flex-direction:column; border:1px solid var(--border); border-radius:16px; overflow:hidden; background:var(--card); text-decoration:none; color:inherit; transition:border-color .15s; }
    .card:hover { border-color: color-mix(in srgb, var(--accent) 50%, var(--border)); }
    .frame { aspect-ratio:16/10; background:#000; overflow:hidden; pointer-events:none; }
    .frame iframe { width:200%; height:200%; border:0; transform:scale(.5); transform-origin:top left; }
    .meta { padding:.9rem 1rem 1.1rem; }
    .meta h2 { font-size:1rem; margin:0 0 .25rem; letter-spacing:-0.02em; }
    .sub,.date { margin:0; color:var(--muted); font-size:.8rem; }
    footer { max-width:1100px; margin:0 auto; padding:0 1.25rem 2.5rem; color:var(--muted); font-size:.75rem; }
    footer code { color:var(--text); }
  </style>
</head>
<body>
  <header>
    <h1>DGX Lab</h1>
    <p>Self-contained games &amp; visuals built with local models on NVIDIA DGX Spark — published as static pages (Spark stays private).</p>
  </header>
  <main>
    ${cards || "<p style=\"color:var(--muted)\">No published builds yet.</p>"}
  </main>
  <footer>
    Generated by L.A.I.L Lab · ${entries.length} build${entries.length === 1 ? "" : "s"}
    ${siteBase ? ` · <code>${esc(siteBase)}</code>` : ""}
  </footer>
</body>
</html>
`;
}

build();
