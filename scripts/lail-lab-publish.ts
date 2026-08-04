#!/usr/bin/env bun
/**
 * lail-lab-publish — one-shot Hermes/agent helper
 *
 * Publish a built HTML game / artifact into L.A.I.L Lab gallery.
 *
 * Usage:
 *   bun run scripts/lail-lab-publish.ts \
 *     --title "Runner v1" \
 *     --from ./out/index.html \
 *     --model nvidia/Qwen3.6-27B-NVFP4 \
 *     --brief "Self-contained HTML runner, no CDN" \
 *     --public
 *
 * Env:
 *   LAIL_API=http://127.0.0.1:8787
 *   LAIL_PUBLIC_BASE=http://100.86.121.44:3000   # for share URLs in output
 */
import { resolve } from "path";

function arg(name: string, fallback = ""): string {
  const i = process.argv.indexOf(`--${name}`);
  if (i >= 0 && process.argv[i + 1] && !process.argv[i + 1].startsWith("--")) {
    return process.argv[i + 1];
  }
  return fallback;
}

function has(name: string): boolean {
  return process.argv.includes(`--${name}`);
}

async function main() {
  const title = arg("title");
  const from = arg("from");
  if (!title || !from) {
    console.error(
      "Usage: lail-lab-publish --title TITLE --from PATH [--model ID] [--type html-game] [--brief TEXT] [--tags a,b] [--public] [--entry index.html]",
    );
    process.exit(2);
  }

  const api = (process.env.LAIL_API || "http://127.0.0.1:8787").replace(/\/$/, "");
  const publicBase = (process.env.LAIL_PUBLIC_BASE || "http://100.86.121.44:3000").replace(
    /\/$/,
    "",
  );

  const body = {
    title,
    from: resolve(from),
    model_id: arg("model", process.env.OPENAI_MODEL || "unknown"),
    task_type: arg("type", "html-game"),
    brief: arg("brief", ""),
    entry: arg("entry", "") || undefined,
    tags: arg("tags", "html,hermes")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean),
    hermes: { source: "lail_lab_publish" },
    share_public: has("public"),
  };

  const r = await fetch(`${api}/api/lab/runs/import`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await r.text();
  let data: Record<string, unknown>;
  try {
    data = JSON.parse(text);
  } catch {
    console.error(text);
    process.exit(1);
  }
  if (!r.ok) {
    console.error(JSON.stringify(data, null, 2));
    process.exit(1);
  }

  const id = String(data.id || "");
  const gallery = `${publicBase}/lab/${id}`;
  const play = `${publicBase}${data.play_url || ""}`;
  let publicUrl: string | null = data.public_url
    ? String(data.public_url).startsWith("http")
      ? String(data.public_url)
      : `${publicBase}${data.public_url}`
    : null;

  if (has("public") && id && !publicUrl) {
    const s = await fetch(`${api}/api/lab/runs/${id}/share`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ public: true }),
    });
    const sj = (await s.json()) as Record<string, unknown>;
    if (sj.public_url) {
      const pu = String(sj.public_url);
      publicUrl = pu.startsWith("http") ? pu : `${publicBase}${pu}`;
    }
    data = { ...data, ...sj };
  }

  const out = {
    ok: true,
    id,
    title: data.title,
    model_id: data.model_id,
    task_fingerprint: data.task_fingerprint,
    gallery_url: gallery,
    play_url: play,
    public_url: publicUrl,
    compare_hint: data.task_fingerprint
      ? `${publicBase}/lab/compare?fingerprint=${data.task_fingerprint}`
      : null,
  };
  console.log(JSON.stringify(out, null, 2));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
