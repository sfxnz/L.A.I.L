import type { ModelCard } from "@lail/shared";
import { getSettings, openAiBase } from "./settings";
import { wsHub } from "../ws/hub";
import { randomUUID } from "crypto";
import { mkdirSync } from "fs";
import { join } from "path";
import { config } from "../config";

export function hfSearchQuery(raw: string | undefined | null): string {
  const q = String(raw ?? "").trim();
  return q || "safetensors";
}

export async function searchHuggingFace(q: string, limit = 24): Promise<ModelCard[]> {
  const url = new URL("https://huggingface.co/api/models");
  url.searchParams.set("search", hfSearchQuery(q));
  url.searchParams.set("limit", String(limit));
  url.searchParams.set("full", "false");
  url.searchParams.set("config", "false");
  const headers: Record<string, string> = { Accept: "application/json" };
  const token = getSettings().hfToken;
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(url.toString(), { headers });
  if (!res.ok) throw new Error(`HF search failed: ${res.status}`);
  const data = (await res.json()) as Array<Record<string, unknown>>;
  return data.map((m) => {
    const id = String(m.id || m.modelId || "");
    const tags = (m.tags as string[]) || [];
    const quants = tags.filter((t) => /gguf|gptq|awq|fp8|nf4|q[2-8]_|bnb|nvfp4/i.test(t));
    return {
      id,
      name: id.split("/").pop() || id,
      author: id.includes("/") ? id.split("/")[0] : undefined,
      downloads: Number(m.downloads || 0),
      likes: Number(m.likes || 0),
      tags,
      pipeline_tag: m.pipeline_tag as string | undefined,
      library_name: m.library_name as string | undefined,
      license: (m.cardData as { license?: string } | undefined)?.license,
      quantizations: quants.slice(0, 8),
      hardwareFit: estimateFit(tags, Number(m.downloads || 0)),
      local: false,
      backends: ["vllm", "llamacpp"],
    } satisfies ModelCard;
  });
}

function estimateFit(tags: string[], downloads: number): ModelCard["hardwareFit"] {
  const blob = tags.join(" ").toLowerCase();
  if (/1b|2b|3b|mini|tiny|phi-3|gemma-2b/.test(blob)) return "excellent";
  if (/7b|8b|9b|gguf|q4|q5/.test(blob)) return "good";
  if (/70b|72b|405b|671b/.test(blob)) return "tight";
  if (downloads > 1_000_000) return "good";
  return "unknown";
}

export async function listLocalModels(): Promise<ModelCard[]> {
  const settings = getSettings();
  const out: ModelCard[] = [];
  const seen = new Set<string>();

  for (const kind of ["vllm", "llamacpp"] as const) {
    const be = settings.backends[kind];
    if (!be?.enabled) continue;
    try {
      const base = be.url.replace(/\/$/, "").replace(/\/v1$/, "");
      const r = await fetch(`${base}/v1/models`, { signal: AbortSignal.timeout(3000) });
      if (!r.ok) continue;
      const j = (await r.json()) as { data?: Array<{ id: string }> };
      for (const m of j.data || []) {
        if (seen.has(m.id)) {
          const existing = out.find((x) => x.id === m.id);
          if (existing && !existing.backends?.includes(kind)) {
            existing.backends = [...(existing.backends || []), kind];
          }
          continue;
        }
        seen.add(m.id);
        out.push({
          id: m.id,
          name: m.id.split("/").pop() || m.id,
          local: true,
          backends: [kind],
          tags: ["local", kind],
          hardwareFit: "good",
        });
      }
    } catch {
      /* offline */
    }
  }

  // Also probe default OpenAI base once
  try {
    const base = openAiBase();
    const r = await fetch(`${base}/models`, { signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      const j = (await r.json()) as { data?: Array<{ id: string }> };
      for (const m of j.data || []) {
        if (seen.has(m.id)) continue;
        seen.add(m.id);
        out.push({
          id: m.id,
          name: m.id.split("/").pop() || m.id,
          local: true,
          backends: [settings.defaultBackend],
          tags: ["local"],
          hardwareFit: "good",
        });
      }
    }
  } catch {
    /* offline */
  }

  return out;
}

const jobs = new Map<string, { status: string; progress: number; message: string }>();

export function getPullJob(id: string) {
  return jobs.get(id);
}

/**
 * Download weights via Hugging Face CLI into data/models.
 * Use with vLLM (HF id / path) or llama.cpp (GGUF path after download).
 */
export async function pullModel(
  model: string,
  backend: "hf" | "vllm" | "llamacpp" = "hf",
): Promise<{ jobId: string }> {
  const jobId = randomUUID();
  jobs.set(jobId, { status: "running", progress: 0.05, message: "starting HF download" });
  wsHub.publish(`download:${jobId}`, {
    type: "download_progress",
    jobId,
    progress: 0.05,
    message: "starting HF download",
  });

  (async () => {
    try {
      const dest = join(config.dataDir, "models", model.replace(/\//g, "__"));
      mkdirSync(dest, { recursive: true });
      jobs.set(jobId, { status: "running", progress: 0.15, message: `downloading ${model}` });
      wsHub.publish(`download:${jobId}`, {
        type: "download_progress",
        jobId,
        progress: 0.15,
        message: `downloading ${model} → ${dest}`,
      });

      // Prefer huggingface-cli, then hf
      const cmds = [
        ["huggingface-cli", "download", model, "--local-dir", dest],
        ["hf", "download", model, "--local-dir", dest],
      ];
      let lastErr = "no HF CLI found";
      let ok = false;
      for (const cmd of cmds) {
        try {
          const proc = Bun.spawn(cmd, {
            cwd: config.root,
            stdout: "pipe",
            stderr: "pipe",
            env: {
              ...process.env,
              ...(getSettings().hfToken ? { HF_TOKEN: getSettings().hfToken } : {}),
            },
          });
          const stderr = await new Response(proc.stderr).text();
          const stdout = await new Response(proc.stdout).text();
          await proc.exited;
          if (proc.exitCode === 0) {
            ok = true;
            jobs.set(jobId, { status: "running", progress: 0.9, message: "finalizing" });
            wsHub.publish(`download:${jobId}`, {
              type: "download_progress",
              jobId,
              progress: 0.9,
              message: (stdout || stderr || "done").slice(0, 200),
            });
            break;
          }
          lastErr = stderr || stdout || `exit ${proc.exitCode}`;
        } catch (e) {
          lastErr = e instanceof Error ? e.message : String(e);
        }
      }
      if (!ok) {
        throw new Error(
          `HF download failed (${lastErr}). Install: pip install -U "huggingface_hub[cli]". ` +
            `Then serve with vLLM (Server tab) or point llama.cpp at the GGUF under ${dest}. Backend hint: ${backend}`,
        );
      }
      jobs.set(jobId, { status: "done", progress: 1, message: "done" });
      wsHub.publish(`download:${jobId}`, { type: "download_done", jobId, model });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      jobs.set(jobId, { status: "error", progress: 0, message });
      wsHub.publish(`download:${jobId}`, { type: "download_error", jobId, message });
    }
  })();

  return { jobId };
}
