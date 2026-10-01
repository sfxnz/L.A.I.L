import type { BackendKind, LabSettings } from "@lail/shared";
import { getDb } from "../db/schema";
import { config } from "../config";

const KEY = "lab_settings";

function defaults(): LabSettings {
  return {
    defaultBackend: config.defaultBackend,
    defaultModel: config.defaultModel,
    backends: { ...config.backends },
  };
}

/** Keep the known engines (config.backends); strip legacy ones (ollama, lmstudio, custom). */
function sanitize(raw: Partial<LabSettings> & { backends?: Record<string, unknown> }): LabSettings {
  const base = defaults();
  const backends = { ...base.backends };
  const kinds = Object.keys(base.backends) as BackendKind[];
  if (raw.backends) {
    for (const k of kinds) {
      const b = raw.backends[k] as LabSettings["backends"][BackendKind] | undefined;
      if (b && typeof b === "object") {
        backends[k] = {
          url: b.url || backends[k].url,
          enabled: b.enabled !== false,
          label: b.label || backends[k].label,
        };
      }
    }
  }
  let defaultBackend: BackendKind = kinds.includes(raw.defaultBackend as BackendKind)
    ? (raw.defaultBackend as BackendKind)
    : base.defaultBackend;
  // Migrate old default ollama → vllm
  if ((raw.defaultBackend as string) === "ollama" || (raw.defaultBackend as string) === "lmstudio") {
    defaultBackend = "vllm";
  }
  return {
    defaultBackend,
    defaultModel: raw.defaultModel || base.defaultModel,
    backends,
  };
}

export function getSettings(): LabSettings {
  const row = getDb().query("SELECT value FROM settings WHERE key = ?").get(KEY) as
    | { value: string }
    | null;
  if (!row) return defaults();
  try {
    return sanitize(JSON.parse(row.value));
  } catch {
    return defaults();
  }
}

export function putSettings(patch: Partial<LabSettings>): LabSettings {
  const cur = getSettings();
  const next = sanitize({
    ...cur,
    ...patch,
    backends: { ...cur.backends, ...(patch.backends || {}) },
  });
  getDb()
    .query(
      "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
    )
    .run(KEY, JSON.stringify(next));
  return next;
}

export function backendBaseUrl(kind?: BackendKind): string {
  const s = getSettings();
  const k = kind || s.defaultBackend;
  return s.backends[k]?.url || config.backends.vllm.url;
}

export function openAiBase(kind?: BackendKind): string {
  const base = backendBaseUrl(kind).replace(/\/$/, "");
  // Every engine (vLLM, SGLang, llama.cpp, TensorFold) serves the OpenAI API under /v1
  if (base.endsWith("/v1")) return base;
  return `${base}/v1`;
}

/** Placeholders that mean "use whatever the backend is serving". */
export function isPlaceholderModel(model: string | undefined | null): boolean {
  const m = (model || "").trim().toLowerCase();
  return !m || m === "default" || m === "auto" || m === "none";
}

export async function listServedModelIds(kind?: BackendKind): Promise<string[]> {
  const base = openAiBase(kind);
  try {
    const r = await fetch(`${base}/models`, { signal: AbortSignal.timeout(5000) });
    if (!r.ok) return [];
    const j = (await r.json()) as { data?: Array<{ id: string }> };
    return (j.data || []).map((m) => m.id).filter(Boolean);
  } catch {
    return [];
  }
}

/**
 * Model id for a request that named a placeholder: the live served id when the
 * backend serves something, else the configured default (if it is a real id).
 * Pass `served` to reuse an already-fetched id list instead of probing again.
 */
export async function resolveModelId(kind?: BackendKind, served?: string[]): Promise<string> {
  served ??= await listServedModelIds(kind);
  if (served[0]) return served[0];

  const configured = getSettings().defaultModel?.trim() || "";
  if (configured && !isPlaceholderModel(configured)) return configured;

  throw new Error(`Nothing is served at ${openAiBase(kind)}/models. Start a model on Serve first.`);
}
