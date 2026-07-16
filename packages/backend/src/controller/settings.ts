import type { BackendKind, LabSettings } from "@lail/shared";
import { getDb } from "../db/schema";
import { config } from "../config";

const KEY = "lab_settings";

function defaults(): LabSettings {
  return {
    defaultBackend: config.defaultBackend,
    defaultModel: config.defaultModel,
    backends: { ...config.backends },
    hfToken: config.hfToken || undefined,
    contextBudgetChars: 32_000,
    contextMaxFileChars: 200_000,
    contextMaxSearchHits: 30,
  };
}

function clampBudget(n: number): number {
  return Math.min(500_000, Math.max(2_000, n));
}

function coercePositiveInt(value: unknown, fallback: number): number {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return Math.floor(n);
}

/** Strip legacy backends (ollama, lmstudio, custom) from stored settings. */
function sanitize(raw: Partial<LabSettings> & { backends?: Record<string, unknown> }): LabSettings {
  const base = defaults();
  const backends = { ...base.backends };
  if (raw.backends) {
    for (const k of ["vllm", "llamacpp"] as BackendKind[]) {
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
  let defaultBackend: BackendKind =
    raw.defaultBackend === "llamacpp" || raw.defaultBackend === "vllm"
      ? raw.defaultBackend
      : base.defaultBackend;
  // Migrate old default ollama → vllm
  if ((raw.defaultBackend as string) === "ollama" || (raw.defaultBackend as string) === "lmstudio") {
    defaultBackend = "vllm";
  }
  return {
    defaultBackend,
    defaultModel: raw.defaultModel || base.defaultModel,
    backends,
    hfToken: raw.hfToken ?? base.hfToken,
    contextBudgetChars: clampBudget(
      coercePositiveInt(raw.contextBudgetChars, base.contextBudgetChars!),
    ),
    contextMaxFileChars: coercePositiveInt(
      raw.contextMaxFileChars,
      base.contextMaxFileChars!,
    ),
    contextMaxSearchHits: coercePositiveInt(
      raw.contextMaxSearchHits,
      base.contextMaxSearchHits!,
    ),
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
  // vLLM and llama.cpp server both expose OpenAI API under /v1
  if (base.endsWith("/v1")) return base;
  return `${base}/v1`;
}

/** Placeholders that mean "use whatever the backend is serving". */
function isPlaceholderModel(model: string | undefined | null): boolean {
  const m = (model || "").trim().toLowerCase();
  // mock-model is used only in unit tests — never treat as a real served id
  return !m || m === "default" || m === "auto" || m === "none" || m === "mock-model";
}

/**
 * Resolve the model id to send to the OpenAI-compatible backend.
 * If Configure has a real model name, use it; otherwise probe /v1/models
 * and pick the first served id (fixes vLLM 404 on `default`).
 */
export async function resolveModelId(kind?: BackendKind): Promise<string> {
  const settings = getSettings();
  const configured = settings.defaultModel?.trim();
  if (!isPlaceholderModel(configured)) {
    return configured!;
  }

  const base = openAiBase(kind);
  try {
    const r = await fetch(`${base}/models`, { signal: AbortSignal.timeout(5000) });
    if (r.ok) {
      const j = (await r.json()) as { data?: Array<{ id: string }> };
      const id = j.data?.[0]?.id;
      if (id) {
        // Persist so Configure UI shows the real model next time
        if (isPlaceholderModel(settings.defaultModel)) {
          putSettings({ defaultModel: id });
        }
        return id;
      }
    }
  } catch {
    /* fall through */
  }
  throw new Error(
    `No model configured and could not auto-detect from ${base}/models. ` +
      `Set Default model in Configure to the served id (e.g. unsloth/Qwen3.6-35B-A3B-NVFP4).`,
  );
}
