/**
 * Bench configuration domain: concurrency levels (1–32, multi-select, ascending),
 * the six presets on keys 1–6, tokens/stream, the interpretation knobs, prefill
 * sizes, and the query-string form Status → Bench uses to hand a config over.
 */

export const CONCURRENCY_LEVELS: readonly number[] = Array.from({ length: 32 }, (_, i) => i + 1);

/** Keys `1 2 3 4 5 6` toggle ×1 ×2 ×4 ×8 ×16 ×32. */
export const LEVEL_PRESETS: readonly number[] = [1, 2, 4, 8, 16, 32];

export function presetForKey(key: string): number | null {
  const k = Number(key);
  if (!Number.isInteger(k) || k < 1 || k > LEVEL_PRESETS.length) return null;
  return LEVEL_PRESETS[k - 1];
}

export function sortConcurrencies(selected: Iterable<number>): number[] {
  return [...new Set([...selected].filter((n) => Number.isInteger(n) && n >= 1 && n <= 32))].sort(
    (a, b) => a - b,
  );
}

/** Toggle one level; the selection never becomes empty. */
export function toggleLevel(selected: ReadonlySet<number>, n: number): Set<number> {
  const next = new Set(selected);
  if (next.has(n)) {
    if (next.size > 1) next.delete(n);
  } else if (n >= 1 && n <= 32) {
    next.add(n);
  }
  return next;
}

export const TOKENS_PER_STREAM: readonly number[] = [256, 512, 1024];
export const DEFAULT_TOKENS = 512;
export const DEFAULT_LEVELS: readonly number[] = [1, 2, 4, 8];
/** tok/s per stream below which a level no longer feels interactive. */
export const DEFAULT_FLOOR = 20;
export const DEFAULT_SLO_MS = 500;
export const DEFAULT_PACK = "prose";

/** The old four decode families plus the two load-only packs, for rows whose pack list hasn't loaded. */
export const PACK_LABELS: Record<string, string> = {
  prose: "Prose",
  structured: "Structured",
  code: "Code",
  json: "JSON",
  "chat-short": "Chat (short)",
  mixed: "Mixed",
};

export function packLabel(id: string | null | undefined, packs?: ReadonlyArray<{ id: string; label: string }>): string {
  if (!id) return "Unknown";
  return packs?.find((p) => p.id === id)?.label ?? PACK_LABELS[id] ?? id;
}

export const PREFILL_SIZES: readonly number[] = [8192, 16384, 32768, 65536, 131072, 262144];
export const DEFAULT_PREFILL_SIZES: readonly number[] = [8192, 16384, 32768, 65536, 131072];

export function toggleSize(selected: ReadonlySet<number>, size: number): Set<number> {
  const next = new Set(selected);
  if (next.has(size)) {
    if (next.size > 1) next.delete(size);
  } else {
    next.add(size);
  }
  return next;
}

export function sortSizes(selected: Iterable<number>): number[] {
  return [...new Set([...selected].filter((n) => Number.isInteger(n) && n >= 16))].sort((a, b) => a - b);
}

export type DecodeConfig = {
  pack: string;
  levels: number[];
  maxTokens: number;
  floor: number;
  sloMs: number;
};

export type PrefillConfig = { sizes: number[] };

export function defaultDecodeConfig(): DecodeConfig {
  return {
    pack: DEFAULT_PACK,
    levels: [...DEFAULT_LEVELS],
    maxTokens: DEFAULT_TOKENS,
    floor: DEFAULT_FLOOR,
    sloMs: DEFAULT_SLO_MS,
  };
}

export function defaultPrefillConfig(): PrefillConfig {
  return { sizes: [...DEFAULT_PREFILL_SIZES] };
}

function intList(raw: string | null): number[] {
  if (!raw) return [];
  return raw
    .split(/[,\s]+/)
    .map((s) => Number(s))
    .filter((n) => Number.isInteger(n) && n > 0);
}

function int(raw: string | null, lo: number, hi: number): number | null {
  if (raw == null || raw === "") return null;
  const n = Number(raw);
  return Number.isFinite(n) && n >= lo && n <= hi ? n : null;
}

/** `?tab=decode&pack=prose&levels=1,2,4&tokens=512&floor=20&slo=500` */
export function decodeConfigToQuery(cfg: DecodeConfig): string {
  const q = new URLSearchParams();
  q.set("tab", "decode");
  q.set("pack", cfg.pack);
  q.set("levels", cfg.levels.join(","));
  q.set("tokens", String(cfg.maxTokens));
  if (cfg.floor !== DEFAULT_FLOOR) q.set("floor", String(cfg.floor));
  if (cfg.sloMs !== DEFAULT_SLO_MS) q.set("slo", String(cfg.sloMs));
  return q.toString();
}

export function decodeConfigFromQuery(q: URLSearchParams, base = defaultDecodeConfig()): DecodeConfig {
  const levels = sortConcurrencies(intList(q.get("levels")));
  const tokens = int(q.get("tokens"), 1, 65_536);
  const floor = int(q.get("floor"), 0, 10_000);
  const slo = int(q.get("slo"), 0, 600_000);
  const pack = q.get("pack");
  return {
    pack: pack && /^[a-z0-9-]+$/i.test(pack) ? pack : base.pack,
    levels: levels.length ? levels : base.levels,
    maxTokens: tokens ?? base.maxTokens,
    floor: floor ?? base.floor,
    sloMs: slo ?? base.sloMs,
  };
}

export function prefillConfigToQuery(cfg: PrefillConfig): string {
  const q = new URLSearchParams();
  q.set("tab", "prefill");
  q.set("sizes", cfg.sizes.join(","));
  return q.toString();
}

export function prefillConfigFromQuery(q: URLSearchParams, base = defaultPrefillConfig()): PrefillConfig {
  const sizes = sortSizes(intList(q.get("sizes")));
  return { sizes: sizes.length ? sizes : base.sizes };
}
