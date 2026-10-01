/**
 * Memory Capacity forecast (brief §5.10 #2): from the engine's KV capacity in
 * tokens, how many sequences of a given length fit at once. Pure maths; the
 * caller shows <Nil/> when capacity is unknown — nothing here fabricates.
 */

export const FORECAST_SEQ_TOKENS = 32_768;

export type KvForecast = {
  capacityTokens: number;
  seqTokens: number;
  /** whole sequences of `seqTokens` that fit in the KV pool */
  fits: number;
};

export function kvForecast(
  capacityTokens: number | null | undefined,
  seqTokens = FORECAST_SEQ_TOKENS,
): KvForecast | null {
  if (capacityTokens == null || !Number.isFinite(capacityTokens) || capacityTokens <= 0) return null;
  if (!Number.isFinite(seqTokens) || seqTokens <= 0) return null;
  return { capacityTokens, seqTokens, fits: Math.floor(capacityTokens / seqTokens) };
}

/**
 * Token counts the way the community writes them: exact multiples of 1024 are
 * binary-k (32768 → "32k", 131072 → "128k", 524288 → "512k"); anything else is
 * decimal (1500 → "1.5k", 800 → "800", 2.1M → "2.1M").
 */
export function fmtTokensK(n: number): string {
  if (!Number.isFinite(n)) return "";
  if (n >= 1024 && n % 1024 === 0 && n / 1024 < 1000) return `${n / 1024}k`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  if (n >= 10_000) return `${Math.round(n / 1000)}k`;
  if (n >= 1000) return `${(n / 1000).toFixed(1).replace(/\.0$/, "")}k`;
  return String(Math.round(n));
}

/** "KV capacity 524k tokens · fits 16 × 32k" */
export function forecastLine(f: KvForecast): string {
  return `KV capacity ${fmtTokensK(f.capacityTokens)} tokens · fits ${f.fits} × ${fmtTokensK(f.seqTokens)}`;
}

/**
 * KV contract: the serve-engine sends `engine.kv_usage_pct` as a PERCENT, 0–100, always.
 * No unit guessing — 0.84 means 0.84 %, never 84 %.
 */
export function kvFraction(pct: number): number {
  return Math.max(0, Math.min(1, pct / 100));
}

/** KV tokens in use, from the pool's usage percent. */
export function kvUsedTokens(capacityTokens: number | null | undefined, pct: number | null | undefined): number | null {
  if (capacityTokens == null || pct == null || !Number.isFinite(capacityTokens) || !Number.isFinite(pct)) return null;
  return Math.round(capacityTokens * kvFraction(pct));
}

/**
 * "0.8%" below 10 %, "42%" above; any real use below 0.05 % reads "<0.1%", never "0.0%"
 * (0.04 % of a 1.69M-token pool is ~680 tokens). The branch is chosen on the rounded
 * value, so 9.96 reads "10%", not "10.0%".
 */
export function fmtKvPct(pct: number): string {
  const p = Math.max(0, Math.min(100, pct));
  if (p === 0) return "0%";
  if (p < 0.05) return "<0.1%";
  const tenth = Math.round(p * 10) / 10;
  return tenth < 10 ? `${tenth.toFixed(1)}%` : `${Math.round(p)}%`;
}
