/** Unique-prefix prompts of a target token size for the prefill bench. */

const WORDS =
  "the cache stays warm while every strand of the cluster streams tokens through unified memory and the scheduler balances prefill against decode so that latency stays flat under load as batches grow blocks fill pages spill routers hum fabrics sync clocks drift nodes agree logs scroll gauges settle".split(
    " ",
  );

/** Deterministic filler: same (nonce, chars) → same text, so resizing only extends or trims. */
export function fillerText(nonce: string, chars: number): string {
  let seed = 2166136261;
  for (const ch of nonce) seed = (Math.imul(seed ^ ch.charCodeAt(0), 16777619) >>> 0) || 1;
  const parts: string[] = [`Document ${nonce}.`];
  let len = parts[0].length;
  let n = 0;
  while (len <= chars) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    let w = WORDS[seed % WORDS.length];
    n++;
    if (n % 11 === 0) w += ".";
    if (n % 97 === 0) w += "\n";
    parts.push(w);
    len += w.length + 1;
  }
  return parts.join(" ").slice(0, Math.max(chars, parts[0].length));
}

export type TokenCounter = (text: string) => Promise<number | null>;

/** vLLM `POST /tokenize` on the chat-templated prompt; null when the route is unavailable. */
export function vllmTokenCounter(baseUrl: string, model: string, signal: AbortSignal): TokenCounter {
  return async (content) => {
    try {
      const r = await fetch(`${baseUrl}/tokenize`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content }], add_generation_prompt: true }),
        signal,
      });
      if (!r.ok) return null;
      const j = (await r.json()) as { count?: number };
      return typeof j.count === "number" ? j.count : null;
    } catch {
      return null;
    }
  };
}

export type PrefillSizer = {
  /** chars per token; starts at 3.5 and is corrected from `usage.prompt_tokens` when the tokenizer is unavailable. */
  charsPerToken: number;
  /** undefined until the first call; false once `/tokenize` failed. */
  tokenizeAvailable?: boolean;
};

/**
 * Build a prompt of ~`size` tokens. With the tokenizer: iterate until within 2 %.
 * Without: estimate from `charsPerToken` (the caller corrects it from `usage`).
 */
export async function buildPrefillPrompt(
  sizer: PrefillSizer,
  count: TokenCounter,
  size: number,
  nonce: string,
): Promise<{ text: string; measured: number | null }> {
  let chars = Math.round(size * sizer.charsPerToken);
  let text = fillerText(nonce, chars);
  if (sizer.tokenizeAvailable === false) return { text, measured: null };
  for (let attempt = 0; attempt < 5; attempt++) {
    const n = await count(text);
    if (n === null) {
      sizer.tokenizeAvailable = false;
      return { text, measured: null };
    }
    sizer.tokenizeAvailable = true;
    if (Math.abs(n - size) / size <= 0.02) return { text, measured: n };
    chars = Math.max(16, Math.round((chars * size) / n));
    text = fillerText(nonce, chars);
  }
  return { text, measured: await count(text) };
}
