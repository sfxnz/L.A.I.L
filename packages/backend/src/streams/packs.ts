import type { StreamPack, StreamPromptRef } from "@lail/shared";

type PackPrompt = { title: string; text: string; pack: string };
type Pack = Omit<StreamPack, "prompts"> & { prompts: PackPrompt[] };

// The four decode-bench families, verbatim from serve-engine perf.py (_WORKLOAD_FAMILY).
const FAMILIES: Pack[] = [
  {
    id: "prose",
    label: "Prose",
    default_max_tokens: 256,
    prompts: [
      {
        pack: "prose",
        title: "prose_essay",
        text:
          "Continue this essay in the same voice. Do not stop.\n\n" +
          "Decode throughput and time-to-first-token feel different when a coding " +
          "agent shares a long system prompt across tabs on a DGX Spark with unified " +
          "memory. The KV cache is the product, not a leftover after util. ",
      },
    ],
  },
  {
    id: "structured",
    label: "Structured",
    default_max_tokens: 256,
    prompts: [
      {
        pack: "structured",
        title: "structured_fields",
        text:
          "Repeat the following labeled fields over and over. Do not stop.\n\n" +
          "model: Qwen3.6-27B-NVFP4\nquant: nvfp4\ncontext_len: 32768\ntp: 2\n" +
          "prefill_tok_s: 410\ndecode_tok_s: 61\nkv_policy: prefix_cache\n" +
          "headroom_gib: 18\nnotes: GB10 UMA, QSFP RoCE up\n",
      },
    ],
  },
  {
    id: "code",
    label: "Code",
    default_max_tokens: 256,
    prompts: [
      {
        pack: "code",
        title: "code_impl",
        text:
          "Continue this Python module. No markdown fences. Do not stop.\n\n" +
          "from __future__ import annotations\n\n" +
          "class TokenBucket:\n" +
          "    def __init__(self, rate: float, burst: int) -> None:\n" +
          "        self.rate = rate\n" +
          "        self.burst = burst\n" +
          "        self.tokens = float(burst)\n",
      },
    ],
  },
  {
    id: "json",
    label: "JSON",
    default_max_tokens: 256,
    prompts: [
      {
        pack: "json",
        title: "json_object",
        text:
          "Continue this JSON array. Valid JSON only. Do not stop.\n\n" +
          '[{"spark_id":"spark1","serving":true,"temperature_c":47,' +
          '"gpu_util_pct":12,"decode_tok_per_s":61.2},',
      },
    ],
  },
];

const CHAT_SHORT: Pack = {
  id: "chat-short",
  label: "Chat (short)",
  default_max_tokens: 256,
  prompts: [
    ["tcp_vs_udp", "Explain the difference between TCP and UDP in three sentences."],
    ["haiku", "Write a haiku about a GPU cluster humming at 3 a.m."],
    [
      "kv_cache",
      "In one paragraph, explain how KV caching speeds up LLM decoding and what it costs in memory.",
    ],
    [
      "dashboard_names",
      "Suggest three names for a home-lab monitoring dashboard and explain each in one line.",
    ],
    [
      "concise_rewrite",
      "Rewrite this sentence to be more concise: 'Due to the fact that the model was quantized, it is the case that memory usage went down.'",
    ],
    [
      "ttft_causes",
      "List five common causes of high time-to-first-token in an inference server, one line each.",
    ],
    ["translate", "Translate 'the cache is warm' into French, German and Spanish."],
    [
      "regex",
      "Write a regular expression that matches an IPv4 address and explain each part briefly.",
    ],
  ].map(([title, text]) => ({ pack: "chat-short", title, text })),
};

// Rotates through the other packs: one prompt from each family, then four short chats.
const MIXED: Pack = {
  id: "mixed",
  label: "Mixed",
  default_max_tokens: 256,
  prompts: [...FAMILIES.map((f) => f.prompts[0]), ...CHAT_SHORT.prompts.slice(0, 4)],
};

const PACKS: Pack[] = [...FAMILIES, CHAT_SHORT, MIXED];

export function listPacks(): StreamPack[] {
  return PACKS;
}

export function getPack(id: string): Pack | undefined {
  return PACKS.find((p) => p.id === id);
}

/**
 * Round-robin prompt assignment. Each strand gets a unique marker line so prefix
 * caching cannot share KV blocks between strands; the marker goes in front of the
 * family text (a trailing marker would leave the shared prefix cacheable and sit
 * between the "continue this…" fragment and the model's continuation).
 */
export function assignPrompts(
  pack: Pack,
  count: number,
  nonce: string,
  startIndex = 0,
  level?: number,
): StreamPromptRef[] {
  const out: StreamPromptRef[] = [];
  for (let k = 0; k < count; k++) {
    const i = startIndex + k;
    const p = pack.prompts[k % pack.prompts.length];
    out.push({
      i,
      title: p.title,
      text: `[strand ${i} · ${nonce}]\n\n${p.text}`,
      pack: p.pack,
      ...(level === undefined ? {} : { level }),
    });
  }
  return out;
}
