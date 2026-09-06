/**
 * Quantisation, parsed from the model id. Ordered so the most specific token
 * wins (NVFP4 before FP4, GPTQ/AWQ before INT4). Null when the id says nothing —
 * the card shows <Nil/>; we never guess "BF16".
 */
const RULES: ReadonlyArray<[RegExp, string]> = [
  [/nvfp4/i, "NVFP4"],
  [/mxfp4/i, "MXFP4"],
  [/fp8/i, "FP8"],
  [/awq/i, "AWQ"],
  [/gptq/i, "GPTQ"],
  [/gguf/i, "GGUF"],
  [/\bexl2\b|-exl2/i, "EXL2"],
  [/w4a16|int4|4bit|-4bit/i, "INT4"],
  [/w8a8|w8a16|int8|8bit|-8bit/i, "INT8"],
  [/fp4/i, "FP4"],
  [/bf16/i, "BF16"],
  [/fp16/i, "FP16"],
];

export function parseQuant(modelId: string | null | undefined): string | null {
  if (!modelId) return null;
  const tail = modelId.split("/").pop() || modelId;
  for (const [re, label] of RULES) if (re.test(tail)) return label;
  return null;
}
