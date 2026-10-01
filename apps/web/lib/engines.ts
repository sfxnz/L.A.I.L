/** Display names for the serving engines the serve-engine detects and launches. */
const LABELS: Record<string, string> = {
  vllm: "vLLM",
  sglang: "SGLang",
  llamacpp: "llama.cpp",
  tensorfold: "TensorFold",
};

/** "vLLM" for "vllm"; an unknown name as-is; null when nothing was detected. */
export function engineLabel(name: string | null | undefined): string | null {
  if (!name) return null;
  return LABELS[name.toLowerCase()] ?? name;
}
