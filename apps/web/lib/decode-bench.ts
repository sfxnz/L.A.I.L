export const WORKLOAD_KINDS = ["structured", "prose", "code", "json"] as const;
export type WorkloadKind = (typeof WORKLOAD_KINDS)[number];

export const WORKLOAD_LABELS: Record<WorkloadKind, string> = {
  structured: "Structured",
  prose: "Prose",
  code: "Code",
  json: "JSON",
};

export const CONCURRENCY_LEVELS: readonly number[] = Array.from(
  { length: 32 },
  (_, i) => i + 1,
);

export function sortConcurrencies(selected: Iterable<number>): number[] {
  return [...selected].filter((n) => n >= 1 && n <= 32).sort((a, b) => a - b);
}

export function isWorkloadKind(v: unknown): v is WorkloadKind {
  return typeof v === "string" && (WORKLOAD_KINDS as readonly string[]).includes(v);
}

export function decodeRunLabel(summary: Record<string, unknown> | undefined): string | null {
  const w = summary?.workload;
  if (!isWorkloadKind(w)) return null;
  return WORKLOAD_LABELS[w];
}
