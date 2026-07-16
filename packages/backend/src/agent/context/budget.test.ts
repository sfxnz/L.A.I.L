import { describe, expect, test } from "bun:test";
import { applyBudget, truncateBody } from "./budget";
import type { ContextChunk } from "./types";

describe("applyBudget", () => {
  test("keeps high priority when over budget", () => {
    const chunks: ContextChunk[] = [
      { kind: "open_tab", label: "tab", body: "x".repeat(1000), priority: 40 },
      { kind: "mention_file", label: "m", body: "y".repeat(100), priority: 20 },
    ];
    const r = applyBudget(chunks, 200);
    expect(r.chunks.some((c) => c.kind === "mention_file")).toBe(true);
    expect(r.truncated).toBe(true);
  });
});

describe("truncateBody", () => {
  test("head-tail truncates long body", () => {
    const r = truncateBody("a".repeat(1000), 100);
    expect(r.length).toBeLessThanOrEqual(120); // marker overhead
    expect(r.includes("…")).toBe(true);
  });
});
