import { describe, expect, test } from "bun:test";
import { runHref } from "./run-href";

describe("runHref", () => {
  test("each kind opens its own view; kinds without one have no link", () => {
    expect(runHref({ kind: "decode", run_id: "a b" })).toBe("/bench?run=a%20b");
    expect(runHref({ kind: "prefill", run_id: "p" })).toBe("/bench?run=p");
    expect(runHref({ kind: "agentic_tool_eval", run_id: "t" })).toBe("/evals/tool/t");
    expect(runHref({ kind: "legacy_decode", run_id: "x" })).toBeNull();
    expect(runHref({ kind: "legacy_perf_workflow", run_id: "x" })).toBeNull();
  });
});
