/**
 * Unit tests for L.A.I.L console chrome helpers.
 * Run: `cd apps/web && bun test lib/ide-chrome.test.ts`
 */
import { describe, expect, test } from "bun:test";
import { WORKSPACE_NAV, isProseRoute } from "./ide-chrome";

describe("ide-chrome nav contract", () => {
  test("exposes the six job-based nav destinations", () => {
    expect(WORKSPACE_NAV.map((n) => n.label)).toEqual([
      "Status",
      "Serve",
      "Bench",
      "Streams",
      "Evals",
      "Configure",
    ]);
    expect(WORKSPACE_NAV.map((n) => n.href)).toEqual([
      "/status",
      "/server",
      "/bench",
      "/streams",
      "/evals",
      "/configure",
    ]);
  });

  test("only Configure is a prose-width page", () => {
    expect(isProseRoute("/configure")).toBe(true);
    expect(isProseRoute("/configure/backends")).toBe(true);
    expect(isProseRoute("/status")).toBe(false);
    expect(isProseRoute("/streams")).toBe(false);
  });
});
