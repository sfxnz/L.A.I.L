/**
 * Structural tests: shipped AppShell must contain serve/evals chrome labels.
 */
import { describe, expect, test } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";
import { WORKSPACE_NAV_LABELS } from "./ide-chrome";

const webRoot = join(import.meta.dir, "..");

describe("shipped AppShell source labels", () => {
  const shell = readFileSync(join(webRoot, "components/layout/AppShell.tsx"), "utf8");

  test("is a top-nav console shell (not IDE sidebar)", () => {
    expect(shell).toContain("Local AI Lab");
    expect(shell).toContain("vLLM");
    expect(shell).not.toContain("Pinned");
    expect(shell).not.toContain("Tasks");
    expect(shell).not.toContain('placeholder="Search"');
  });

  test("contains all workspace nav labels", () => {
    for (const label of WORKSPACE_NAV_LABELS) {
      expect(shell).toContain(label);
    }
  });
});

describe("shipped Workbench retirement page", () => {
  const wb = readFileSync(join(webRoot, "app/workbench/page.tsx"), "utf8");

  test("points users to Hermes / Serve / Evals / Connect", () => {
    expect(wb).toContain("Hermes");
    expect(wb).toContain("/server");
    expect(wb).toContain("/evals");
    expect(wb).toContain("/connect");
    expect(wb).not.toContain("sfxnz.github.io");
    expect(wb).not.toContain('href="/lab"');
  });
});

describe("shipped Evals route exists", () => {
  test("evals page present", () => {
    const p = readFileSync(join(webRoot, "app/evals/page.tsx"), "utf8");
    expect(p).toContain("Run smoke");
    expect(p).toContain("benchPerf");
  });
});
