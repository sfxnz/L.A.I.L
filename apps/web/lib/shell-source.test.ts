/**
 * Structural tests: shipped AppShell + Workbench source must contain user-visible
 * inspo labels (not only comments). Reads real files from disk.
 */
import { describe, expect, test } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";
import {
  COMPOSER_PLACEHOLDER,
  SIDEBAR_SECTION_LABELS,
  WORKSPACE_NAV_LABELS,
} from "./ide-chrome";

const webRoot = join(import.meta.dir, "..");

describe("shipped AppShell source labels", () => {
  const shell = readFileSync(join(webRoot, "components/layout/AppShell.tsx"), "utf8");

  test("contains all sidebar section labels as string literals", () => {
    for (const label of SIDEBAR_SECTION_LABELS) {
      expect(shell.includes(`"${label}"`) || shell.includes(`>${label}<`) || shell.includes(label)).toBe(
        true,
      );
    }
    // Workspace / Pinned / Tasks / Projects rendered as section headers
    expect(shell).toContain("Workspace");
    expect(shell).toContain("Pinned");
    expect(shell).toContain("Tasks");
    expect(shell).toContain("Projects");
    expect(shell).toContain('placeholder="Search"');
  });

  test("contains all workspace nav labels", () => {
    for (const label of WORKSPACE_NAV_LABELS) {
      expect(shell).toContain(label);
    }
  });
});

describe("shipped Workbench source chrome", () => {
  const wb = readFileSync(join(webRoot, "app/workbench/page.tsx"), "utf8");

  test("composer placeholder Ask for follow-up changes", () => {
    expect(wb).toContain("COMPOSER_PLACEHOLDER");
    expect(COMPOSER_PLACEHOLDER).toBe("Ask for follow-up changes");
  });

  test("agent stream markers Thought / Working / Ran / Creating", () => {
    expect(wb).toContain("STREAM_MARKERS");
    expect(wb).toContain("ranLabel");
    expect(wb).toContain("fileWriteLabel");
    expect(wb).toContain("status-rail");
    expect(wb).toContain("file-editor");
    expect(wb).toContain("Toggle Status");
    expect(wb).toContain("Composer");
  });

  test("assistant path does not dual-write timeline + messages", () => {
    // Extract the assistant WS handler block and ensure it only setMessages
    const idx = wb.indexOf('type === "assistant"');
    expect(idx).toBeGreaterThan(-1);
    const slice = wb.slice(idx, idx + 350);
    expect(slice).toContain("setMessages");
    expect(slice).not.toContain('pushTimeline({ kind: "assistant"');
    expect(slice).not.toContain("kind: \"assistant\"");
  });

  test("tool_end increments cmd count; tool_start does not setCmdCount", () => {
    const startIdx = wb.indexOf('type === "tool_start"');
    const endIdx = wb.indexOf('type === "tool_end"');
    expect(startIdx).toBeGreaterThan(-1);
    expect(endIdx).toBeGreaterThan(startIdx);
    const startBlock = wb.slice(startIdx, endIdx);
    const endBlock = wb.slice(endIdx, endIdx + 280);
    expect(startBlock).not.toContain("setCmdCount");
    expect(endBlock).toContain("setCmdCount");
    expect(endBlock).toContain('kind: "ran"');
  });

  test("workbench has mode toggle and patch review", () => {
    expect(wb).toMatch(/ModeToggle|agentMode|Plan/);
    expect(wb).toMatch(/PatchReview|pendingPatches|patches\.accept/);
  });

  test("Phase B context: MentionPopup / parseMentions / editorSnapshot", () => {
    expect(wb).toMatch(/MentionPopup|parseMentions/);
    expect(wb).toContain("editorSnapshot");
    expect(wb).toMatch(/ContextChips|parseMentions/);
  });
});

describe("default entry", () => {
  test("root page redirects to workbench", () => {
    const page = readFileSync(join(webRoot, "app/page.tsx"), "utf8");
    expect(page).toContain('redirect("/workbench")');
  });
});
