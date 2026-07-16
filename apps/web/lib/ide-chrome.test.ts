/**
 * Unit tests for shipped IDE chrome helpers (real module under apps/web/lib).
 * Run: `cd apps/web && bun test lib/ide-chrome.test.ts`
 */
import { describe, expect, test } from "bun:test";
import {
  AGENT_MODES,
  AGENT_MODE_LABELS,
  COMPOSER_PLACEHOLDER,
  SIDEBAR_SECTION_LABELS,
  STREAM_MARKERS,
  WORKSPACE_NAV,
  WORKSPACE_NAV_LABELS,
  fileWriteLabel,
  groupTimeline,
  ranLabel,
} from "./ide-chrome";

describe("ide-chrome sidebar contract", () => {
  test("exposes inspo section labels", () => {
    expect([...SIDEBAR_SECTION_LABELS]).toEqual([
      "Search",
      "Workspace",
      "Pinned",
      "Tasks",
      "Projects",
    ]);
  });

  test("exposes full workspace nav destinations", () => {
    expect([...WORKSPACE_NAV_LABELS]).toEqual([
      "Status",
      "Workbench",
      "Models",
      "Configure",
      "Usage",
      "Integrations",
      "Server",
    ]);
    expect(WORKSPACE_NAV.map((n) => n.href)).toEqual([
      "/status",
      "/workbench",
      "/models",
      "/configure",
      "/usage",
      "/integrations",
      "/server",
    ]);
  });
});

describe("ide-chrome composer / stream", () => {
  test("follow-up placeholder matches inspo", () => {
    expect(COMPOSER_PLACEHOLDER).toBe("Ask for follow-up changes");
  });

  test("stream markers include Thought, Working, Ran, Creating, Status, Proposed", () => {
    expect(STREAM_MARKERS.thought).toBe("Thought");
    expect(STREAM_MARKERS.working).toBe("Working");
    expect(STREAM_MARKERS.ran).toBe("Ran");
    expect(STREAM_MARKERS.creating).toBe("Creating");
    expect(STREAM_MARKERS.status).toBe("Status");
    expect(STREAM_MARKERS.proposed).toBe("Proposed");
  });

  test("agent modes expose Cursor-style Plan / Ask / Agent labels", () => {
    expect([...AGENT_MODES]).toEqual(["plan", "ask", "agent"]);
    expect(AGENT_MODE_LABELS.plan).toBe("Plan");
    expect(AGENT_MODE_LABELS.ask).toBe("Ask");
    expect(AGENT_MODE_LABELS.agent).toBe("Agent");
  });

  test("groupTimeline maps patch kind to patch stream block", () => {
    const blocks = groupTimeline([
      { kind: "patch", text: "apps/web/lib/ide-chrome.ts" },
    ]);
    expect(blocks).toEqual([{ type: "patch", path: "apps/web/lib/ide-chrome.ts" }]);
  });


  test("one tool_start + tool_end pair counts as Ran 1 command (not 2)", () => {
    // Real WS pairing: tool_start → kind tool (ignored for count), tool_end → kind ran
    const blocks = groupTimeline([
      { kind: "tool", text: "run_shell", meta: { phase: "start" } },
      {
        kind: "ran",
        text: "ls -la",
        meta: { phase: "end", output: "README.md\n" },
      },
    ]);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].type).toBe("ran");
    if (blocks[0].type === "ran") {
      expect(blocks[0].count).toBe(1);
      expect(ranLabel(blocks[0].count, blocks[0].detail)).toBe("Ran 1 command · ls -la");
      expect(blocks[0].output).toContain("README.md");
    }
  });

  test("two tool_end events count as Ran 2 commands", () => {
    const blocks = groupTimeline([
      { kind: "tool", text: "run_shell" },
      { kind: "ran", text: "ls" },
      { kind: "tool", text: "read_file" },
      { kind: "ran", text: "Read README.md" },
    ]);
    expect(blocks).toHaveLength(1);
    if (blocks[0].type === "ran") {
      expect(blocks[0].count).toBe(2);
      expect(ranLabel(blocks[0].count)).toBe("Ran 2 commands");
    }
  });

  test("groupTimeline maps full agent turn without double-counting tools", () => {
    const blocks = groupTimeline([
      { kind: "thought", text: "I'll explore the app" },
      { kind: "tool", text: "run_shell" },
      { kind: "ran", text: "ls -la", meta: { output: "README.md\n" } },
      { kind: "file", text: "local-ai-survival-guide.md", meta: { creating: true } },
      { kind: "status", text: "Creating guide" },
      { kind: "assistant", text: "Done writing the guide." },
    ]);
    expect(blocks[0]).toEqual({ type: "thought", text: "I'll explore the app" });
    expect(blocks[1].type).toBe("ran");
    if (blocks[1].type === "ran") {
      expect(blocks[1].count).toBe(1);
    }
    expect(blocks[2]).toEqual({
      type: "file",
      path: "local-ai-survival-guide.md",
      creating: true,
    });
    expect(fileWriteLabel("local-ai-survival-guide.md", true)).toBe(
      "Creating local-ai-survival-guide.md",
    );
    expect(blocks[3]).toEqual({ type: "status", text: "Creating guide" });
    // assistant must NOT appear in stream (messages pane only — no duplicate bubbles)
    expect(blocks.some((b) => b.type === "assistant")).toBe(false);
  });
});
