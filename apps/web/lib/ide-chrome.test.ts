/**
 * Unit tests for L.A.I.L console chrome helpers.
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

describe("ide-chrome nav contract", () => {
  test("exposes lab section label", () => {
    expect([...SIDEBAR_SECTION_LABELS]).toEqual(["Lab"]);
  });

  test("exposes serve+evals+lab nav destinations", () => {
    expect([...WORKSPACE_NAV_LABELS]).toEqual([
      "Status",
      "Serve",
      "Evals",
      "Lab",
      "Connect",
      "Models",
      "Configure",
    ]);
    expect(WORKSPACE_NAV.map((n) => n.href)).toEqual([
      "/status",
      "/server",
      "/evals",
      "/lab",
      "/connect",
      "/models",
      "/configure",
    ]);
  });
});

describe("ide-chrome stream helpers (legacy agent)", () => {
  test("follow-up placeholder kept for API compat", () => {
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

  test("agent modes expose Plan / Ask / Agent labels", () => {
    expect([...AGENT_MODES]).toEqual(["plan", "ask", "agent"]);
    expect(AGENT_MODE_LABELS.plan).toBe("Plan");
    expect(AGENT_MODE_LABELS.ask).toBe("Ask");
    expect(AGENT_MODE_LABELS.agent).toBe("Agent");
  });

  test("groupTimeline maps patch kind to patch stream block", () => {
    const blocks = groupTimeline([{ kind: "patch", text: "apps/web/lib/ide-chrome.ts" }]);
    expect(blocks).toEqual([{ type: "patch", path: "apps/web/lib/ide-chrome.ts" }]);
  });

  test("one tool_start + tool_end pair counts as Ran 1 command (not 2)", () => {
    const blocks = groupTimeline([
      { kind: "tool", text: "run_shell", meta: { phase: "start" } },
      { kind: "ran", text: "run_shell" },
    ]);
    expect(blocks).toEqual([{ type: "ran", count: 1, detail: "run_shell" }]);
  });

  test("two tool_end events count as Ran 2 commands", () => {
    const blocks = groupTimeline([
      { kind: "ran", text: "run_shell" },
      { kind: "ran", text: "read_file" },
    ]);
    expect(blocks[0]).toEqual({ type: "ran", count: 2, detail: "read_file" });
  });

  test("groupTimeline maps full agent turn without double-counting tools", () => {
    const blocks = groupTimeline([
      { kind: "thought", text: "planning" },
      { kind: "tool", text: "run_shell" },
      { kind: "ran", text: "ls" },
      { kind: "status", text: "ok" },
    ]);
    expect(blocks.map((b) => b.type)).toEqual(["thought", "ran", "status"]);
  });

  test("ranLabel and fileWriteLabel", () => {
    expect(ranLabel(1)).toBe("Ran 1 command");
    expect(ranLabel(2, "ls")).toBe("Ran 2 commands · ls");
    expect(fileWriteLabel("a.html", true)).toBe("Creating a.html");
    expect(fileWriteLabel("a.html", false)).toBe("Wrote a.html");
  });
});
