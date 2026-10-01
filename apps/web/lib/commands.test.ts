import { describe, expect, test } from "bun:test";
import type { LabStatus } from "./api";
import {
  BENCH_DECODE_HREF,
  COMMANDS,
  filterCommands,
  groupCommands,
  scoreCommand,
  visibleCommands,
  type CommandContext,
} from "./commands";

function ctx(over: Partial<CommandContext> = {}): CommandContext {
  return {
    router: { push: () => {}, replace: () => {} },
    pathname: "/status",
    status: null,
    needToken: false,
    strands: { running: 0, waiting: 0 },
    theme: { setChoice: () => {} },
    toast: () => {},
    openSheet: () => {},
    ...over,
  };
}

const serving: LabStatus = {
  controller: "ok",
  defaultBackend: "vllm",
  defaultModel: "auto",
  openAiBase: "http://127.0.0.1:8000/v1",
  backends: {},
  serve: { healthy: true, base_url: "http://127.0.0.1:8000", model_id: "org/model-35b" },
};

describe("command registry", () => {
  test("ids are unique and every command sits in a known group", () => {
    const ids = COMMANDS.map((c) => c.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(groupCommands(COMMANDS).map(([g]) => g)).toEqual(["Navigate", "Run", "Copy", "Theme", "View", "Help"]);
  });

  test("nav commands carry their g-chord in display form", () => {
    const bench = COMMANDS.find((c) => c.id === "nav:/bench")!;
    expect(bench.shortcut).toEqual(["g", "b"]);
    expect(COMMANDS.find((c) => c.id === "nav:/usage")?.shortcut).toBeUndefined();
    expect(COMMANDS.find((c) => c.id === "nav:/lab")?.hint).toBe("/lab");
    expect(COMMANDS.find((c) => c.id === "help:shortcuts")?.shortcut).toEqual(["?"]);
  });

  test("`when` gates: idle lab hides Run/Copy/View, serving shows them", () => {
    const idle = visibleCommands(COMMANDS, ctx()).map((c) => c.id);
    expect(idle).not.toContain("run:bench-decode");
    expect(idle).not.toContain("copy:model");
    expect(idle).not.toContain("run:stop");
    expect(idle).not.toContain("view:wall");
    expect(idle).not.toContain("help:token");

    const live = visibleCommands(COMMANDS, ctx({ status: serving, pathname: "/streams" })).map((c) => c.id);
    expect(live).toEqual(expect.arrayContaining(["run:bench-decode", "run:streams-prose-8", "copy:endpoint", "copy:model", "view:wall"]));
    expect(live).not.toContain("view:details");

    const running = visibleCommands(COMMANDS, ctx({ status: serving, strands: { running: 3, waiting: 1 } })).map((c) => c.id);
    expect(running).toContain("run:stop");
    expect(running).not.toContain("run:streams-prose-8");

    expect(visibleCommands(COMMANDS, ctx({ needToken: true })).map((c) => c.id)).toContain("help:token");
    expect(visibleCommands(COMMANDS, ctx({ pathname: "/bench" })).map((c) => c.id)).toContain("view:details");
  });

  test("bench run command hands the page its own query params", () => {
    const q = new URLSearchParams(BENCH_DECODE_HREF.split("?")[1]);
    expect(q.get("pack")).toBe("prose");
    expect(q.get("levels")).toBe("1,2,4");
    expect(q.get("tokens")).toBe("512");
    let pushed = "";
    void COMMANDS.find((c) => c.id === "run:bench-decode")!.run(ctx({ router: { push: (h) => (pushed = h), replace: () => {} } }));
    expect(pushed).toBe(BENCH_DECODE_HREF);
  });
});

describe("fuzzy filter", () => {
  test("label prefix beats word prefix beats substring beats subsequence", () => {
    const c = (label: string, hint?: string) => ({ label, hint, group: "Navigate" as const });
    expect(scoreCommand(c("Bench"), "ben")).toBe(5);
    expect(scoreCommand(c("Streams: load prose ×8"), "load")).toBe(4);
    expect(scoreCommand(c("Copy endpoint URL", "the served base URL"), "served")).toBe(3);
    expect(scoreCommand(c("Configure"), "cfg")).toBe(1);
    expect(scoreCommand(c("Status"), "zzz")).toBe(0);
    expect(scoreCommand(c("Status"), "")).toBe(1);
  });

  test("'bench' puts the Bench room first and keeps registry order on ties", () => {
    const out = filterCommands(COMMANDS, "bench");
    expect(out[0].id).toBe("nav:/bench");
    expect(out.map((c) => c.id)).toContain("run:bench-decode");
    expect(out.map((c) => c.id)).not.toContain("nav:/status");
  });

  test("matches group names and treats × as x", () => {
    expect(filterCommands(COMMANDS, "theme").slice(0, 3).map((c) => c.group)).toEqual(["Theme", "Theme", "Theme"]);
    expect(filterCommands(COMMANDS, "x8").map((c) => c.id)).toContain("run:streams-prose-8");
  });
});
