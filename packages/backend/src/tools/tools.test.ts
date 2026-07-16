import { describe, expect, test, beforeAll, afterAll } from "bun:test";
import { existsSync, mkdirSync, writeFileSync, readFileSync, rmSync } from "fs";
import { join } from "path";
import { createWorkspace } from "../controller/workspaces";
import { runTool, toolDefinitions } from "./index";

const TMP = `/tmp/lail-tools-test-${process.pid}`;
let workspaceId = "";

describe("toolDefinitions", () => {
  test("includes review-first edit tools, not write_file", () => {
    const names = toolDefinitions.map((t) => t.function.name);
    expect(names).toContain("search_replace");
    expect(names).toContain("create_file");
    expect(names).toContain("delete_file");
    expect(names).not.toContain("write_file");
    expect(names).toContain("list_dir");
    expect(names).toContain("read_file");
    expect(names).toContain("grep");
    expect(names).toContain("plan");
    expect(names).toContain("run_shell");
  });
});

describe("patch proposal tools (no disk write)", () => {
  beforeAll(() => {
    rmSync(TMP, { recursive: true, force: true });
    mkdirSync(TMP, { recursive: true });
    writeFileSync(join(TMP, "existing.txt"), "hello world\n", "utf8");
    const ws = createWorkspace(`tools-test-${process.pid}`, TMP);
    workspaceId = ws.id;
  });

  afterAll(() => {
    rmSync(TMP, { recursive: true, force: true });
  });

  test("search_replace returns patchProposal without writing", async () => {
    const before = readFileSync(join(TMP, "existing.txt"), "utf8");
    const result = await runTool(workspaceId, "search_replace", {
      path: "existing.txt",
      old_string: "hello",
      new_string: "goodbye",
    });
    expect(result.ok).toBe(true);
    expect(result.patchProposal).toEqual({
      path: "existing.txt",
      oldString: "hello",
      newString: "goodbye",
      op: "replace",
    });
    expect(readFileSync(join(TMP, "existing.txt"), "utf8")).toBe(before);
  });

  test("create_file returns patchProposal without writing", async () => {
    const result = await runTool(workspaceId, "create_file", {
      path: "new-file.txt",
      content: "brand new",
    });
    expect(result.ok).toBe(true);
    expect(result.patchProposal).toEqual({
      path: "new-file.txt",
      oldString: "",
      newString: "brand new",
      op: "create",
    });
    expect(existsSync(join(TMP, "new-file.txt"))).toBe(false);
  });

  test("delete_file returns patchProposal without deleting", async () => {
    const result = await runTool(workspaceId, "delete_file", {
      path: "existing.txt",
    });
    expect(result.ok).toBe(true);
    expect(result.patchProposal).toEqual({
      path: "existing.txt",
      oldString: "",
      newString: "",
      op: "delete",
    });
    expect(existsSync(join(TMP, "existing.txt"))).toBe(true);
  });

  test("rejects absolute paths", async () => {
    const result = await runTool(workspaceId, "search_replace", {
      path: "/etc/passwd",
      old_string: "a",
      new_string: "b",
    });
    expect(result.ok).toBe(false);
    expect(result.patchProposal).toBeUndefined();
  });
});
