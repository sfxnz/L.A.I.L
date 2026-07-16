import { describe, expect, test, beforeEach, afterEach } from "bun:test";
import { mkdirSync, writeFileSync, readFileSync, rmSync } from "fs";
import { join } from "path";
import { randomUUID } from "crypto";

import {
  applySearchReplaceToContent,
  createPatchStore,
} from "./patch-store";

const TMP = `/tmp/lail-patch-test-${process.pid}`;

describe("applySearchReplaceToContent", () => {
  test("replaces exactly once", () => {
    const r = applySearchReplaceToContent("hello world", "world", "there");
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.content).toBe("hello there");
  });

  test("fails on zero matches", () => {
    const r = applySearchReplaceToContent("abc", "zzz", "q");
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe("no_match");
  });

  test("fails on ambiguous matches", () => {
    const r = applySearchReplaceToContent("aa aa", "aa", "b");
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe("ambiguous");
  });

  test("empty oldString is no_match", () => {
    const r = applySearchReplaceToContent("abc", "", "x");
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe("no_match");
  });
});

describe("PatchStore accept", () => {
  beforeEach(() => {
    rmSync(TMP, { recursive: true, force: true });
    mkdirSync(TMP, { recursive: true });
    writeFileSync(join(TMP, "a.txt"), "line1\nfoo\nline3\n");
  });
  afterEach(() => {
    rmSync(TMP, { recursive: true, force: true });
  });

  test("propose then accept writes disk", () => {
    const store = createPatchStore();
    const patch = store.propose({
      runId: randomUUID(),
      sessionId: randomUUID(),
      path: "a.txt",
      oldString: "foo",
      newString: "bar",
      op: "replace",
      workspaceRoot: TMP,
    });
    expect(patch.status).toBe("pending");
    const applied = store.accept(patch.id, TMP);
    expect(applied.status).toBe("accepted");
    expect(readFileSync(join(TMP, "a.txt"), "utf8")).toContain("bar");
  });

  test("reject leaves file unchanged", () => {
    const store = createPatchStore();
    const patch = store.propose({
      runId: randomUUID(),
      sessionId: randomUUID(),
      path: "a.txt",
      oldString: "foo",
      newString: "bar",
      op: "replace",
      workspaceRoot: TMP,
    });
    const rejected = store.reject(patch.id);
    expect(rejected.status).toBe("rejected");
    expect(readFileSync(join(TMP, "a.txt"), "utf8")).toContain("foo");
    expect(readFileSync(join(TMP, "a.txt"), "utf8")).not.toContain("bar");
  });
});
