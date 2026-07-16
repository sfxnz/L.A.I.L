import { describe, expect, test, beforeEach, afterEach } from "bun:test";
import { mkdirSync, writeFileSync, rmSync } from "fs";
import { join } from "path";
import { loadIgnore, isIgnored } from "./ignore";

const TMP = `/tmp/lail-ignore-${process.pid}`;

describe("ignore", () => {
  beforeEach(() => {
    rmSync(TMP, { recursive: true, force: true });
    mkdirSync(TMP, { recursive: true });
    writeFileSync(join(TMP, ".gitignore"), "dist/\n*.log\n");
    writeFileSync(join(TMP, ".lailignore"), "secrets/\n");
  });
  afterEach(() => rmSync(TMP, { recursive: true, force: true }));

  test("default skips node_modules and .git", () => {
    const ig = loadIgnore(TMP);
    expect(isIgnored(ig, "node_modules/x")).toBe(true);
    expect(isIgnored(ig, ".git/config")).toBe(true);
    expect(isIgnored(ig, "src/a.ts")).toBe(false);
  });

  test("respects gitignore and lailignore", () => {
    const ig = loadIgnore(TMP);
    expect(isIgnored(ig, "dist/out.js")).toBe(true);
    expect(isIgnored(ig, "foo.log")).toBe(true);
    expect(isIgnored(ig, "secrets/key")).toBe(true);
  });
});
