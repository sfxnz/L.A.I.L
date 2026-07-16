import { describe, expect, test, beforeEach, afterEach } from "bun:test";
import { mkdirSync, writeFileSync, rmSync } from "fs";
import { join } from "path";
import { ripgrepSearch } from "./search";

const TMP = `/tmp/lail-search-${process.pid}`;

describe("ripgrepSearch", () => {
  beforeEach(() => {
    rmSync(TMP, { recursive: true, force: true });
    mkdirSync(TMP, { recursive: true });
  });
  afterEach(() => rmSync(TMP, { recursive: true, force: true }));

  test("finds unique string in workspace file", async () => {
    if (!Bun.which("rg")) return; // soft-skip when rg missing
    writeFileSync(join(TMP, "note.txt"), "prefix UNIQUE_STRING_XYZ suffix\n");
    const r = await ripgrepSearch({
      rootPath: TMP,
      query: "UNIQUE_STRING_XYZ",
      maxHits: 10,
    });
    expect(r.ok).toBe(true);
    expect(r.hits).toBeGreaterThanOrEqual(1);
    expect(r.output).toContain("UNIQUE_STRING_XYZ");
    expect(r.output).toContain("note.txt");
  });

  test("returns no matches for missing query", async () => {
    if (!Bun.which("rg")) return;
    writeFileSync(join(TMP, "a.txt"), "hello\n");
    const r = await ripgrepSearch({
      rootPath: TMP,
      query: "this_string_does_not_exist_zzz",
      maxHits: 5,
    });
    expect(r.ok).toBe(true);
    expect(r.hits).toBe(0);
    expect(r.output).toContain("no matches");
  });
});
