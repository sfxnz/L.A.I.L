import { describe, expect, test, beforeEach, afterEach } from "bun:test";
import { mkdirSync, writeFileSync, rmSync } from "fs";
import { join } from "path";
import { buildContextPack } from "./packer";

const TMP = `/tmp/lail-packer-${process.pid}`;

describe("buildContextPack", () => {
  beforeEach(() => {
    rmSync(TMP, { recursive: true, force: true });
    mkdirSync(TMP, { recursive: true });
    writeFileSync(join(TMP, "a.txt"), "hello-unique\n");
  });
  afterEach(() => rmSync(TMP, { recursive: true, force: true }));

  test("includes mentioned and open file content from disk", async () => {
    const pack = await buildContextPack({
      rootPath: TMP,
      snapshot: {
        openFiles: [{ path: "a.txt" }],
        activePath: "a.txt",
        mentions: [{ type: "file", path: "a.txt" }],
      },
      budgetChars: 32_000,
    });
    expect(pack.contextMessage).not.toBeNull();
    expect(pack.contextMessage!.content).toContain("hello-unique");
    expect(pack.chunks.some((c) => c.kind === "mention_file")).toBe(true);
    expect(pack.systemExtra).toMatch(/Attached \d+ context chunk/);
  });

  test("merges parseMentions from message", async () => {
    writeFileSync(join(TMP, "b.txt"), "from-message-mention\n");
    const pack = await buildContextPack({
      rootPath: TMP,
      snapshot: { openFiles: [], mentions: [] },
      budgetChars: 32_000,
      message: "please read @file b.txt",
    });
    expect(pack.contextMessage!.content).toContain("from-message-mention");
  });

  test("selection is packed at high priority", async () => {
    const pack = await buildContextPack({
      rootPath: TMP,
      snapshot: {
        openFiles: [],
        mentions: [],
        selection: {
          path: "a.txt",
          startLine: 1,
          endLine: 1,
          text: "selected-line-content",
        },
      },
      budgetChars: 32_000,
    });
    expect(pack.chunks.some((c) => c.kind === "selection")).toBe(true);
    expect(pack.contextMessage!.content).toContain("selected-line-content");
  });

  test("missing mention becomes note chunk", async () => {
    const pack = await buildContextPack({
      rootPath: TMP,
      snapshot: {
        openFiles: [],
        mentions: [{ type: "file", path: "nope.txt" }],
      },
      budgetChars: 32_000,
    });
    expect(pack.chunks.some((c) => c.kind === "note")).toBe(true);
    expect(pack.contextMessage!.content).toMatch(/not found/i);
  });

  test("tiny budget truncates when many open tabs", async () => {
    for (let i = 0; i < 8; i++) {
      writeFileSync(join(TMP, `f${i}.txt`), ("x".repeat(200) + "\n").repeat(3));
    }
    const pack = await buildContextPack({
      rootPath: TMP,
      snapshot: {
        openFiles: Array.from({ length: 8 }, (_, i) => ({ path: `f${i}.txt` })),
        mentions: [],
      },
      budgetChars: 150,
    });
    expect(pack.truncated).toBe(true);
  });

  test("folder mention lists entries", async () => {
    mkdirSync(join(TMP, "sub"), { recursive: true });
    writeFileSync(join(TMP, "sub", "inner.txt"), "inner-body\n");
    const pack = await buildContextPack({
      rootPath: TMP,
      snapshot: {
        openFiles: [],
        mentions: [{ type: "folder", path: "sub" }],
      },
      budgetChars: 32_000,
    });
    expect(pack.chunks.some((c) => c.kind === "mention_folder")).toBe(true);
    expect(pack.contextMessage!.content).toContain("sub/inner.txt");
  });
});
