import { describe, expect, test } from "bun:test";
import { join, resolve } from "path";
import { tmpdir } from "os";
import { config } from "./config";

describe("test isolation", () => {
  test("tests never point at the repo data dir", () => {
    const repoData = resolve(join(config.root, "data"));
    expect(config.dataDir.startsWith(repoData)).toBe(false);
    expect(config.dbPath.startsWith(repoData)).toBe(false);
    expect(config.dbPath.startsWith(resolve(tmpdir()))).toBe(true);
  });
});
