import { describe, expect, test } from "bun:test";
import { parseMentions } from "./mentions";

describe("parseMentions", () => {
  test("parses @file path", () => {
    expect(parseMentions("see @file src/a.ts please")).toEqual([
      { type: "file", path: "src/a.ts" },
    ]);
  });
  test("parses @folder", () => {
    expect(parseMentions("@folder packages/backend")).toEqual([
      { type: "folder", path: "packages/backend" },
    ]);
  });
  test("parses @search and @code alias", () => {
    expect(parseMentions('@search "foo bar" and @code baz')).toEqual([
      { type: "search", query: "foo bar" },
      { type: "search", query: "baz" },
    ]);
  });
  test("dedupes identical mentions", () => {
    const m = parseMentions("@file a.ts @file a.ts");
    expect(m).toEqual([{ type: "file", path: "a.ts" }]);
  });
});
