import { describe, expect, test } from "bun:test";
import { SseParser } from "./sse-parser";

describe("SseParser", () => {
  test("yields one payload per event and passes [DONE] through", () => {
    const p = new SseParser();
    expect(p.feed('data: {"a":1}\n\ndata: {"b":2}\n\ndata: [DONE]\n\n')).toEqual(['{"a":1}', '{"b":2}', "[DONE]"]);
  });

  test("reassembles partial chunks, including a \\r\\n split across reads", () => {
    const p = new SseParser();
    expect(p.feed('data: {"cho')).toEqual([]);
    expect(p.feed('ices":[]}\r')).toEqual([]);
    expect(p.feed("\n\r")).toEqual([]);
    expect(p.feed('\ndata: {"x":1}\n')).toEqual(['{"choices":[]}']);
    expect(p.feed("\n")).toEqual(['{"x":1}']);
  });

  test("joins multi-line data fields and ignores comments, event and id fields", () => {
    const p = new SseParser();
    const out = p.feed(": keep-alive\n\nevent: message\nid: 7\ndata: line one\ndata: line two\n\n");
    expect(out).toEqual(["line one\nline two"]);
  });

  test("strips exactly one leading space and keeps empty data lines", () => {
    const p = new SseParser();
    expect(p.feed("data:  two spaces\n\ndata:\n\n")).toEqual([" two spaces", ""]);
  });

  test("end() flushes a trailing event without a blank line", () => {
    const p = new SseParser();
    expect(p.feed("data: tail")).toEqual([]);
    expect(p.end()).toEqual(["tail"]);
    expect(p.end()).toEqual([]);
  });
});
