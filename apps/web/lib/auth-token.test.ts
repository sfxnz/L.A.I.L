import { describe, expect, test, beforeEach } from "bun:test";
import { getClientToken, setClientToken, tokenQuery } from "./auth-token";

const mem: Record<string, string> = {};
(globalThis as { sessionStorage: Storage }).sessionStorage = {
  getItem: (k: string) => (k in mem ? mem[k] : null),
  setItem: (k: string, v: string) => {
    mem[k] = String(v);
  },
  removeItem: (k: string) => {
    delete mem[k];
  },
  clear: () => {
    for (const k of Object.keys(mem)) delete mem[k];
  },
  key: () => null,
  get length() {
    return Object.keys(mem).length;
  },
} as Storage;

beforeEach(() => {
  for (const k of Object.keys(mem)) delete mem[k];
});

describe("client LAIL_TOKEN", () => {
  test("empty until set", () => {
    expect(getClientToken()).toBe("");
  });

  test("round-trips through sessionStorage", () => {
    setClientToken(" secret ");
    expect(getClientToken()).toBe("secret");
    setClientToken("");
    expect(getClientToken()).toBe("");
  });

  test("tokenQuery appends only when set", () => {
    expect(tokenQuery("/api/jobs/1/logs")).toBe("/api/jobs/1/logs");
    setClientToken("abc");
    expect(tokenQuery("/api/jobs/1/logs")).toBe("/api/jobs/1/logs?token=abc");
    expect(tokenQuery("/api/jobs/1/logs?x=1")).toBe("/api/jobs/1/logs?x=1&token=abc");
  });
});
