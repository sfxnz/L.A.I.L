import { describe, expect, test, beforeEach } from "bun:test";
import { ApiError } from "./api";
import { getClientToken, isUnauthorizedError, setClientToken, tokenQuery } from "./auth-token";

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

  test("isUnauthorizedError: the controller's 401, by status — never by message text", () => {
    const gate = { error: "unauthorized", message: "LAIL_TOKEN required" };
    expect(isUnauthorizedError(new ApiError(401, JSON.stringify(gate), gate))).toBe(true);
    // Text that merely mentions 401 / unauthorized / LAIL_TOKEN is a real failure.
    expect(isUnauthorizedError(new Error("401"))).toBe(false);
    expect(isUnauthorizedError(new ApiError(502, "run 20260401_401 failed: LAIL_TOKEN unauthorized", null))).toBe(false);
    // A 401 relayed from elsewhere (e.g. HF Hub through serve-engine) is not the token gate.
    expect(isUnauthorizedError(new ApiError(401, '{"detail":"Unauthorized"}', { message: "Unauthorized" }))).toBe(false);
    expect(isUnauthorizedError("serve-engine unreachable")).toBe(false);
    expect(isUnauthorizedError(null)).toBe(false);
  });
});
