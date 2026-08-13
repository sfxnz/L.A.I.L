import { describe, expect, test } from "bun:test";
import {
  assertSafeBind,
  BindPolicyError,
  isLoopbackHost,
  tokenMatches,
} from "./bind";

describe("bind policy", () => {
  test("treats loopback as local", () => {
    expect(isLoopbackHost("127.0.0.1")).toBe(true);
    expect(isLoopbackHost("localhost")).toBe(true);
    expect(isLoopbackHost("::1")).toBe(true);
    expect(isLoopbackHost("0.0.0.0")).toBe(false);
    expect(isLoopbackHost("10.0.0.5")).toBe(false);
  });

  test("allows empty token on loopback", () => {
    expect(() => assertSafeBind({ host: "127.0.0.1", token: "" })).not.toThrow();
  });

  test("requires token off-loopback", () => {
    expect(() => assertSafeBind({ host: "0.0.0.0", token: "" })).toThrow(BindPolicyError);
    expect(() => assertSafeBind({ host: "0.0.0.0", token: "secret" })).not.toThrow();
  });

  test("allowInsecure skips the token requirement", () => {
    expect(() =>
      assertSafeBind({ host: "0.0.0.0", token: "", allowInsecure: true }),
    ).not.toThrow();
  });
});

describe("tokenMatches", () => {
  test("accepts bearer and x-lail-token", () => {
    const req = (h: Record<string, string>) =>
      new Request("http://127.0.0.1/api/health", { headers: h });
    expect(tokenMatches(req({ authorization: "Bearer abc" }), "abc")).toBe(true);
    expect(tokenMatches(req({ "x-lail-token": "abc" }), "abc")).toBe(true);
    expect(tokenMatches(req({ authorization: "Bearer no" }), "abc")).toBe(false);
    expect(tokenMatches(req({}), "abc")).toBe(false);
    expect(tokenMatches(req({}), "")).toBe(true);
  });
});
