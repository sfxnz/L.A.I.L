import { describe, expect, test } from "bun:test";
import {
  allowQueryToken,
  assertSafeBind,
  BindPolicyError,
  isLoopbackHost,
  isPublicUnauthedPath,
  resolveCorsOrigin,
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
    const req = (h: Record<string, string>, url = "http://127.0.0.1/api/health") =>
      new Request(url, { headers: h });
    expect(tokenMatches(req({ authorization: "Bearer abc" }), "abc")).toBe(true);
    expect(tokenMatches(req({ "x-lail-token": "abc" }), "abc")).toBe(true);
    expect(tokenMatches(req({ authorization: "Bearer no" }), "abc")).toBe(false);
    expect(tokenMatches(req({}), "abc")).toBe(false);
    expect(tokenMatches(req({}), "")).toBe(true);
  });

  test("query token is opt-in", () => {
    const req = new Request("http://127.0.0.1/api/serve/start?token=abc");
    expect(tokenMatches(req, "abc")).toBe(false);
    expect(tokenMatches(req, "abc", { allowQuery: true })).toBe(true);
  });
});

describe("allowQueryToken / public paths / CORS", () => {
  test("query token only on ws and job logs", () => {
    expect(allowQueryToken("/ws")).toBe(true);
    expect(allowQueryToken("/api/jobs/abc/logs")).toBe(true);
    expect(allowQueryToken("/api/serve/start")).toBe(false);
    expect(allowQueryToken("/api/bootstrap")).toBe(false);
  });

  test("public share GETs are unauthed", () => {
    expect(isPublicUnauthedPath("/api/lab/p/foo/index.html", "GET")).toBe(true);
    expect(isPublicUnauthedPath("/p/foo", "GET")).toBe(true);
    expect(isPublicUnauthedPath("/api/lab/public/foo", "GET")).toBe(true);
    expect(isPublicUnauthedPath("/api/lab/p/foo/index.html", "POST")).toBe(false);
    expect(isPublicUnauthedPath("/api/serve/start", "POST")).toBe(false);
  });

  test("never reflects a foreign origin", () => {
    const allow = ["http://127.0.0.1:3000"];
    expect(resolveCorsOrigin("https://evil.example", allow)).toBeUndefined();
    expect(resolveCorsOrigin("http://127.0.0.1:3000", allow)).toBe("http://127.0.0.1:3000");
    expect(resolveCorsOrigin("http://127.0.0.1:9999", allow)).toBe("http://127.0.0.1:9999");
  });
});
