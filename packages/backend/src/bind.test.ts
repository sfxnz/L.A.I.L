import { describe, expect, test } from "bun:test";
import {
  allowQueryToken,
  assertSafeBind,
  BindPolicyError,
  isLoopbackHost,
  isCrossSiteWrite,
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
  test("query token only on job logs and stream-run events", () => {
    expect(allowQueryToken("/ws")).toBe(false); // WS hub retired
    expect(allowQueryToken("/api/jobs/abc/logs")).toBe(true);
    expect(allowQueryToken("/api/streams/runs/abc/events")).toBe(true);
    expect(allowQueryToken("/api/streams/runs/abc/stop")).toBe(false);
    expect(allowQueryToken("/api/streams/runs")).toBe(false);
    expect(allowQueryToken("/api/serve/start")).toBe(false);
    expect(allowQueryToken("/api/configure")).toBe(false);
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
    // Other loopback ports (dev apps, artifact servers) are not the lab UI.
    expect(resolveCorsOrigin("http://127.0.0.1:9999", allow)).toBeUndefined();
    expect(resolveCorsOrigin("http://localhost:8766", allow)).toBeUndefined();
  });

  test("cross-site writes must be preflighted JSON", () => {
    const allow = ["http://127.0.0.1:3000"];
    const r = (method: string, h: Record<string, string>) =>
      new Request("http://127.0.0.1:8787/api/serve/stop", { method, headers: h });
    // curl / Hermes: no Origin
    expect(isCrossSiteWrite(r("POST", {}), allow)).toBe(false);
    // the lab UI itself
    expect(isCrossSiteWrite(r("POST", { origin: "http://127.0.0.1:3000" }), allow)).toBe(false);
    expect(isCrossSiteWrite(r("POST", { origin: "http://spark1:3000", "sec-fetch-site": "same-origin" }), allow)).toBe(false);
    // UI proxied by Next under a LAN / Tailscale host: JSON, so it was preflighted or same-origin
    expect(isCrossSiteWrite(r("POST", { origin: "http://100.64.0.7:3000", "content-type": "application/json" }), allow)).toBe(false);
    // no-cors simple requests from anywhere else
    expect(isCrossSiteWrite(r("POST", { origin: "https://evil.example" }), allow)).toBe(true);
    expect(isCrossSiteWrite(r("POST", { origin: "https://evil.example", "content-type": "text/plain" }), allow)).toBe(true);
    expect(isCrossSiteWrite(r("POST", { origin: "http://127.0.0.1:8766", "content-type": "application/x-www-form-urlencoded" }), allow)).toBe(true);
    expect(isCrossSiteWrite(r("PUT", { origin: "null", "sec-fetch-site": "cross-site" }), allow)).toBe(true);
    // reads are never blocked here
    expect(isCrossSiteWrite(r("GET", { origin: "https://evil.example" }), allow)).toBe(false);
  });
});
