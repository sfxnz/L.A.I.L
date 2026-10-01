import { afterEach, describe, expect, test } from "bun:test";
import { ApiError, api } from "./api";
import { isUnauthorizedError } from "./auth-token";

const origFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = origFetch;
});

function reply(status: number, body: string) {
  const seen: Array<{ url: string; init: RequestInit }> = [];
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    seen.push({ url, init });
    return new Response(body, { status });
  }) as unknown as typeof fetch;
  return seen;
}

describe("api req()", () => {
  test("a non-2xx reply throws ApiError carrying the HTTP status and parsed body", async () => {
    reply(401, '{"error":"unauthorized","message":"LAIL_TOKEN required"}');
    const err = await api.labStatus().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(401);
    expect(err.json).toEqual({ error: "unauthorized", message: "LAIL_TOKEN required" });
    expect(err.message).toContain("LAIL_TOKEN required");
    expect(isUnauthorizedError(err)).toBe(true);
  });

  test("an outage is not a token problem, even if its text says 401", async () => {
    reply(502, '{"error":"serve_engine_unreachable","message":"connect ECONNREFUSED 127.0.0.1:401"}');
    const err = await api.labStatus().catch((e) => e);
    expect(err.status).toBe(502);
    expect(isUnauthorizedError(err)).toBe(false);
  });

  test("a non-JSON error body still throws with status", async () => {
    reply(500, "Internal Server Error");
    const err = await api.labStatus().catch((e) => e);
    expect(err.status).toBe(500);
    expect(err.json).toBeNull();
  });

  test("method/body from the caller survive and JSON content type is always sent", async () => {
    const seen = reply(200, "{}");
    await api.configure.put({ defaultModel: "auto" });
    expect(seen[0].init.method).toBe("PUT");
    expect(seen[0].init.body).toBe('{"defaultModel":"auto"}');
    expect(new Headers(seen[0].init.headers).get("content-type")).toBe("application/json");
  });
});
