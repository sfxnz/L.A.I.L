import { describe, expect, test } from "bun:test";
import { createApp } from "./app";
import { config } from "./config";

describe("LAIL_TOKEN middleware", () => {
  test("health stays open; other routes require the token when set", async () => {
    const prev = config.token;
    config.token = "secret";
    try {
      const app = createApp();
      const health = await app.request("/api/health");
      expect(health.status).toBe(200);
      const denied = await app.request("/api/bootstrap");
      expect(denied.status).toBe(401);
      const ok = await app.request("/api/bootstrap", {
        headers: { "x-lail-token": "secret" },
      });
      expect(ok.status).toBe(200);
    } finally {
      config.token = prev;
    }
  });

  test("query token is rejected on non-stream routes", async () => {
    const prev = config.token;
    config.token = "secret";
    try {
      const app = createApp();
      const viaQuery = await app.request("/api/bootstrap?token=secret");
      expect(viaQuery.status).toBe(401);
    } finally {
      config.token = prev;
    }
  });

  test("query token is accepted on job-log EventSource path", async () => {
    const prev = config.token;
    const origFetch = globalThis.fetch;
    config.token = "secret";
    globalThis.fetch = (async () => new Response("ok", { status: 200 })) as unknown as typeof fetch;
    try {
      const app = createApp();
      const denied = await app.request("/api/jobs/x/logs");
      expect(denied.status).toBe(401);
      const logs = await app.request("/api/jobs/x/logs?token=secret");
      expect(logs.status).toBe(200);
    } finally {
      config.token = prev;
      globalThis.fetch = origFetch;
    }
  });

  test("public share GETs stay unauthed when a token is set", async () => {
    const prev = config.token;
    config.token = "secret";
    try {
      const app = createApp();
      const share = await app.request("/api/lab/p/nope/index.html");
      expect(share.status).not.toBe(401);
      const short = await app.request("/p/nope");
      expect(short.status).not.toBe(401);
    } finally {
      config.token = prev;
    }
  });
});

describe("CORS origin callback", () => {
  test("does not reflect a foreign Origin when token is unset", async () => {
    const prev = config.token;
    config.token = "";
    try {
      const app = createApp();
      const res = await app.request("/api/health", {
        headers: { Origin: "https://evil.example" },
      });
      expect(res.headers.get("access-control-allow-origin")).not.toBe(
        "https://evil.example",
      );
    } finally {
      config.token = prev;
    }
  });

  test("preflight from a foreign origin is not approved", async () => {
    const prev = config.token;
    config.token = "";
    try {
      const app = createApp();
      const res = await app.request("/api/serve/start", {
        method: "OPTIONS",
        headers: {
          Origin: "https://evil.example",
          "Access-Control-Request-Method": "POST",
        },
      });
      expect(res.headers.get("access-control-allow-origin")).not.toBe(
        "https://evil.example",
      );
    } finally {
      config.token = prev;
    }
  });

  test("loopback Origin is allowed", async () => {
    const prev = config.token;
    config.token = "";
    try {
      const app = createApp();
      const res = await app.request("/api/health", {
        headers: { Origin: "http://127.0.0.1:3010" },
      });
      expect(res.headers.get("access-control-allow-origin")).toBe(
        "http://127.0.0.1:3010",
      );
    } finally {
      config.token = prev;
    }
  });
});

