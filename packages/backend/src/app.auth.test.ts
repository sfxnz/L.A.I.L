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
      const viaQuery = await app.request("/api/bootstrap?token=secret");
      expect(viaQuery.status).toBe(200);
    } finally {
      config.token = prev;
    }
  });
});

