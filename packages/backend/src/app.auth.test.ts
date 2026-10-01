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
      const denied = await app.request("/api/configure");
      expect(denied.status).toBe(401);
      const ok = await app.request("/api/configure", {
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
      const viaQuery = await app.request("/api/configure?token=secret");
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

  test("query token is accepted on the stream-run EventSource path", async () => {
    const prev = config.token;
    config.token = "secret";
    try {
      const app = createApp();
      const denied = await app.request("/api/streams/runs/x/events");
      expect(denied.status).toBe(401);
      // Unknown run → 404 from the route itself, i.e. the middleware let it through.
      const passed = await app.request("/api/streams/runs/x/events?token=secret");
      expect(passed.status).toBe(404);
      const stop = await app.request("/api/streams/runs/x/stop?token=secret", { method: "POST" });
      expect(stop.status).toBe(401);
    } finally {
      config.token = prev;
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

  test("only the configured web origin is allowed, not any loopback port", async () => {
    const prev = config.token;
    config.token = "";
    try {
      const app = createApp();
      const web = `http://127.0.0.1:${config.webPort}`;
      const ok = await app.request("/api/health", { headers: { Origin: web } });
      expect(ok.headers.get("access-control-allow-origin")).toBe(web);
      const other = await app.request("/api/health", { headers: { Origin: "http://127.0.0.1:8766" } });
      expect(other.headers.get("access-control-allow-origin")).not.toBe("http://127.0.0.1:8766");
    } finally {
      config.token = prev;
    }
  });
});

describe("cross-site write guard (no token)", () => {
  test("a no-cors POST from another site never reaches the handler", async () => {
    const prev = config.token;
    const origFetch = globalThis.fetch;
    const hits: string[] = [];
    config.token = "";
    globalThis.fetch = (async (u: string) => {
      hits.push(String(u));
      return Response.json({ job_id: "j" });
    }) as unknown as typeof fetch;
    try {
      const app = createApp();
      const evil = await app.request("/api/serve/stop", {
        method: "POST",
        headers: { Origin: "https://evil.example", "Content-Type": "text/plain" },
        body: "x",
      });
      expect(evil.status).toBe(403);
      expect(hits).toEqual([]);
      // curl / Hermes (no Origin) and the web UI (JSON) still work.
      const curl = await app.request("/api/serve/stop", { method: "POST" });
      expect(curl.status).toBe(200);
      const ui = await app.request("/api/serve/stop", {
        method: "POST",
        headers: { Origin: `http://127.0.0.1:${config.webPort}`, "Content-Type": "application/json" },
      });
      expect(ui.status).toBe(200);
      expect(hits.length).toBe(2);
    } finally {
      config.token = prev;
      globalThis.fetch = origFetch;
    }
  });
});

