import { afterEach, describe, expect, test } from "bun:test";
import { createApp } from "../app";
import { config } from "../config";
import { proxyTimeoutMs } from "./serve-proxy";

const origFetch = globalThis.fetch;
const prevToken = config.token;
afterEach(() => {
  globalThis.fetch = origFetch;
  config.token = prevToken;
});

type Call = { url: string; init: RequestInit };
function fakeEngine(respond: (call: Call) => Response | Promise<Response>): Call[] {
  const calls: Call[] = [];
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    const call = { url: String(url), init };
    calls.push(call);
    return respond(call);
  }) as unknown as typeof fetch;
  return calls;
}

describe("serve-engine proxy", () => {
  test("forwards allowlisted paths with query and the engine token", async () => {
    config.token = "";
    const calls = fakeEngine(() => Response.json({ ok: true }));
    const app = createApp();
    for (const p of ["/api/status", "/api/cluster", "/api/jobs", "/api/jobs/j1/cancel", "/api/serve/recommend?model=a%2Fb", "/api/runs/tool-eval/board?limit=4", "/api/bench/perf"]) {
      expect((await app.request(p)).status).toBe(200);
    }
    expect(calls.map((c) => c.url.replace(config.serveEngineUrl, ""))).toEqual([
      "/api/status",
      "/api/cluster",
      "/api/jobs",
      "/api/jobs/j1/cancel",
      "/api/serve/recommend?model=a%2Fb",
      "/api/runs/tool-eval/board?limit=4",
      "/api/bench/perf",
    ]);
  });

  test("unused engine routes are not exposed", async () => {
    const calls = fakeEngine(() => Response.json({}));
    const app = createApp();
    for (const p of ["/api/hardware", "/api/chat", "/api/nope"]) {
      expect((await app.request(p, { method: p === "/api/chat" ? "POST" : "GET" })).status).toBe(404);
    }
    expect(calls).toEqual([]);
  });

  test("keeps upstream headers except hop-by-hop / encoding", async () => {
    fakeEngine(
      () =>
        new Response("{}", {
          headers: {
            "content-type": "application/json",
            "cache-control": "max-age=5",
            "content-disposition": 'attachment; filename="run.json"',
            "content-encoding": "gzip",
          },
        }),
    );
    const res = await createApp().request("/api/runs/r1");
    expect(res.headers.get("cache-control")).toBe("max-age=5");
    expect(res.headers.get("content-disposition")).toBe('attachment; filename="run.json"');
    expect(res.headers.get("content-encoding")).toBeNull();
  });

  test("lifts Bun.serve's 10 s idle cap for forwarded calls (the proxy bounds them itself)", async () => {
    fakeEngine(() => Response.json({ ok: true }));
    const lifted: Array<[Request, number]> = [];
    const server = { timeout: (req: Request, s: number) => lifted.push([req, s]) };
    const res = await createApp().request("/api/smoke", { method: "POST" }, server);
    expect(res.status).toBe(200);
    expect(lifted.length).toBe(1);
    expect(lifted[0][1]).toBe(0);
  });

  test("a wedged handler times out as 504; slow-by-design routes get a long budget", async () => {
    expect(proxyTimeoutMs("/status")).toBe(15_000);
    expect(proxyTimeoutMs("/serve/start")).toBe(15_000);
    expect(proxyTimeoutMs("/serve/recommend")).toBe(200_000);
    expect(proxyTimeoutMs("/smoke")).toBe(370_000); // > 2 x serve-engine's 180 s httpx timeout
    expect(proxyTimeoutMs("/cluster")).toBe(60_000); // > one 18 s ssh probe plus local probes
    const calls = fakeEngine(() => {
      throw new DOMException("The operation timed out.", "TimeoutError");
    });
    const res = await createApp().request("/api/status");
    expect(res.status).toBe(504);
    expect(((await res.json()) as { error: string }).error).toBe("serve_engine_timeout");
    expect(calls[0].init.signal).toBeInstanceOf(AbortSignal);
  });

  test("the browser's disconnect aborts the upstream call (SSE included)", async () => {
    const calls = fakeEngine(() => new Response("event: log\ndata: x\n\n", { headers: { "content-type": "text/event-stream" } }));
    const ac = new AbortController();
    const res = await createApp().request("/api/jobs/j1/logs", {
      headers: { accept: "text/event-stream" },
      signal: ac.signal,
    });
    expect(res.headers.get("content-encoding")).toBe("identity");
    expect(calls[0].init.signal!.aborted).toBe(false);
    ac.abort();
    expect(calls[0].init.signal!.aborted).toBe(true);
  });

  test("engine down → 502 JSON", async () => {
    fakeEngine(() => {
      throw new TypeError("fetch failed");
    });
    const res = await createApp().request("/api/status");
    expect(res.status).toBe(502);
    expect(((await res.json()) as { error: string }).error).toBe("serve_engine_unreachable");
  });
});
