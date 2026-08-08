import { Hono } from "hono";
import { config } from "../config";

export const serveProxy = new Hono();

async function forward(c: { req: { raw: Request; url: string; method: string; header: (n: string) => string | undefined } }, path: string) {
  const target = `${config.serveEngineUrl}${path}${new URL(c.req.url).search}`;
  const headers = new Headers();
  const ct = c.req.header("content-type");
  if (ct) headers.set("content-type", ct);
  const accept = c.req.header("accept");
  if (accept) headers.set("accept", accept);

  const init: RequestInit = { method: c.req.method, headers };
  if (c.req.method !== "GET" && c.req.method !== "HEAD") {
    init.body = await c.req.raw.arrayBuffer();
  }

  try {
    const res = await fetch(target, init);
    // SSE / stream passthrough
    const rct = res.headers.get("content-type") || "application/json";
    if (rct.includes("text/event-stream") || rct.includes("stream")) {
      return new Response(res.body, {
        status: res.status,
        headers: {
          "Content-Type": rct,
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
          // Never let an intermediary compress or buffer an event stream.
          // Next's dev proxy honours the browser's `Accept-Encoding: gzip` and
          // gzips this response; gzip buffers, so the browser holds an open
          // connection and receives ZERO bytes until the stream closes — the
          // job dock sits on "running / 0 log bytes" for the whole serve while
          // curl (which sends no Accept-Encoding by default) streams fine.
          // `identity` opts the stream out of compression; `X-Accel-Buffering`
          // does the same for nginx-style proxies in front of the lab.
          "Content-Encoding": "identity",
          "X-Accel-Buffering": "no",
        },
      });
    }
    const buf = await res.arrayBuffer();
    return new Response(buf, {
      status: res.status,
      headers: { "Content-Type": rct },
    });
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    return Response.json(
      {
        error: "serve_engine_unreachable",
        message,
        hint: `Start serve-engine on ${config.serveEngineUrl} (bun run dev starts it automatically).`,
      },
      { status: 502 },
    );
  }
}

const paths = [
  "/status",
  "/hardware",
  "/serve/start",
  "/serve/stop",
  "/serve/agent-restore",
  "/serve/recommend",
  "/serve/recipes",
  "/jobs",
  "/smoke",
  "/chat",
  "/bench/perf",
  "/bench/agentic",
  "/bench/tool-eval-status",
  "/runs",
];

// Explicit routes for Hono
serveProxy.all("/status", (c) => forward(c, "/api/status"));
serveProxy.all("/cluster", (c) => forward(c, "/api/cluster"));
serveProxy.all("/hardware", (c) => forward(c, "/api/hardware"));
serveProxy.all("/serve/*", (c) => {
  const u = new URL(c.req.url);
  const sub = u.pathname.replace(/^\/api/, "");
  return forward(c, `/api${sub}`);
});
serveProxy.all("/jobs", (c) => forward(c, "/api/jobs"));
serveProxy.all("/jobs/*", (c) => {
  const u = new URL(c.req.url);
  const sub = u.pathname.replace(/^\/api/, "");
  return forward(c, `/api${sub}`);
});
serveProxy.all("/smoke", (c) => forward(c, "/api/smoke"));
serveProxy.all("/chat", (c) => forward(c, "/api/chat"));
serveProxy.all("/bench/*", (c) => {
  const u = new URL(c.req.url);
  const sub = u.pathname.replace(/^\/api/, "");
  return forward(c, `/api${sub}`);
});
serveProxy.all("/runs", (c) => forward(c, "/api/runs"));
serveProxy.all("/runs/*", (c) => {
  const u = new URL(c.req.url);
  const sub = u.pathname.replace(/^\/api/, "");
  return forward(c, `/api${sub}`);
});

void paths;
