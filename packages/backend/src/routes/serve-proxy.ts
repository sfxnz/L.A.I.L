import { Hono } from "hono";
import { config } from "../config";

export const serveProxy = new Hono();

/** serve-engine routes reachable through the controller (paths below /api). */
const FORWARDED = /^\/(status|cluster|smoke|jobs(\/.+)?|serve\/.+|bench\/.+|runs(\/.+)?)$/;

/**
 * A wedged serve-engine handler must not hang the dashboard. Most calls answer in
 * milliseconds (jobs run in the background). The synchronous ones get a budget
 * above their own upstream timeouts, so the proxy never 504s a call that is
 * still going to succeed:
 * - smoke: two sequential httpx calls at 180 s each (/v1/models, then a completion)
 * - recommend: Hub card / vendor recipe fetches
 * - cluster: local probes plus an ssh probe per remote node (up to 18 s each)
 */
const SLOW_MS: Record<string, number> = {
  "/smoke": 370_000,
  "/serve/recommend": 200_000,
  "/cluster": 60_000,
};

export function proxyTimeoutMs(path: string): number {
  return SLOW_MS[path] ?? 15_000;
}

/** Bun.serve hands itself to app.fetch as the env (absent under app.request in tests). */
type BunServerEnv = { timeout?: (req: Request, seconds: number) => void } | undefined;

/** Hop-by-hop or re-encoded by fetch (bodies arrive decoded), so never copied back. */
const DROP_RESPONSE = ["connection", "keep-alive", "transfer-encoding", "content-encoding", "content-length"];

serveProxy.all("/*", async (c) => {
  const path = c.req.path.replace(/^\/api/, "");
  if (!FORWARDED.test(path)) return c.notFound();
  const target = `${config.serveEngineUrl}/api${path}${new URL(c.req.url).search}`;

  const headers = new Headers();
  const ct = c.req.header("content-type");
  if (ct) headers.set("content-type", ct);
  const accept = c.req.header("accept");
  if (accept) headers.set("accept", accept);
  if (config.token) headers.set("x-lail-token", config.token);

  // Bun.serve closes a request that has sent nothing for 10 s (its idleTimeout),
  // which would cut smoke/recommend/cluster and quiet log streams long before the
  // bounds below. This proxy bounds every call itself, so lift Bun's cap.
  (c.env as BunServerEnv)?.timeout?.(c.req.raw, 0);

  // SSE lives as long as the browser keeps it open; everything else is bounded.
  // Either way the browser's disconnect is propagated upstream.
  const sse = (accept || "").includes("text/event-stream") || /^\/jobs\/[^/]+\/logs$/.test(path);
  const signal = sse
    ? c.req.raw.signal
    : AbortSignal.any([c.req.raw.signal, AbortSignal.timeout(proxyTimeoutMs(path))]);

  const init: RequestInit = { method: c.req.method, headers, signal };
  if (c.req.method !== "GET" && c.req.method !== "HEAD") {
    init.body = await c.req.raw.arrayBuffer();
  }

  try {
    const res = await fetch(target, init);
    const out = new Headers(res.headers);
    for (const h of DROP_RESPONSE) out.delete(h);
    const rct = res.headers.get("content-type") || "application/json";
    out.set("content-type", rct);
    if (rct.includes("text/event-stream")) {
      out.set("cache-control", "no-cache");
      // Never let an intermediary compress or buffer an event stream.
      // Next's dev proxy honours the browser's `Accept-Encoding: gzip` and
      // gzips this response; gzip buffers, so the browser holds an open
      // connection and receives ZERO bytes until the stream closes — the
      // job dock sits on "running / 0 log bytes" for the whole serve while
      // curl (which sends no Accept-Encoding by default) streams fine.
      // `identity` opts the stream out of compression; `X-Accel-Buffering`
      // does the same for nginx-style proxies in front of the lab.
      out.set("content-encoding", "identity");
      out.set("x-accel-buffering", "no");
      return new Response(res.body, { status: res.status, headers: out });
    }
    return new Response(await res.arrayBuffer(), { status: res.status, headers: out });
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    if (e instanceof DOMException && e.name === "TimeoutError") {
      return Response.json(
        { error: "serve_engine_timeout", message: `serve-engine did not answer ${path} in ${proxyTimeoutMs(path) / 1000} s` },
        { status: 504 },
      );
    }
    return Response.json(
      {
        error: "serve_engine_unreachable",
        message,
        hint: `Start serve-engine on ${config.serveEngineUrl} (bun run dev starts it automatically).`,
      },
      { status: 502 },
    );
  }
});
