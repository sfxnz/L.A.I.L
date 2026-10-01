import { config } from "../config";
import { readServeStatus } from "../streams/probes";
import { isPlaceholderModel, listServedModelIds, openAiBase, resolveModelId } from "./settings";

const SERVED_TTL_MS = 5000;
const servedCache = new Map<string, { at: number; ids: string[] }>();

/** Served model ids per base URL, probed at most once per 5 s (not once per request). */
async function servedIds(base: string): Promise<string[]> {
  const hit = servedCache.get(base);
  if (hit && Date.now() - hit.at < SERVED_TTL_MS) return hit.ids;
  const ids = await listServedModelIds(undefined, base);
  servedCache.set(base, { at: Date.now(), ids });
  return ids;
}

/**
 * Where /v1 goes: the endpoint the serve-engine detected answering (SGLang on :30000,
 * llama.cpp on :8080, …), else the configured default backend. The serve-engine answers
 * from its sampler's cached snapshot, so this is one loopback read per request.
 */
async function targetBase(): Promise<string> {
  const serving = (await readServeStatus())?.serving_base;
  return serving ? `${serving}/v1` : openAiBase();
}

/**
 * Request headers that belong to this hop or to L.A.I.L itself. The operator's
 * LAIL_TOKEN (Authorization / X-Lail-Token) and browser cookies must never reach
 * the model backend. Without a LAIL_TOKEN the controller does not consume
 * Authorization, so a client's own backend key (vLLM --api-key) passes through.
 */
function upstreamHeaders(req: Request): Headers {
  const headers = new Headers(req.headers);
  for (const h of ["host", "connection", "content-length", "cookie", "x-lail-token"]) headers.delete(h);
  if (config.token) headers.delete("authorization");
  headers.set("content-type", "application/json");
  return headers;
}

export async function proxyOpenAI(req: Request, path: string): Promise<Response> {
  const base = await targetBase();
  const url = `${base}${path.startsWith("/") ? path : `/${path}`}${new URL(req.url).search}`;

  // The client's disconnect aborts the upstream request, so vLLM stops generating.
  const init: RequestInit = { method: req.method, headers: upstreamHeaders(req), signal: req.signal };

  let bodyText: string | undefined;
  if (req.method !== "GET" && req.method !== "HEAD") {
    bodyText = await req.text();
    // Only placeholder ids ("", auto, default) are bound to the live served model;
    // a real id, even a wrong one, goes through untouched so typos fail loudly.
    if (path.includes("completions")) {
      try {
        const body = JSON.parse(bodyText) as { model?: string; [k: string]: unknown };
        if (isPlaceholderModel(body.model)) {
          body.model = await resolveModelId(undefined, await servedIds(base));
          bodyText = JSON.stringify(body);
        }
      } catch {
        /* not JSON, or nothing served: leave the body for the backend to reject */
      }
    }
    init.body = bodyText;
  }

  let upstream: Response;
  try {
    upstream = await fetch(url, init);
  } catch (e) {
    return Response.json(
      {
        error: "backend_unreachable",
        message: e instanceof Error ? e.message : String(e),
        backend: base,
      },
      { status: 502 },
    );
  }
  const ct = upstream.headers.get("content-type") || "";

  // Stream pass-through
  if (ct.includes("text/event-stream") || bodyText?.includes('"stream":true')) {
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "Content-Type": ct || "text/event-stream",
        "Cache-Control": "no-cache",
      },
    });
  }

  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: { "Content-Type": ct || "application/json" },
  });
}
