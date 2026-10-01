import { openAiBase, listServedModelIds, resolveModelId } from "./settings";

const SERVED_TTL_MS = 5000;
const servedCache = new Map<string, { at: number; ids: string[] }>();

/** Served model ids per base URL, probed at most once per 5 s (not once per request). */
async function servedIds(base: string): Promise<string[]> {
  const hit = servedCache.get(base);
  if (hit && Date.now() - hit.at < SERVED_TTL_MS) return hit.ids;
  const ids = await listServedModelIds();
  servedCache.set(base, { at: Date.now(), ids });
  return ids;
}

export async function proxyOpenAI(req: Request, path: string): Promise<Response> {
  const targetBase = openAiBase();
  const url = `${targetBase}${path.startsWith("/") ? path : `/${path}`}${new URL(req.url).search}`;

  const headers = new Headers(req.headers);
  headers.delete("host");
  headers.set("content-type", "application/json");

  const init: RequestInit = {
    method: req.method,
    headers,
  };

  let bodyText: string | undefined;
  if (req.method !== "GET" && req.method !== "HEAD") {
    bodyText = await req.text();
    // Always bind chat/completions to the live served model (Server is source of truth).
    // A model that is really being served passes through untouched; placeholders and
    // stale ids are rewritten to the live id.
    if (path.includes("chat/completions") || path.includes("completions")) {
      try {
        const body = JSON.parse(bodyText) as { model?: string; [k: string]: unknown };
        const requested = (body.model || "").trim();
        const served = await servedIds(targetBase);
        if (!served.includes(requested)) {
          body.model = await resolveModelId(undefined, served);
          bodyText = JSON.stringify(body);
        }
      } catch {
        /* leave body as-is */
      }
    }
    init.body = bodyText;
  }

  const upstream = await fetch(url, init);
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
