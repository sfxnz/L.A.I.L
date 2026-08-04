import { openAiBase, getSettings, resolveModelId } from "./settings";
import { recordUsage } from "./usage";

export async function proxyOpenAI(req: Request, path: string): Promise<Response> {
  const settings = getSettings();
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
    // Always bind chat/completions to the live served model (Server is source of truth)
    if (path.includes("chat/completions") || path.includes("completions")) {
      try {
        const body = JSON.parse(bodyText) as { model?: string; [k: string]: unknown };
        const live = await resolveModelId();
        const requested = (body.model || "").trim();
        const reqL = requested.toLowerCase();
        if (
          !requested ||
          reqL === "default" ||
          reqL === "auto" ||
          requested !== live
        ) {
          body.model = live;
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
    // Tee usage approximately on stream end is hard; meter after clone for non-stream only.
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "Content-Type": ct || "text/event-stream",
        "Cache-Control": "no-cache",
      },
    });
  }

  const text = await upstream.text();
  try {
    const j = JSON.parse(text) as {
      model?: string;
      usage?: { prompt_tokens?: number; completion_tokens?: number };
    };
    if (j.usage) {
      recordUsage({
        model: j.model || settings.defaultModel,
        prompt: j.usage.prompt_tokens || 0,
        completion: j.usage.completion_tokens || 0,
        source: "proxy",
      });
    }
  } catch {
    /* not json */
  }

  return new Response(text, {
    status: upstream.status,
    headers: { "Content-Type": ct || "application/json" },
  });
}
