import { Hono } from "hono";
import { cors } from "hono/cors";
import { serveProxy } from "./routes/serve-proxy";
import { getSettings, putSettings } from "./controller/settings";
import { fetchServe, labStatusOf, liveHub, liveResponse, probeBackends } from "./live";
import { getUsageSummary } from "./controller/usage";
import { proxyOpenAI } from "./controller/llm-proxy";
import { streamsEngine } from "./streams/engine";
import { createStreamsRoutes } from "./streams/routes";
import { config } from "./config";
import { allowQueryToken, isCrossSiteWrite, isPublicUnauthedPath, isUntrustedHost, resolveCorsOrigin, tokenMatches } from "./bind";
import {
  artifactBody,
  compareLabRuns,
  getLabRun,
  getPublicBySlug,
  importLabRun,
  listLabFiles,
  listLabRuns,
  listLabRunsByFingerprint,
  publicPlayHeaders,
  playKeyMatches,
  publishLabRun,
  resolvePublicFile,
  resolveRunArtifact,
} from "./lab/store";

/** Bun.serve hands itself to app.fetch as the env (absent under app.request in tests). */
type BunServerEnv = { timeout?: (req: Request, seconds: number) => void } | undefined;

export function createApp() {
  const app = new Hono();
  const corsAllow = [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    `http://127.0.0.1:${config.webPort}`,
    `http://localhost:${config.webPort}`,
    ...config.corsOrigins,
  ];
  app.use(
    "*",
    cors({
      origin: (origin) => resolveCorsOrigin(origin, corsAllow),
      allowMethods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
      allowHeaders: ["Content-Type", "Authorization", "X-Lail-Token"],
    }),
  );

  // Without a token, a DNS-rebinding page would be same-origin with the lab: refuse
  // requests addressed to a host name the lab does not answer to.
  app.use("*", async (c, next) => {
    if (config.token || !isUntrustedHost(c.req.raw, corsAllow)) return next();
    return c.json({ error: "untrusted_host", message: "Host not allowed (set LAIL_CORS_ORIGINS or LAIL_TOKEN)" }, 403);
  });

  app.use("*", async (c, next) => {
    if (!isCrossSiteWrite(c.req.raw, corsAllow)) return next();
    return c.json(
      { error: "cross_site_write", message: "Cross-site write refused (send Content-Type: application/json)" },
      403,
    );
  });

  app.use("*", async (c, next) => {
    if (!config.token) return next();
    if (c.req.method === "OPTIONS") return next();
    const path = new URL(c.req.url).pathname;
    if (isPublicUnauthedPath(path, c.req.method)) return next();
    if (tokenMatches(c.req.raw, config.token, { allowQuery: allowQueryToken(path) })) {
      return next();
    }
    return c.json(
      { error: "unauthorized", message: "LAIL_TOKEN required (Authorization: Bearer or X-Lail-Token)" },
      401,
    );
  });

  app.get("/api/health", (c) =>
    c.json({ status: "ok", service: "lail-controller", version: "0.1.0" }),
  );

  app.get("/api/configure", (c) => c.json(getSettings()));
  app.put("/api/configure", async (c) => {
    const body = await c.req.json();
    return c.json(putSettings(body));
  });

  app.get("/api/usage", (c) => c.json(getUsageSummary()));

  // The live stream every dashboard tab subscribes to (see live.ts).
  app.get("/api/live", (c) => {
    // Bun.serve closes a request that sends nothing for 10 s; the stream pings
    // every 5 s, but a stalled upstream must not cut every tab at once.
    (c.env as BunServerEnv)?.timeout?.(c.req.raw, 0);
    return liveResponse(liveHub, c.req.raw.signal);
  });

  // One-shot status (curl, and the browser's fallback when the stream fails).
  app.get("/api/lab-status", async (c) => {
    const [serve, backends] = await Promise.all([fetchServe(), probeBackends()]);
    return c.json(labStatusOf(serve, backends));
  });

  // ── Lab gallery (Hermes task artifacts) ─────────────────────────
  app.get("/api/lab/runs", (c) => {
    const limit = Number(c.req.query("limit") || 50);
    const task = c.req.query("task_type") || "";
    const model = c.req.query("model") || "";
    const fp = c.req.query("fingerprint") || "";
    const rows = listLabRuns(
      Math.min(fp ? 50 : 200, Math.max(1, limit)),
      (m) =>
        (!fp || m.task_fingerprint === fp) &&
        (!task || m.task_type === task) &&
        (!model || (m.model_id || "").includes(model)),
    );
    return c.json({ runs: rows, count: rows.length });
  });

  app.get("/api/lab/compare", (c) => {
    const ids = (c.req.query("ids") || "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean)
      .slice(0, 4);
    if (ids.length < 2) return c.json({ error: "need_2_to_4_ids" }, 400);
    return c.json(compareLabRuns(ids));
  });

  app.get("/api/lab/runs/:id", (c) => {
    const run = getLabRun(c.req.param("id"));
    if (!run) return c.json({ error: "not_found" }, 404);
    const siblings = run.task_fingerprint
      ? listLabRunsByFingerprint(run.task_fingerprint, 12).filter((r) => r.id !== run.id)
      : [];
    return c.json({ ...run, files: listLabFiles(run.id), siblings });
  });

  app.post("/api/lab/runs/import", async (c) => {
    try {
      const body = await c.req.json();
      if (!body?.from || !body?.title) {
        return c.json({ error: "title_and_from_required" }, 400);
      }
      const run = importLabRun({
        title: String(body.title),
        from: String(body.from),
        task_type: body.task_type,
        model_id: body.model_id,
        entry: body.entry,
        tags: body.tags,
        brief: body.brief,
        eval_run_id: body.eval_run_id,
        hermes: body.hermes,
        serve: body.serve,
        share_public: !!body.share_public,
      });
      return c.json(run, 201);
    } catch (e) {
      const err = e as Error & { code?: string };
      const status =
        err.code === "not_found" ? 404 : err.code === "secret_detected" ? 422 : 400;
      return c.json({ error: err.code || "import_failed", message: err.message }, status);
    }
  });

  app.post("/api/lab/runs/:id/share", async (c) => {
    try {
      const body = await c.req.json().catch(() => ({ public: true }));
      const makePublic = body.public !== false;
      const run = publishLabRun(c.req.param("id"), makePublic);
      return c.json(run);
    } catch (e) {
      const err = e as Error & { code?: string };
      const status =
        err.code === "not_found" ? 404 : err.code === "secret_detected" ? 422 : 400;
      return c.json({ error: err.code || "share_failed", message: err.message }, status);
    }
  });

  // Private play: the path's key is the capability (no token: iframes cannot send
  // one). Same CSP as public shares — model-written HTML is untrusted. Files that
  // are not web assets (the run's sources) download instead of rendering.
  app.get("/api/lab/play/:id/:key/*", (c) => {
    const { id, key } = c.req.param();
    if (!playKeyMatches(id, key)) return c.json({ error: "not_found" }, 404);
    const rel = c.req.path.slice(`/api/lab/play/${id}/${key}/`.length);
    try {
      const { abs, contentType, playable } = resolveRunArtifact(id, rel);
      const headers: Record<string, string> = {
        ...publicPlayHeaders(contentType),
        "Cache-Control": "private, no-cache",
      };
      if (!playable) headers["Content-Disposition"] = "attachment";
      return new Response(artifactBody(abs, contentType), { headers });
    } catch (e) {
      const err = e as Error & { code?: string };
      const status = err.code === "forbidden_type" ? 403 : err.code === "bad_path" ? 400 : 404;
      return c.json({ error: err.code || "error", message: err.message }, status);
    }
  });

  // Public static share — artifacts only (under /api so Next rewrites always hit controller)
  app.get("/api/lab/public/:slug", (c) => {
    const pub = getPublicBySlug(c.req.param("slug"));
    if (!pub) return c.json({ error: "not_found" }, 404);
    return c.json({
      slug: pub.slug,
      ...pub.meta,
      play_url: `/api/lab/p/${pub.slug}/index.html`,
    });
  });

  // No slash redirects — Next.js 308-strips trailing slashes and loops with 302-add-slash.
  // Share URL is always .../index.html so relative game assets resolve under /p/<slug>/.
  app.get("/api/lab/p/:slug", (c) => servePublicIndex(c.req.param("slug")));
  app.get("/api/lab/p/:slug/", (c) => servePublicIndex(c.req.param("slug")));
  app.get("/api/lab/p/:slug/index.html", (c) => servePublicIndex(c.req.param("slug")));

  app.get("/api/lab/p/:slug/*", (c) => {
    const slug = c.req.param("slug");
    let rel = c.req.path.replace(`/api/lab/p/${slug}/`, "").replace(/^\/+/, "");
    if (!rel || rel === "index.html") return servePublicIndex(slug);
    try {
      const { abs, contentType } = resolvePublicFile(slug, rel);
      return new Response(artifactBody(abs, contentType), {
        headers: publicPlayHeaders(contentType),
      });
    } catch (e) {
      const err = e as Error & { code?: string };
      const status = err.code === "forbidden_type" ? 403 : 404;
      return c.json({ error: err.code || "error", message: err.message }, status);
    }
  });

  // Short aliases → stable index.html URL
  app.get("/p/:slug", (c) => c.redirect(`/api/lab/p/${c.req.param("slug")}/index.html`, 302));
  app.get("/p/:slug/", (c) => c.redirect(`/api/lab/p/${c.req.param("slug")}/index.html`, 302));
  app.get("/p/:slug/*", (c) => {
    const slug = c.req.param("slug");
    const rel = c.req.path.replace(`/p/${slug}/`, "");
    return c.redirect(`/api/lab/p/${slug}/${rel}`, 302);
  });

  // Streams engine: controller-side fan-out, one SSE per subscriber
  app.route("/api/streams", createStreamsRoutes(streamsEngine));

  // Legacy + serve proxy under /api
  app.route("/api", serveProxy);

  // OpenAI-compatible proxy
  app.all("/v1/*", (c) => {
    const path = new URL(c.req.url).pathname.replace(/^\/v1/, "") || "/";
    return proxyOpenAI(c.req.raw, path);
  });

  return app;
}

function servePublicIndex(slug: string): Response {
  try {
    const { abs, contentType } = resolvePublicFile(slug, "index.html");
    return new Response(artifactBody(abs, contentType), {
      headers: publicPlayHeaders(contentType),
    });
  } catch {
    return Response.json({ error: "not_found" }, { status: 404 });
  }
}
