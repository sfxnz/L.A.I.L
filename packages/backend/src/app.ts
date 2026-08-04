import { Hono } from "hono";
import { cors } from "hono/cors";
import { serveProxy } from "./routes/serve-proxy";
import { getSettings, putSettings, openAiBase } from "./controller/settings";
import {
  ensureDefaultWorkspace,
  listWorkspaces,
  createWorkspace,
  getWorkspace,
  updateWorkspace,
  buildTree,
} from "./controller/workspaces";
import {
  listSessions,
  createSession,
  getSession,
  updateSession,
  listMessages,
} from "./controller/sessions";
import { runAgent, cancelAgentRun } from "./controller/agent";
import {
  listPatches,
  acceptPatch,
  rejectPatch,
  acceptAllPatches,
} from "./controller/patches";
import { approvalHub } from "./agent/approvals";
import { getUsageSummary } from "./controller/usage";
import { searchHuggingFace, listLocalModels, pullModel, getPullJob } from "./controller/models";
import { proxyOpenAI } from "./controller/llm-proxy";
import { config } from "./config";
import {
  ensureDemoLabRuns,
  getLabRun,
  importLabRun,
  listLabFiles,
  listLabRuns,
  resolveLabFile,
} from "./lab/store";
import { readFileSync } from "fs";

export function createApp() {
  const app = new Hono();
  // Seed gallery once if empty (demo HTML)
  try {
    ensureDemoLabRuns();
  } catch {
    /* */
  }

  app.use(
    "*",
    cors({
      origin: "*",
      allowMethods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
      allowHeaders: ["Content-Type", "Authorization"],
    }),
  );

  app.get("/api/health", (c) =>
    c.json({ status: "ok", service: "lail-controller", version: "0.1.0" }),
  );

  // Bootstrap default workspace
  app.get("/api/bootstrap", (c) => {
    const ws = ensureDefaultWorkspace();
    return c.json({ workspace: ws, settings: getSettings() });
  });

  app.get("/api/configure", (c) => c.json(getSettings()));
  app.put("/api/configure", async (c) => {
    const body = await c.req.json();
    return c.json(putSettings(body));
  });

  app.get("/api/workspaces", (c) => {
    ensureDefaultWorkspace();
    return c.json(listWorkspaces());
  });
  app.post("/api/workspaces", async (c) => {
    const body = await c.req.json();
    return c.json(createWorkspace(body.name || "project", body.rootPath));
  });
  app.get("/api/workspaces/:id", (c) => {
    const ws = getWorkspace(c.req.param("id"));
    if (!ws) return c.json({ error: "not_found" }, 404);
    return c.json(ws);
  });
  app.patch("/api/workspaces/:id", async (c) => {
    try {
      const body = await c.req.json();
      const ws = updateWorkspace(c.req.param("id"), body);
      if (!ws) return c.json({ error: "not_found" }, 404);
      return c.json(ws);
    } catch (e) {
      const err = e as Error & { code?: string; recovery?: string };
      return c.json({ error: err.code || "error", message: err.message, recovery: err.recovery }, 400);
    }
  });
  app.get("/api/workspaces/:id/tree", (c) => {
    return c.json(buildTree(c.req.param("id")));
  });
  app.get("/api/workspaces/:id/file", async (c) => {
    const rel = c.req.query("path") || "";
    try {
      const { resolveInWorkspace } = await import("./controller/workspaces");
      const { readFileSync, statSync } = await import("fs");
      const abs = resolveInWorkspace(c.req.param("id"), rel);
      const st = statSync(abs);
      if (!st.isFile()) return c.json({ error: "not_a_file" }, 400);
      if (st.size > 2_000_000) return c.json({ error: "too_large", size: st.size }, 413);
      const content = readFileSync(abs, "utf8");
      return c.json({ path: rel, content, size: st.size });
    } catch (e) {
      const err = e as Error & { code?: string };
      return c.json({ error: err.code || "error", message: err.message }, 400);
    }
  });
  app.put("/api/workspaces/:id/file", async (c) => {
    const body = await c.req.json();
    const rel = String(body.path || "");
    try {
      const { resolveInWorkspace } = await import("./controller/workspaces");
      const { writeFileSync, mkdirSync } = await import("fs");
      const { dirname } = await import("path");
      const abs = resolveInWorkspace(c.req.param("id"), rel);
      mkdirSync(dirname(abs), { recursive: true });
      writeFileSync(abs, String(body.content ?? ""), "utf8");
      return c.json({ ok: true, path: rel });
    } catch (e) {
      const err = e as Error & { code?: string };
      return c.json({ error: err.code || "error", message: err.message }, 400);
    }
  });

  app.get("/api/sessions", (c) => c.json(listSessions()));
  app.post("/api/sessions", async (c) => {
    const body = await c.req.json().catch(() => ({}));
    const ws = ensureDefaultWorkspace();
    const session = createSession(body.title || "New session", body.workspaceId || ws.id);
    return c.json(session);
  });
  app.get("/api/sessions/:id", (c) => {
    const s = getSession(c.req.param("id"));
    if (!s) return c.json({ error: "not_found" }, 404);
    return c.json({ session: s, messages: listMessages(s.id) });
  });
  app.patch("/api/sessions/:id", async (c) => {
    const body = await c.req.json();
    const s = updateSession(c.req.param("id"), body);
    if (!s) return c.json({ error: "not_found" }, 404);
    return c.json(s);
  });

  app.post("/api/agent/run", async (c) => {
    try {
      const body = await c.req.json();
      const result = await runAgent({
        sessionId: body.sessionId,
        message: body.message,
        workspaceId: body.workspaceId,
        mode: body.mode, // plan|ask|agent, optional
        editorSnapshot: body.editorSnapshot,
      });
      return c.json(result);
    } catch (e) {
      const err = e as Error & { code?: string; recovery?: string };
      return c.json({ error: err.code || "error", message: err.message, recovery: err.recovery }, 400);
    }
  });

  app.post("/api/agent/runs/:runId/cancel", (c) => {
    const ok = cancelAgentRun(c.req.param("runId"));
    return c.json({ ok });
  });

  app.post("/api/agent/runs/:runId/shell-approvals/:approvalId", async (c) => {
    const body = await c.req.json().catch(() => ({}));
    const decision = body.decision === "allow" ? "allow" : "deny";
    const ok = approvalHub.decide(c.req.param("approvalId"), decision);
    return c.json({ ok });
  });

  app.get("/api/patches", (c) => {
    return c.json(
      listPatches({
        sessionId: c.req.query("sessionId") || undefined,
        runId: c.req.query("runId") || undefined,
        status: c.req.query("status") || undefined,
      }),
    );
  });

  app.post("/api/patches/:id/accept", (c) => {
    try {
      return c.json(acceptPatch(c.req.param("id")));
    } catch (e) {
      const err = e as Error & { code?: string };
      const status = err.code === "NOT_FOUND" ? 404 : 400;
      return c.json({ error: err.code || "error", message: err.message }, status);
    }
  });

  app.post("/api/patches/:id/reject", (c) => {
    try {
      return c.json(rejectPatch(c.req.param("id")));
    } catch (e) {
      const err = e as Error & { code?: string };
      const status = err.code === "NOT_FOUND" ? 404 : 400;
      return c.json({ error: err.code || "error", message: err.message }, status);
    }
  });

  app.post("/api/patches/accept-all", async (c) => {
    try {
      const body = await c.req.json().catch(() => ({}));
      return c.json(acceptAllPatches(body));
    } catch (e) {
      const err = e as Error & { code?: string };
      return c.json({ error: err.code || "error", message: err.message }, 400);
    }
  });

  app.get("/api/usage", (c) => c.json(getUsageSummary()));

  app.get("/api/models", async (c) => {
    const local = await listLocalModels();
    return c.json({ local });
  });
  app.get("/api/models/search", async (c) => {
    const q = c.req.query("q") || "gguf";
    try {
      const results = await searchHuggingFace(q);
      return c.json({ results });
    } catch (e) {
      return c.json({ results: [], error: e instanceof Error ? e.message : String(e) }, 502);
    }
  });
  app.post("/api/models/pull", async (c) => {
    const body = await c.req.json();
    const result = await pullModel(body.model, body.backend || "hf");
    return c.json(result);
  });
  app.get("/api/models/pull/:jobId", (c) => {
    const job = getPullJob(c.req.param("jobId"));
    if (!job) return c.json({ error: "not_found" }, 404);
    return c.json(job);
  });

  // Merged lab status: controller + serve-engine + backends
  app.get("/api/lab-status", async (c) => {
    const settings = getSettings();
    const backends: Record<string, { ok: boolean; url: string; error?: string }> = {};
    for (const [k, v] of Object.entries(settings.backends)) {
      if (!v.enabled) continue;
      try {
        const base = v.url.replace(/\/$/, "").replace(/\/v1$/, "");
        const r = await fetch(`${base}/v1/models`, {
          signal: AbortSignal.timeout(2000),
        });
        backends[k] = { ok: r.ok, url: v.url };
      } catch (e) {
        backends[k] = {
          ok: false,
          url: v.url,
          error: e instanceof Error ? e.message : String(e),
        };
      }
    }

    let serve: unknown = null;
    try {
      const r = await fetch(`${config.serveEngineUrl}/api/status`, {
        signal: AbortSignal.timeout(3000),
      });
      if (r.ok) serve = await r.json();
      else serve = { error: `serve-engine ${r.status}` };
    } catch (e) {
      serve = { error: e instanceof Error ? e.message : String(e), unreachable: true };
    }

    return c.json({
      controller: "ok",
      defaultBackend: settings.defaultBackend,
      defaultModel: settings.defaultModel,
      openAiBase: openAiBase(),
      backends,
      serve,
    });
  });

  // ── Lab gallery (Hermes task artifacts) ─────────────────────────
  app.get("/api/lab/runs", (c) => {
    const limit = Number(c.req.query("limit") || 50);
    const task = c.req.query("task_type") || "";
    const model = c.req.query("model") || "";
    let rows = listLabRuns(Math.min(200, Math.max(1, limit)));
    if (task) rows = rows.filter((r) => r.task_type === task);
    if (model) rows = rows.filter((r) => (r.model_id || "").includes(model));
    return c.json({ runs: rows, count: rows.length });
  });

  app.get("/api/lab/runs/:id", (c) => {
    const run = getLabRun(c.req.param("id"));
    if (!run) return c.json({ error: "not_found" }, 404);
    return c.json({ ...run, files: listLabFiles(run.id) });
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
      const status = err.code === "not_found" ? 404 : 400;
      return c.json({ error: err.code || "import_failed", message: err.message }, status);
    }
  });

  app.get("/api/lab/runs/:id/play", (c) => {
    const id = c.req.param("id");
    const run = getLabRun(id);
    if (!run) return c.json({ error: "not_found" }, 404);
    const entry = run.entry || "index.html";
    return c.redirect(`/api/lab/runs/${id}/files/artifacts/${entry}`, 302);
  });

  app.get("/api/lab/runs/:id/files/*", (c) => {
    const id = c.req.param("id");
    const rel = c.req.path.replace(`/api/lab/runs/${id}/files/`, "");
    try {
      const { abs, contentType } = resolveLabFile(id, rel);
      const data = readFileSync(abs);
      return new Response(data, {
        headers: {
          "Content-Type": contentType,
          "Cache-Control": "no-cache",
          // allow iframe embed from L.A.I.L web origin
          "X-Frame-Options": "SAMEORIGIN",
        },
      });
    } catch (e) {
      const err = e as Error & { code?: string };
      const status = err.code === "not_found" ? 404 : 400;
      return c.json({ error: err.code || "error", message: err.message }, status);
    }
  });

  // Legacy + serve proxy under /api
  app.route("/api", serveProxy);

  // OpenAI-compatible proxy
  app.all("/v1/*", (c) => {
    const path = new URL(c.req.url).pathname.replace(/^\/v1/, "") || "/";
    return proxyOpenAI(c.req.raw, path);
  });

  return app;
}
