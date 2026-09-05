import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import { listPacks } from "./packs";
import { StreamsError, StreamsEngine } from "./engine";

export function createStreamsRoutes(engine: StreamsEngine) {
  const r = new Hono();

  r.get("/packs", (c) => c.json(listPacks()));

  r.get("/runs", (c) => c.json(engine.list()));

  r.post("/runs", async (c) => {
    const body = await c.req.json().catch(() => null);
    try {
      const run_id = await engine.createRun(body);
      return c.json({ run_id }, 201);
    } catch (e) {
      if (e instanceof StreamsError) return c.json({ error: e.code, message: e.message, ...e.extra }, e.status);
      return c.json({ error: "error", message: e instanceof Error ? e.message : String(e) }, 500);
    }
  });

  r.get("/runs/:id", (c) => {
    const snap = engine.snapshot(c.req.param("id"));
    if (!snap) return c.json({ error: "not_found" }, 404);
    return c.json(snap);
  });

  r.post("/runs/:id/stop", (c) => {
    const id = c.req.param("id");
    if (!engine.stop(id)) return c.json({ error: "not_found" }, 404);
    return c.json({ ok: true, run_id: id });
  });

  // One SSE per subscriber: `hello` + snapshot, then live events until `done`.
  r.get("/runs/:id/events", (c) => {
    const id = c.req.param("id");
    const sub = engine.subscribe(id);
    if (!sub) return c.json({ error: "not_found" }, 404);
    // Same reasoning as serve-proxy: never let an intermediary compress or buffer the stream.
    c.header("Content-Encoding", "identity");
    c.header("X-Accel-Buffering", "no");
    return streamSSE(c, async (stream) => {
      stream.onAbort(() => engine.unsubscribe(id, sub));
      try {
        for (;;) {
          const ev = await sub.next();
          if (!ev) break;
          await stream.writeSSE({ event: ev.type, data: JSON.stringify(ev) });
          if (ev.type === "done") break;
        }
      } finally {
        engine.unsubscribe(id, sub);
      }
    });
  });

  return r;
}
