import { createApp } from "./app";
import { assertSafeBind, tokenMatches } from "./bind";
import { config } from "./config";
import { getDb } from "./db/schema";
import { ensureDefaultWorkspace } from "./controller/workspaces";
import { wsHub } from "./ws/hub";
import { mkdirSync } from "fs";

assertSafeBind({
  host: config.host,
  token: config.token,
  allowInsecure: config.allowInsecureBind,
});

mkdirSync(config.dataDir, { recursive: true });
mkdirSync(config.workspacesDir, { recursive: true });
getDb();
ensureDefaultWorkspace();

const app = createApp();

type WsData = { id: string };

const server = Bun.serve<WsData>({
  hostname: config.host,
  port: config.port,
  fetch(req, server) {
    const url = new URL(req.url);
    if (url.pathname === "/ws") {
      if (config.token && !tokenMatches(req, config.token, { allowQuery: true })) {
        return new Response("unauthorized", { status: 401 });
      }
      const ok = server.upgrade(req, { data: { id: crypto.randomUUID() } });
      if (ok) return undefined as unknown as Response;
      return new Response("WebSocket upgrade failed", { status: 400 });
    }
    return app.fetch(req);
  },
  websocket: {
    open(ws) {
      const id = ws.data.id;
      wsHub.add(id, (data) => {
        try {
          ws.send(data);
        } catch {
          /* closed */
        }
      });
      ws.send(JSON.stringify({ channel: "system", event: { type: "hello", id } }));
    },
    message(ws, message) {
      const id = ws.data.id;
      try {
        const msg = JSON.parse(String(message)) as { subscribe?: string };
        if (msg.subscribe) wsHub.subscribe(id, msg.subscribe);
      } catch {
        /* ignore */
      }
    },
    close(ws) {
      const id = ws.data.id;
      wsHub.remove(id);
    },
  },
});

console.log(`L.A.I.L controller listening on http://${server.hostname}:${server.port}`);
console.log(`  WS   ws://${server.hostname}:${server.port}/ws`);
console.log(`  Serve-engine proxy → ${config.serveEngineUrl}`);
