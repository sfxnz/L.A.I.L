import { createApp } from "./app";
import { assertSafeBind } from "./bind";
import { config } from "./config";
import { getDb } from "./db/schema";
import { startUsageMeter } from "./controller/usage";
import { mkdirSync } from "fs";

assertSafeBind({
  host: config.host,
  token: config.token,
  allowInsecure: config.allowInsecureBind,
});

mkdirSync(config.dataDir, { recursive: true });
getDb();
startUsageMeter();

const app = createApp();

const server = Bun.serve({
  hostname: config.host,
  port: config.port,
  fetch: app.fetch,
});

console.log(`L.A.I.L controller listening on http://${server.hostname}:${server.port}`);
console.log(`  Serve-engine proxy → ${config.serveEngineUrl}`);
