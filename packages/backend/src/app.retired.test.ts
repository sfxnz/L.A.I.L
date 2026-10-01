import { describe, expect, test } from "bun:test";
import { createApp } from "./app";
import { getDb } from "./db/schema";

describe("retired agent/workbench surface", () => {
  test("agent, workspace, patch and model-pull routes are gone", async () => {
    const app = createApp();
    for (const [method, path] of [
      ["GET", "/api/bootstrap"],
      ["GET", "/api/workspaces"],
      ["POST", "/api/workspaces"],
      ["PUT", "/api/workspaces/x/file"],
      ["GET", "/api/sessions"],
      ["POST", "/api/agent/run"],
      ["GET", "/api/patches"],
      ["GET", "/api/models"],
      ["POST", "/api/models/pull"],
    ] as const) {
      const res = await app.request(path, {
        method,
        headers: { "Content-Type": "application/json" },
        body: method === "GET" ? undefined : "{}",
      });
      expect(`${method} ${path} ${res.status}`).toBe(`${method} ${path} 404`);
    }
  });

  test("the retired tables are not created", () => {
    const names = (
      getDb().query("SELECT name FROM sqlite_master WHERE type = 'table'").all() as Array<{ name: string }>
    ).map((r) => r.name);
    for (const t of ["workspaces", "sessions", "messages", "agent_runs", "patches"]) {
      expect(names).not.toContain(t);
    }
  });

  test("configure carries no agent context budget and no HF token", async () => {
    const res = await createApp().request("/api/configure");
    const body = (await res.json()) as Record<string, unknown>;
    expect(Object.keys(body).sort()).toEqual(["backends", "defaultBackend", "defaultModel"]);
  });
});
