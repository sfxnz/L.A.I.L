import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { createApp } from "../app";
import { config } from "../config";
import { artifactBody, importLabRun, playKey, publishLabRun } from "./store";

const prevToken = config.token;
const src = mkdtempSync(join(tmpdir(), "lail-play-src-"));
let run: ReturnType<typeof importLabRun>;
let other: ReturnType<typeof importLabRun>;

beforeAll(() => {
  config.token = "op-secret";
  mkdirSync(join(src, "game"), { recursive: true });
  writeFileSync(join(src, "game", "index.html"), '<script src="js/main.js"></script>');
  mkdirSync(join(src, "game", "js"));
  writeFileSync(join(src, "game", "js", "main.js"), "console.log(1)");
  writeFileSync(join(src, "game", "js", "level.mjs"), "export const level = 1;");
  writeFileSync(join(src, "game", "level.json"), '{"level":1}');
  writeFileSync(join(src, "game", "run.sh"), "rm -rf /");
  run = importLabRun({ title: "game", from: join(src, "game") });
  other = importLabRun({ title: "other", from: join(src, "game", "index.html") });
});

afterAll(() => {
  config.token = prevToken;
  rmSync(src, { recursive: true, force: true });
});

describe("private lab play (capability URL)", () => {
  test("play_url and artifacts_url carry the run's key, never the token", () => {
    expect(run.artifacts_url).toBe(`/api/lab/play/${run.id}/${playKey(run.id)}/`);
    expect(run.play_url).toBe(`${run.artifacts_url}index.html`);
    expect(run.play_url).not.toContain("op-secret");
  });

  test("an iframe (no token) loads the entry and its relative assets, under the untrusted-content CSP", async () => {
    const app = createApp();
    const page = await app.request(run.play_url);
    expect(page.status).toBe(200);
    const csp = page.headers.get("content-security-policy") || "";
    expect(csp).toContain("default-src 'none'");
    expect(csp).toContain("sandbox allow-scripts");
    expect(csp).not.toContain("allow-same-origin");
    expect(page.headers.get("cache-control")).toBe("private, no-cache");
    const asset = await app.request(`${run.artifacts_url}js/main.js`);
    expect(asset.status).toBe(200);
    expect(await asset.text()).toBe("console.log(1)");
  });

  test("HTML gets an in-memory storage shim before its own scripts; other files are untouched", async () => {
    const app = createApp();
    const html = await (await app.request(run.play_url)).text();
    expect(html.indexOf("localStorage")).toBeGreaterThanOrEqual(0);
    expect(html.indexOf("defineProperty")).toBeLessThan(html.indexOf('<script src="js/main.js">'));
    expect(await (await app.request(`${run.artifacts_url}js/main.js`)).text()).toBe("console.log(1)");
  });

  test("the shim goes after a doctype, never before it (no quirks mode)", () => {
    const f = join(src, "doc.html");
    writeFileSync(f, "<!DOCTYPE html>\n<html><body><script>localStorage.getItem('best')</script></body></html>");
    const out = String(artifactBody(f, "text/html; charset=utf-8"));
    expect(out.startsWith("<!DOCTYPE html><script>")).toBe(true);
    expect(String(artifactBody(f, "text/plain"))).not.toContain("defineProperty");
  });

  test("wrong, foreign or missing keys and non-artifact files are refused", async () => {
    const app = createApp();
    expect((await app.request(`/api/lab/play/${run.id}/${"0".repeat(32)}/index.html`)).status).toBe(404);
    expect((await app.request(`/api/lab/play/${run.id}/${playKey(other.id)}/index.html`)).status).toBe(404);
    expect((await app.request(`${run.artifacts_url}..%2Fmeta.json`)).status).not.toBe(200);
    // The old token-only /files route is gone; with a token set it is simply unauthorized.
    expect((await app.request(`/api/lab/runs/${run.id}/files/artifacts/index.html`)).status).toBe(401);
  });

  test("rotating LAIL_TOKEN revokes old play links", async () => {
    const url = run.play_url;
    config.token = "rotated";
    try {
      expect((await createApp().request(url)).status).toBe(404);
    } finally {
      config.token = "op-secret";
    }
  });

  test("the sandboxed (Origin: null) page may load its module scripts and data", async () => {
    const app = createApp();
    for (const [f, type] of [
      ["js/level.mjs", "text/javascript; charset=utf-8"],
      ["level.json", "application/json"],
    ]) {
      const res = await app.request(`${run.artifacts_url}${f}`, { headers: { origin: "null" } });
      expect(res.status).toBe(200);
      expect(res.headers.get("access-control-allow-origin")).toBe("*");
      expect(res.headers.get("access-control-allow-credentials")).toBeNull();
      expect(res.headers.get("content-type")).toBe(type);
    }
  });

  test("non-web files (the run's sources) download as opaque bytes instead of rendering", async () => {
    const res = await createApp().request(`${run.artifacts_url}run.sh`);
    expect(res.status).toBe(200);
    expect(res.headers.get("content-disposition")).toBe("attachment");
    expect(res.headers.get("content-type")).toBe("application/octet-stream");
    expect(await res.text()).toBe("rm -rf /");
  });

  test("the retired /runs/:id/play alias is gone", async () => {
    const res = await createApp().request(`/api/lab/runs/${run.id}/play`, {
      headers: { "x-lail-token": "op-secret" },
    });
    expect(res.status).toBe(404);
  });
});

describe("public lab share", () => {
  test("same CORS allowance for the opaque origin; non-web files stay private", async () => {
    const shared = publishLabRun(run.id, true);
    const base = `/api/lab/p/${shared.share!.slug}/`;
    const app = createApp();
    const mod = await app.request(`${base}js/level.mjs`, { headers: { origin: "null" } });
    expect(mod.status).toBe(200);
    expect(mod.headers.get("access-control-allow-origin")).toBe("*");
    expect((await app.request(`${base}run.sh`)).status).toBe(403);
  });
});
