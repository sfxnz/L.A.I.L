import { afterEach, describe, expect, test } from "bun:test";
import { config } from "../config";
import { getSettings, openAiBase } from "./settings";
import { proxyOpenAI } from "./llm-proxy";

const origFetch = globalThis.fetch;
const prevToken = config.token;
afterEach(() => {
  globalThis.fetch = origFetch;
  config.token = prevToken;
});

type Seen = { url: string; headers: Headers; body: string | undefined };

function fakeBackend(served: string[], status: Record<string, unknown> = {}): Seen[] {
  const seen: Seen[] = [];
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    if (String(url).startsWith(`${config.serveEngineUrl}/api/status`)) return Response.json(status);
    if (String(url).endsWith("/models")) return Response.json({ data: served.map((id) => ({ id })) });
    seen.push({ url: String(url), headers: new Headers(init?.headers), body: init?.body as string | undefined });
    return Response.json({ ok: true });
  }) as unknown as typeof fetch;
  return seen;
}

function chat(model: string, headers: Record<string, string> = {}) {
  return new Request("http://127.0.0.1:8787/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify({ model, messages: [] }),
  });
}

describe("/v1 proxy", () => {
  test("never forwards the operator token or cookies upstream", async () => {
    config.token = "op-secret";
    const seen = fakeBackend(["org/live"]);
    await proxyOpenAI(
      chat("org/live", { Authorization: "Bearer op-secret", "X-Lail-Token": "op-secret", Cookie: "a=b" }),
      "/chat/completions",
    );
    expect(seen[0].headers.get("authorization")).toBeNull();
    expect(seen[0].headers.get("x-lail-token")).toBeNull();
    expect(seen[0].headers.get("cookie")).toBeNull();
  });

  test("without a LAIL token, a client's own backend key passes through", async () => {
    config.token = "";
    const seen = fakeBackend(["org/live"]);
    await proxyOpenAI(chat("org/live", { Authorization: "Bearer vllm-key" }), "/chat/completions");
    expect(seen[0].headers.get("authorization")).toBe("Bearer vllm-key");
  });

  test("placeholder ids bind to the served model; real ids pass untouched", async () => {
    const seen = fakeBackend(["org/live"]);
    const before = getSettings().defaultModel;
    await proxyOpenAI(chat("auto"), "/chat/completions");
    await proxyOpenAI(chat("org/typo"), "/chat/completions");
    expect(JSON.parse(seen[0].body!).model).toBe("org/live");
    expect(JSON.parse(seen[1].body!).model).toBe("org/typo");
    expect(getSettings().defaultModel).toBe(before); // no settings write on the hot path
  });

  test("follows the endpoint the serve-engine detected, else the configured default", async () => {
    let seen = fakeBackend(["org/live"], { healthy: true, base_url: "http://127.0.0.1:30000" });
    await proxyOpenAI(chat("auto"), "/chat/completions");
    expect(seen[0].url).toBe("http://127.0.0.1:30000/v1/chat/completions");
    expect(JSON.parse(seen[0].body!).model).toBe("org/live");

    seen = fakeBackend(["org/live"], { healthy: false, base_url: "http://127.0.0.1:30000" });
    await proxyOpenAI(chat("org/live"), "/chat/completions");
    expect(seen[0].url).toBe(`${openAiBase()}/chat/completions`);
  });

  test("backend down → 502 JSON, not a bare 500", async () => {
    globalThis.fetch = (async () => {
      throw new Error("ECONNREFUSED");
    }) as unknown as typeof fetch;
    const res = await proxyOpenAI(chat("org/live"), "/chat/completions");
    expect(res.status).toBe(502);
    expect(((await res.json()) as { error: string }).error).toBe("backend_unreachable");
  });
});
