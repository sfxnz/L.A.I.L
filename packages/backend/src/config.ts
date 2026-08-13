import { resolve, join } from "path";

const root = resolve(process.env.LAIL_ROOT || join(import.meta.dir, "../../.."));

export const config = {
  host: process.env.LAIL_HOST || "127.0.0.1",
  port: Number(process.env.LAIL_API_PORT || 8787),
  webPort: Number(process.env.LAIL_WEB_PORT || 3000),
  token: (process.env.LAIL_TOKEN || "").trim(),
  allowInsecureBind: ["1", "true", "yes"].includes(
    (process.env.LAIL_INSECURE_BIND || "").trim().toLowerCase(),
  ),
  corsOrigins: (process.env.LAIL_CORS_ORIGINS || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean),
  serveEngineUrl: process.env.LAIL_SERVE_ENGINE_URL || `http://127.0.0.1:${process.env.LAIL_SERVE_ENGINE_PORT || 8765}`,
  root,
  dataDir: resolve(process.env.LAIL_DATA_DIR || join(root, "data")),
  workspacesDir: resolve(process.env.LAIL_WORKSPACES_DIR || join(root, "workspaces")),
  dbPath: resolve(
    process.env.LAIL_DB_PATH ||
      join(process.env.LAIL_DATA_DIR || join(root, "data"), "lail.sqlite"),
  ),
  defaultBackend: (process.env.LAIL_DEFAULT_BACKEND || "vllm") as "vllm" | "llamacpp",
  defaultModel: process.env.LAIL_DEFAULT_MODEL || "auto",
  backends: {
    vllm: {
      url: process.env.LAIL_VLLM_URL || "http://127.0.0.1:8000",
      enabled: true,
      label: "vLLM",
    },
    llamacpp: {
      url: process.env.LAIL_LLAMACPP_URL || "http://127.0.0.1:8080",
      enabled: true,
      label: "llama.cpp",
    },
  },
  hfToken: process.env.HF_TOKEN || "",
  /** Internet Funnel origin (legacy) — prefer shareSiteBase for X */
  sharePublicBase: (process.env.LAIL_SHARE_PUBLIC_BASE || "").replace(/\/$/, ""),
  /** GitHub Pages / static site origin, e.g. https://user.github.io/dgx-lab */
  shareSiteBase: (process.env.LAIL_SITE_BASE || "").replace(/\/$/, ""),
};
