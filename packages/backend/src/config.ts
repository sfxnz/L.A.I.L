import { resolve, join } from "path";

const root = resolve(process.env.LAIL_ROOT || join(import.meta.dir, "../../.."));

export const config = {
  host: process.env.LAIL_HOST || "0.0.0.0",
  port: Number(process.env.LAIL_API_PORT || 8787),
  webPort: Number(process.env.LAIL_WEB_PORT || 3000),
  serveEngineUrl: process.env.LAIL_SERVE_ENGINE_URL || `http://127.0.0.1:${process.env.LAIL_SERVE_ENGINE_PORT || 8765}`,
  root,
  dataDir: resolve(process.env.LAIL_DATA_DIR || join(root, "data")),
  workspacesDir: resolve(process.env.LAIL_WORKSPACES_DIR || join(root, "workspaces")),
  dbPath: resolve(process.env.LAIL_DATA_DIR || join(root, "data"), "lail.sqlite"),
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
};
