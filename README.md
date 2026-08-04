# L.A.I.L — Local AI Lab

**Serve & eval console for DGX Spark** — clean white UI to start/stop vLLM, run smoke/perf evals, and copy OpenAI-compatible endpoints for **Hermes** (and other clients).

Agentic coding/chat is **not** the primary surface anymore: wire Hermes to the live `:8000` endpoint after Serve.

| Layer | Stack |
|-------|--------|
| UI | Next.js 16 App Router + React 19 · light console chrome · Tailwind |
| Controller | Bun + Hono · lab-status, models, configure, proxy to serve-engine |
| Serve-engine | Python FastAPI · vLLM Lab Safe / Workflow Max, smoke, benches, run history |
| Models | **vLLM** and **llama.cpp** (no Ollama) |

## Product surface

Top nav:

| Page | Path | Role |
|------|------|------|
| **Status** | `/status` | Health, headroom, containers, recent runs |
| **Serve** | `/server` | Manual flags, auto-config, start/stop, job logs |
| **Evals** | `/evals` | Smoke + perf jobs + run history |
| **Connect** | `/connect` | Hermes / OpenAI base URL snippets |
| **Models** | `/models` | HF search / download |
| **Configure** | `/configure` | Default backend / model |

`/` redirects to **Status**. `/workbench` shows a retirement notice → Hermes.

## Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│  apps/web  Next.js 16 + React 19                             │
│  White console · Status · Serve · Evals · Connect            │
└────────────────────────────┬─────────────────────────────────┘
                             │ REST :8787
┌────────────────────────────▼─────────────────────────────────┐
│  LabController (Bun + Hono)                                  │
│  lab-status · models · configure · serve/* · bench/* proxy   │
└──────────────┬─────────────────────────────┬─────────────────┘
               │                             │
               ▼                             ▼
     vLLM / llama.cpp                  packages/serve-engine
     (OpenAI /v1)                      (Python: docker serve,
                                        smoke, perf, history)
               │
               └──────────▶ Hermes / Mac clients
```

### Controller pattern

One **LabController** is the public API. The Python **serve-engine** keeps the Spark/vLLM serve path (Lab Safe util ≤ 0.4, Workflow Max, stop, agent-restore, benches, envelopes). Composer agent remains in the backend for now but is **out of the primary UI**.

### Model resolution

Configure default model, or use `auto`. If the name is `default` / `auto` / empty, the controller probes `/v1/models` and uses the first served id. Prefer setting the real id after serve (e.g. `laguna`).

## Quick start

### Prerequisites

- [Bun](https://bun.sh) ≥ 1.1  
- Python 3.11+ (serve-engine)  
- Optional: Docker + NVIDIA for vLLM; llama.cpp server for GGUF  

```bash
# Install Bun if needed
curl -fsSL https://bun.sh/install | bash
source ~/.bashrc   # or reopen the shell

cd ~/projects/ai-lab/local-ai-lab

python3 -m venv .venv
source .venv/bin/activate
pip install fastapi 'uvicorn[standard]' httpx pydantic aiosqlite python-multipart sse-starlette

cp .env.example .env
# Edit LAIL_DEFAULT_MODEL to a real served id when vLLM is up
bun install
bun run dev
```

| Service | URL |
|---------|-----|
| Web (Workbench) | http://127.0.0.1:3000 |
| Controller | http://127.0.0.1:8787 |
| Serve-engine | http://127.0.0.1:8765 |
| WebSocket | ws://127.0.0.1:8787/ws |

Default route `/` redirects to **`/workbench`**.

### From a Mac (SSH tunnel)

L.A.I.L binds on the lab host (e.g. spark1). On the Mac:

```bash
ssh -L 3000:127.0.0.1:3000 -L 8787:127.0.0.1:8787 sfxnz@spark1
```

Then open http://127.0.0.1:3000 on the Mac.

### Linux + NVIDIA / DGX Spark

1. Drivers + Docker GPU for vLLM.  
2. **Server → Lab Safe** for comparable benches (util ≤ 0.4).  
3. **Workflow Max** for agent / long context (util ~0.7–0.85; keep ≳15–20 GiB free).  

| Intent | When | Defaults |
|--------|------|----------|
| `lab_safe` | Published A/B | util ≤ **0.4** |
| `workflow_max` | Real agent / long ctx | util **0.7–0.85** |
| `attach` | Already-served model | Probe only |

### Apple Silicon

1. Prefer **llama.cpp** on `:8080`.  
2. **Configure** → backend **llama.cpp**.  
3. Use **Server** when talking to a remote NVIDIA/Spark vLLM endpoint.  

## Workbench (agentic IDE)

**Phase A** delivers a Cursor-style agent platform (modes, review-first patches, cancel, shell approval). **Phase B** adds a **context engine**: open tabs + `@` mentions packed into each agent run under a configurable character budget. Specs: [Phase A design](./docs/superpowers/specs/2026-07-16-lail-cursor-ide-design.md) · [Phase B design](./docs/superpowers/specs/2026-07-16-lail-phase-b-context-engine-design.md). Later phases (Monaco, terminal panel, etc.) are still planned.

1. Open **Workbench** (or `/`).  
2. Ensure vLLM or llama.cpp is healthy (**Status** / **Server**).  
3. Set **Configure → Default model** to the served model id (or leave auto).  
4. Pick a mode, then prompt the agent:

| Mode | Behavior |
|------|----------|
| **Plan** | Design / outline only — no file edits or patches |
| **Ask** | Read-only tools (explore, answer) — no writes |
| **Agent** | Full tools; edits land as **pending patches** until you Accept |

5. **Patches**: Accept / Reject per change (or Accept all). Disk and editor tabs update only after Accept.  
6. **Cancel** stops an in-flight run; **risky shell** commands pause for Allow / Deny.  
7. Use **Save** after manual edits in editor tabs.  
8. **Context (Phase B)**: open editor tabs are auto-included; type `@` in Composer for a path popup (`@file path`), or use `@folder`, `@search "query"`, `@code`. Context chips show mentions + open tab count. Adjust **Configure → Context budget (chars)** (default 32000).

**Composer stream labels** (inspo-style):

| UI | Meaning |
|----|---------|
| **Thought** | Model reasoning step |
| **Ran N command(s)** | Completed tool calls (counts **tool_end** only) |
| **Creating path** | File write / patch proposal; may open editor tab |
| **Working** | In-progress turn |

Bottom placeholder: **Ask for follow-up changes**.

## Server (vLLM serve & evals)

Full parity with the previous lab GUI:

- Lab Safe / Workflow Max start · stop · agent-restore  
- HF auto-configure · live job logs  
- Smoke · perf · golden tools · run history envelopes  

Benches hit whatever is on the vLLM base URL (default `:8000`) — **serve first**, then bench.

## Models & Usage

- **Models**: HF search, local list from vLLM/llama.cpp `/v1/models`, HF download jobs (not Ollama pull).  
- **Usage**: lifetime tokens, heatmap, mix, top models (metered from proxy + agent).  

## Environment

See [`.env.example`](./.env.example).

| Variable | Role |
|----------|------|
| `LAIL_DEFAULT_BACKEND` | `vllm` or `llamacpp` |
| `LAIL_VLLM_URL` | Default `http://127.0.0.1:8000` |
| `LAIL_LLAMACPP_URL` | Default `http://127.0.0.1:8080` |
| `LAIL_DEFAULT_MODEL` | Served model id, or `auto` |
| `LAIL_API_PORT` / `LAIL_WEB_PORT` / `LAIL_SERVE_ENGINE_PORT` | Ports |
| `LAIL_DATA_DIR` / `LAIL_WORKSPACES_DIR` | Data + project roots |
| `HF_TOKEN` | Optional gated HF access |

## Monorepo layout

```text
local-ai-lab/
  apps/web/                 Next.js UI (shell, Workbench, Models, Server, …)
    lib/ide-chrome.ts       Sidebar + stream chrome contract (tested)
  packages/backend/         Bun LabController + agent + proxy
  packages/serve-engine/    Python vLLM serve/bench API
  packages/shared/          Shared TS types
  workspaces/demo/          Default Composer workspace
  data/                     lail.sqlite, lab.sqlite, runs/, models/
  scripts/dev.ts            One-command: serve-engine + API + web
  docker-compose.yml
  .env.example
```

## OpenAI-compatible proxy

```bash
curl http://127.0.0.1:8787/v1/models
curl http://127.0.0.1:8787/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}'
```

`model: "auto"` / `"default"` is rewritten to the first live model when possible. Traffic is metered into **Usage**.

## Tests

```bash
cd apps/web
bun test lib/ide-chrome.test.ts lib/shell-source.test.ts
```

These assert inspo sidebar labels, Composer placeholder, Thought/Ran/Creating stream mapping (tool_end counts once), Workbench source chrome, and `/` → Workbench redirect.

## Docker Compose

```bash
docker compose up --build
```

Run vLLM or llama.cpp on the host; set `LAIL_VLLM_URL` / `LAIL_LLAMACPP_URL` (compose uses `host.docker.internal` by default).

## Migration

Previous Vite + FastAPI lab:

`~/projects/ai-lab/local-ai-lab-legacy`

Run envelopes under `data/runs` were carried forward. Serve APIs remain available through the serve-engine proxy.

## License

Private lab software — use on your own hardware.
