# L.A.I.L — Local AI Lab

**Local-first agentic IDE** for your own models — Cursor-style Composer + project tree + file editor, plus lab tooling for **vLLM / llama.cpp** serve, models, and usage.

Not just inference/evals: the primary surface is **Workbench** (agent that reads/writes the repo and runs shell tools). Server, Models, and Usage support the lab around that IDE.

| Layer | Stack |
|-------|--------|
| UI | Next.js 16 App Router + React 19 · dark IDE chrome · Tailwind |
| Controller | Bun + Hono · sessions, workspaces, agent, HF models, usage, OpenAI `/v1` proxy, WebSockets |
| Serve-engine | Python FastAPI (ported) · vLLM Lab Safe / Workflow Max, benches, run history |
| Models | **vLLM** and **llama.cpp** only (no Ollama) |

## Product surface (matches inspo IDE shell)

Left sidebar:

| Section | Contents |
|---------|----------|
| **Search** | Filter tasks / sessions |
| **Workspace** | Status · Workbench · Models · Configure · Usage · Integrations · Server |
| **Pinned** | Pinned chats |
| **Tasks** | Recent Composer sessions |
| **Projects** | Workspace roots + file tree (click a file → editor tab) |

**Workbench** (default home):

- Composer stream: user bubble → **Thought** → **Ran N command(s)** → **Creating** / file write → final answer  
- Bottom bar: **Ask for follow-up changes** · model label · send  
- Optional **editor** tabs when the agent writes (or you open) a project file  
- Right **Status** rail: session state, model, workspace  

Lab pages (same dark chrome): **Models** (HF search / download), **Server** (full vLLM serve + perf/agentic/history), **Usage**, **Configure**, **Status**, **Integrations**.

## Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│  apps/web  Next.js 16 + React 19                             │
│  Cursor-style shell · Workbench IDE · Models / Server / …    │
└────────────────────────────┬─────────────────────────────────┘
                             │ REST + WS :8787
┌────────────────────────────▼─────────────────────────────────┐
│  LabController (Bun + Hono)                                  │
│  sessions · workspaces · composer agent · models · usage     │
│  OpenAI /v1 proxy · serve/* · bench/* · runs/* ──proxy──▶    │
└──────────────┬─────────────────────────────┬─────────────────┘
               │                             │
               ▼                             ▼
     vLLM / llama.cpp                  packages/serve-engine
     (OpenAI /v1)                      (Python: vLLM docker serve,
                                        perf, agentic, history)
```

### Controller pattern

One **LabController** is the public API. The Python **serve-engine** keeps the Spark/vLLM serve path (Lab Safe util ≤ 0.4, Workflow Max, stop, agent-restore, benches, envelopes). Composer, HF library, sessions/workspaces, and token metering live in Bun.

### Model resolution

Configure default model, or use `auto`. If the name is `default` / `auto` / empty, the controller probes `/v1/models` and uses the first served id (avoids vLLM 404 on a placeholder name). Prefer setting the real id after serve, e.g. `unsloth/Qwen3.6-35B-A3B-NVFP4`.

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

1. Open **Workbench** (or `/`).  
2. Ensure vLLM or llama.cpp is healthy (**Status** / **Server**).  
3. Set **Configure → Default model** to the served model id (or leave auto).  
4. Ask the agent to explore files, run shell (cwd = workspace), edit/create docs.  
5. Written files open in **editor tabs**; use **Save** after manual edits.  

**Composer stream labels** (inspo-style):

| UI | Meaning |
|----|---------|
| **Thought** | Model reasoning step |
| **Ran N command(s)** | Completed tool calls (counts **tool_end** only) |
| **Creating path** | File write; opens editor tab |
| **Working** | In-progress turn |

Bottom placeholder: **Ask for follow-up changes**.

Offline fallback: if the LLM is down, tools can still run for simple file work (e.g. survival guide path).

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
