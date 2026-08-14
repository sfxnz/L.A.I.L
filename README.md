# L.A.I.L — Local AI Lab

**Serve & eval console** — paste a Hugging Face id, auto-configure, start/stop vLLM (or llama.cpp), run smoke/perf evals, and copy an OpenAI-compatible endpoint for **Hermes** (or any client). All on your own hardware.

Agentic coding/chat is **not** the primary surface: after Serve, wire Hermes to the live `:8000` endpoint.

| Layer | Stack |
|-------|--------|
| UI | Next.js 16 App Router + React 19 · light console chrome · Tailwind |
| Controller | Bun + Hono · lab-status, models, configure, proxy to serve-engine |
| Serve-engine | Python FastAPI · vLLM auto-configure, smoke, benches, run history |
| Models | **vLLM** and **llama.cpp** (no Ollama) |

## Product surface

Top nav (`apps/web/lib/ide-chrome.ts`):

| Page | Path | Role |
|------|------|------|
| **Status** | `/status` | Health, headroom, containers, recent runs |
| **Serve** | `/server` | Manual flags, auto-config, start/stop, job logs |
| **Models** | `/models` | Hugging Face search + local `/v1/models` |
| **Evals** | `/evals` | Smoke + perf jobs + run history |
| **Connect** | `/connect` | Hermes / OpenAI base URL snippets |
| **Configure** | `/configure` | Default backend / model |

`/` redirects to **Status** (`apps/web/app/page.tsx`). `/workbench` is a retirement notice — Hermes is the agent. `/integrations` is not shipped.

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
               └──────────▶ Hermes / laptop clients
```

### Controller pattern

One **LabController** is the public API. The Python **serve-engine** keeps the vLLM serve path (auto-configure, start, stop, agent-restore, benches). Composer agent remains in the backend for now but is **out of the primary UI**.

### Model resolution

Configure default model, or use `auto`. If the name is `default` / `auto` / empty, the controller probes `/v1/models` and uses the first served id. Prefer setting the real HF id after serve.

## Quick start

A new clone is **one local node** (this host). Multi-node is opt-in via `LAIL_CLUSTER_JSON` — see [`.env.example`](./.env.example).

### Prerequisites

- [Bun](https://bun.sh) ≥ 1.1
- Python **3.12** (3.11+ works; 3.12 matches CI and the Docker image)
- **Linux + NVIDIA (required to Start vLLM):** NVIDIA driver, [Docker](https://docs.docker.com/engine/install/), and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) so `docker run --gpus` works. Serve **Start** is docker-only.
- **Apple Silicon:** llama.cpp on `:8080`, or an SSH tunnel to a remote NVIDIA host that runs vLLM.

```bash
# 1. Clone, then stay at the repo root
git clone https://github.com/sfxnz/L.A.I.L.git lail
cd lail

# 2. Bun (skip if `bun --version` already works)
curl -fsSL https://bun.sh/install | bash
# reopen the shell, or: export PATH="$HOME/.bun/bin:$PATH"

# 3. Python deps from packages/serve-engine/pyproject.toml
python3.12 -m venv .venv   # or: python3 -m venv .venv
source .venv/bin/activate
pip install -e "packages/serve-engine[dev]"

# 4. App env + JS deps
cp .env.example .env
# Optional: set HF_TOKEN for gated models.
# Leave LAIL_CLUSTER_JSON unset for a single local node.

bun install
bun run dev
```

Open http://127.0.0.1:3000 — **`/` redirects to Status**.

| Service | URL |
|---------|-----|
| Web (Status) | http://127.0.0.1:3000 |
| Controller | http://127.0.0.1:8787 |
| Serve-engine | http://127.0.0.1:8765 |
| WebSocket | derived from the page host (controller `/ws`) |

### Linux + NVIDIA / DGX Spark

GPU + Docker check (do this once before Serve):

```bash
nvidia-smi && docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

Then in the UI:

1. **Status** — controller up; one local node unless you set `LAIL_CLUSTER_JSON`.
2. **Serve** — paste an HF model id.
3. **Auto-configure** — researches the HF card plus Unsloth / NVIDIA / GitHub / vLLM recipes and sizes flags for this host.
4. **Start** — launches the vLLM container. Watch the job dock.
5. **`/connect`** — copy the Hermes / OpenAI base URL.

Auto-configure picks util, max-model-len, vision, and tensor parallel from the researched recipe plus live hardware (keeps ≳15 GiB reserved on Spark-class UMA). Start still refuses when weights cannot fit.

### Apple Silicon

1. Prefer **llama.cpp** on `:8080`.
2. **Configure** → backend **llama.cpp**.
3. Use **Serve** when the vLLM container runs on a remote NVIDIA/Spark host (tunnel or LAN).

### From another machine (SSH tunnel)

L.A.I.L binds on the lab host. From a laptop:

```bash
ssh -L 3000:127.0.0.1:3000 -L 8787:127.0.0.1:8787 -L 8765:127.0.0.1:8765 "$USER@<lab-host>"
```

Then open http://127.0.0.1:3000. Replace `$USER@<lab-host>` with your SSH login.

## Connect (Hermes)

On **`/connect`**, copy the snippets for the live OpenAI-compatible endpoint.

**Hermes on the same host** (loopback):

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=local
OPENAI_MODEL=<served-model-id>
```

**Hermes / clients on another machine:** use that page’s host (`http://<page-host>:8000/v1`) or the SSH tunnel above. If localhost works on the lab host but a remote client fails, vLLM is likely published on `127.0.0.1:8000` only — tunnel or re-serve with an intentional LAN bind.

## Serve (vLLM serve & evals)

- Auto-configure + start · stop · agent-restore
- HF auto-configure · live job logs
- Smoke · perf · golden tools · run history envelopes

Benches hit whatever is on the vLLM base URL (default `:8000`) — **serve first**, then bench.

## Models & Usage

- **`/models`**: HF search, local list from vLLM/llama.cpp `/v1/models`, HF download jobs (not Ollama pull).
- **`/usage`**: lifetime tokens, heatmap, mix, top models (metered from proxy + agent).

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
| `LAIL_CLUSTER_JSON` | Optional. Unset = one local node. See `.env.example` for a 2-node EXAMPLE |
| `LAIL_HOST` | Bind address. Default `127.0.0.1`. Off-loopback requires `LAIL_TOKEN` |
| `LAIL_TOKEN` | Shared secret when bound off-loopback (`Authorization: Bearer` or `X-Lail-Token`) |
| `LAIL_CORS_ORIGINS` | Extra CORS origins when the UI is not on localhost |
| `LAIL_DEV_ORIGINS` | Optional extra Next.js `allowedDevOrigins` (loopback is already allowed) |

## Retired: Workbench

`/workbench` is not the landing page. It shows a retirement notice pointing at Hermes. Plan / Ask / Agent live in Hermes against the served `:8000` endpoint, not in this console.

## Monorepo layout

```text
lail/
  apps/web/                 Next.js UI (Status, Serve, Evals, Connect, …)
    lib/ide-chrome.ts       Top-nav + stream chrome contract (tested)
  packages/backend/         Bun LabController + agent + proxy
  packages/serve-engine/    Python vLLM serve/bench API (install from pyproject.toml)
  packages/shared/          Shared TS types
  workspaces/demo/          Default workspace root
  data/                     sqlite, runs/, models/ (gitignored runtime state)
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

From the repo root (Bun on `PATH`, venv activated if you use one):

```bash
bun run typecheck
bun test apps/web packages/backend
python3 -m pytest packages/serve-engine/tests -q
```

- `bun run typecheck` — web + backend TypeScript
- `apps/web` tests — nav labels (`lib/ide-chrome.test.ts`), shell source, mentions
- `packages/backend` tests — agent runtime, patches, context packer
- `packages/serve-engine` pytest — auto-config, cluster, captured corpus
- Python runtime pins: `packages/serve-engine/requirements.txt` (`uv pip compile packages/serve-engine/pyproject.toml -o packages/serve-engine/requirements.txt`)

Those tests do **not** assert a `/` → Workbench redirect. `/` redirects to `/status`.

## Docker Compose (console only)

`docker compose up` starts the **web + controller + serve-engine API**. It does **not** serve a model:

- the serve-engine image has no Docker CLI and no NVIDIA runtime
- there is no GPU vLLM service in the compose file
- **Serve → Start** will not launch a container from inside compose

Use this stack to browse Status / Connect and to talk to a vLLM you already run on the host (`LAIL_VLLM_URL`, default `host.docker.internal:8000`). To Start a model from the UI, run `bun run dev` on a Linux+NVIDIA host with Docker and the NVIDIA Container Toolkit.

Host ports are published on `127.0.0.1` only. Remapping them to `0.0.0.0` without `LAIL_TOKEN` exposes start/stop Docker and shell tools on the LAN.

```bash
docker compose up --build
```

## License

Apache-2.0. See [LICENSE](./LICENSE).
