# L.A.I.L — Local AI Lab

**Serve & eval console** — paste a Hugging Face id, auto-configure, start/stop vLLM, SGLang, llama.cpp or TensorFold, run smoke/perf evals, and copy an OpenAI-compatible endpoint for **Hermes** (or any client). All on your own hardware.

Agentic coding/chat is **not** the primary surface: after Serve, wire Hermes to the live `:8000` endpoint.

| Layer | Stack |
|-------|--------|
| UI | Next.js 16 App Router + React 19 · light console chrome · Tailwind |
| Controller | Bun + Hono · live status stream, configure, usage, lab gallery, proxy to serve-engine |
| Serve-engine | Python FastAPI · auto-configure, launch / stop / live metrics per engine, smoke, benches, run history |
| Engines | **vLLM**, **SGLang**, **llama.cpp**, **TensorFold** (no Ollama) |

## Product surface

Top nav (`apps/web/lib/ide-chrome.ts`); Connect (`/connect`) is linked from Status and Serve:

| Page | Path | Role |
|------|------|------|
| **Status** | `/status` | Live: the served model (decode / throughput / TTFT / requests / KV, Hermes wiring), each Spark's hardware, the fabric, the last decode bench |
| **Serve** | `/server` | Auto-config, flags, start/stop, job logs |
| **Bench** | `/bench` | Decode and prefill benches |
| **Streams** | `/streams` | Concurrent strands, live |
| **Evals** | `/evals` | Smoke, golden tools, tool-eval-bench, run log |
| **Connect** | `/connect` | Hermes / OpenAI base URL snippets, Tailscale URL, curl probes |
| **Configure** | `/configure` | Default backend / model |

`/` redirects to **Status** (`apps/web/app/page.tsx`). `/workbench`, `/integrations` and `/models` are retired and redirect to live pages — Hermes is the agent; Serve’s “Download weights first” fetches into the HF cache.

## Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│  apps/web  Next.js 16 + React 19                             │
│  White console · Status · Serve · Evals · Connect            │
└────────────────────────────┬─────────────────────────────────┘
                             │ REST :8787
┌────────────────────────────▼─────────────────────────────────┐
│  LabController (Bun + Hono)                                  │
│  /api/live stream · configure · usage · serve/* bench/* proxy│
└──────────────┬─────────────────────────────┬─────────────────┘
               │                             │
               ▼                             ▼
     vLLM / SGLang / llama.cpp /       packages/serve-engine
     TensorFold
     (OpenAI /v1)                      (Python: docker serve,
                                        smoke, perf, history)
               │
               └──────────▶ Hermes / laptop clients
```

### Live data

serve-engine samples every second (endpoint `/metrics`, this host, a persistent ssh
telemetry stream per remote Spark). The controller long-polls it
(`/api/status?after=<sampled_at_ms>`) and fans each sample out to every open tab over
one server-sent event stream, `GET /api/live` (`meta` once, `history` on connect,
`tick` per sample). The browser reads it with `fetch()` and the `X-Lail-Token`
header, and falls back to polling `/api/lab-status` only if the stream cannot get
through. Every timestamp is the serve-engine host's clock; sparklines sit on a real
time axis and stale data is dimmed with its age, never shown as live.

### Controller pattern

One **LabController** is the public API. The Python **serve-engine** owns the serve path for every engine (auto-configure, start, stop, live metrics, benches). The composer agent is gone from the backend (see *Retired: Workbench*).

### Model resolution

Configure default model, or use `auto`. If the name is `default` / `auto` / empty, the controller probes `/v1/models` and uses the first served id. Prefer setting the real HF id after serve.

## Quick start

A new clone is **this host**, plus any QSFP/RoCE peers that answer ping. Pin topology with `LAIL_CLUSTER_JSON` or gitignored `data/cluster.json` — see [`.env.example`](./.env.example).

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

`bun run dev` serves the web app with `next dev` (hot reload, the React development
build: several MB of unminified JS, re-checked on every render). For the console you
leave open all day, run the same stack with the production web build instead:

```bash
bun run start:prod   # next build (into apps/web/.next-prod), then next start
```

Same ports, same env, same controller and serve-engine; only the web server differs.
Stop the dev stack first — both bind :3000.

| Service | URL |
|---------|-----|
| Web (Status) | http://127.0.0.1:3000 |
| Controller | http://127.0.0.1:8787 |
| Serve-engine | http://127.0.0.1:8765 |

### Linux + NVIDIA / DGX Spark

GPU + Docker check (do this once before Serve):

```bash
nvidia-smi && docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

Then in the UI:

1. **Status** — controller up; this host plus live RoCE peers (or the topology you pinned).
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

Status (the served-model panel) and Serve (the live endpoint panel) show the OpenAI
base URL of the port that answers, with copy buttons for it, the model id and a
ready env block. **`/connect`** has the rest: the Tailscale URL and curl probes.

**Hermes on the same host** (loopback):

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=local
OPENAI_MODEL=<served-model-id>
```

**Hermes / clients on another machine:** use that page’s host (`http://<page-host>:8000/v1`) or the SSH tunnel above. If localhost works on the lab host but a remote client fails, vLLM is likely published on `127.0.0.1:8000` only — tunnel or re-serve with an intentional LAN bind.

## Serve (vLLM serve & evals)

- Auto-configure + start · stop (two clicks) · live job logs — vLLM, SGLang, llama.cpp, TensorFold
- Bench (decode / prefill) on `/bench`; smoke, golden tools and tool-eval-bench on `/evals`

Benches (and the controller's `/v1` proxy) hit the endpoint the serve-engine detects serving — whichever engine and port — else the configured default backend. **Serve first**, then bench.

## Engines

Serve → **Engine** picks one; `packages/serve-engine/app/services/engines/` holds one small
adapter per engine (image, port, CLI from the placement result, readiness, `/metrics`
mapping). Status shows the engine actually answering (`/v1/models` owned_by, else the
metrics prefix), not the configured default.

| Engine | Default image (env override) | Port | TP across Sparks | Live metrics |
|--------|------------------------------|------|------------------|--------------|
| vLLM | `vllm/vllm-openai:v0.27.1` (`LAB_VLLM_IMAGE_MAX`) | 8000 | yes (`--nnodes`, workers `--headless`) | per step |
| SGLang | `lmsysorg/sglang:v0.5.20-cu130` (`LAIL_SGLANG_IMAGE`) | 30000 | yes (`--nnodes/--node-rank/--dist-init-addr`) | per-stream decode from the inter-token histogram, throughput from the scheduler's `gen_throughput`; `--enable-metrics` always on |
| llama.cpp | `ghcr.io/ggml-org/llama.cpp:server-cuda13` (`LAIL_LLAMACPP_IMAGE`) | 8080 | no (single node) | when a request ends (shown as "last"), `--metrics` always on |
| TensorFold | `nvcr.io/nvidia/pytorch:26.07-py3` + `pip install` of the verified commit at start (`LAIL_TENSORFOLD_IMAGE`, `LAIL_TENSORFOLD_PIP`); pip and kernel-build caches persist in `~/.cache/lail-tensorfold` (`LAIL_TENSORFOLD_CACHE`) | **8090** (upstream 8080 is llama.cpp's) | 2 ranks max (`--tp 2 --rank R`) | decode from `/health` running totals; no KV pool % (its ratio is per-stream context fill) |

Every container L.A.I.L launches carries `--label lail.engine=<name>`; Stop removes those
(any state) plus any other serve container that is running, judged by its command or
image on every node — never by a model word in its name. The API binds `127.0.0.1` on
single-node and multi-node serves alike. Recipes in `data/serve_overlays.json` may carry
`"engine": "sglang" | "llamacpp" | "tensorfold"` to apply only to that engine.

## Models & Usage

- **`/models`** is retired (redirects to Server): weights download through Serve’s “Download weights first” option into the HF cache the engine reads.
- **`/usage`**: lifetime tokens, heatmap, mix, top models — metered every 15 s from the engine’s own `/metrics` counters, so it covers every client, including Hermes direct to `:8000` (vLLM always; SGLang with `--enable-metrics`; llama-server with `--metrics`).

## Environment

See [`.env.example`](./.env.example).

| Variable | Role |
|----------|------|
| `LAIL_DEFAULT_BACKEND` | `vllm`, `sglang`, `llamacpp` or `tensorfold` |
| `LAIL_VLLM_URL` | Default `http://127.0.0.1:8000` |
| `LAIL_SGLANG_URL` | Default `http://127.0.0.1:30000` |
| `LAIL_LLAMACPP_URL` | Default `http://127.0.0.1:8080` |
| `LAIL_TENSORFOLD_URL` | Default `http://127.0.0.1:8090` (TensorFold's own default 8080 collides with llama.cpp; L.A.I.L launches it on 8090) |
| `LAIL_DEFAULT_MODEL` | Served model id, or `auto` |
| `LAIL_API_PORT` / `LAIL_WEB_PORT` / `LAIL_SERVE_ENGINE_PORT` | Ports |
| `LAIL_DATA_DIR` | Data root (sqlite, lab runs) |
| `HF_TOKEN` | Optional gated HF access |
| `LAIL_CLUSTER_JSON` | Optional pin. Unset = this host + live RoCE peers. See `.env.example` |
| `LAIL_HOST` | Bind address. Default `127.0.0.1`. Off-loopback requires `LAIL_TOKEN` |
| `LAIL_TOKEN` | Shared secret when bound off-loopback (`Authorization: Bearer` or `X-Lail-Token`) |
| `LAIL_CORS_ORIGINS` | Extra CORS origins when the UI is not on localhost |
| `LAIL_DEV_ORIGINS` | Optional extra Next.js `allowedDevOrigins` (loopback is already allowed) |

## Retired: Workbench

`/workbench` redirects to Status; its controller backend (workspaces, sessions, agent runs, patches) is gone. Plan / Ask / Agent live in Hermes against the served `:8000` endpoint, not in this console.

## Monorepo layout

```text
lail/
  apps/web/                 Next.js UI (Status, Serve, Evals, Connect, …)
    lib/ide-chrome.ts       Top-nav contract (tested)
  packages/backend/         Bun LabController + proxies + streams
  packages/serve-engine/    Python vLLM serve/bench API (install from pyproject.toml)
  packages/shared/          Shared TS types
  workspaces/demo/          Example artifacts (lab gallery import sample)
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
- `apps/web` tests — nav labels (`lib/ide-chrome.test.ts`), API/token handling, live status store, stream runs
- `packages/backend` tests — usage metering, serve-engine and `/v1` proxies, lab play capability, streams
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
