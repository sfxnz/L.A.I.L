# Checkpoint — L.A.I.L (Local AI Lab)

> Maintainer session notes, not user documentation. See [README.md](./README.md) to run L.A.I.L.

Session handoff. Primary product is the **serve & eval console**. Composer/Workbench is retired — Hermes is the agent.

| Surface | Role |
|---------|------|
| **Status** | Landing (`/` → `/status`) — health, cluster, recent runs |
| **Serve** | vLLM auto-configure, start/stop, job logs |
| **Evals** | Smoke + perf + tool-eval history |
| **Connect** | `/connect` — Hermes / OpenAI snippets |
| **Configure** | Default backend / model |

Design (retired Workbench): [Phase A](./docs/superpowers/specs/2026-07-16-lail-cursor-ide-design.md) · [Phase B](./docs/superpowers/specs/2026-07-16-lail-phase-b-context-engine-design.md)

## Start

From the repo root (see README Quick start for clone + `pip install -e "packages/serve-engine[dev]"`):

```bash
export PATH="$HOME/.bun/bin:$PATH"
source .venv/bin/activate   # if using venv for serve-engine
bun run dev
```

| Service | Port |
|---------|------|
| Web | **3000** → `/status` |
| Controller | **8787** |
| Serve-engine | **8765** |

Laptop tunnel: `ssh -L 3000:127.0.0.1:3000 -L 8787:127.0.0.1:8787 -L 8765:127.0.0.1:8765 "$USER@<lab-host>"`

## Model gotcha

If Composer says model `default` does not exist:

1. Serve the model (Server tab).  
2. **Configure** → set Default model to the real id (e.g. `unsloth/Qwen3.6-35B-A3B-NVFP4`).  
3. Or use `auto` and ensure `/v1/models` returns something.  
4. Prefer **New chat** after fixing (old sessions may still show prior errors in history; stream filters error noise).  

## GPU util (Spark UMA)

Same as legacy lab:

- Auto-configure sizes util (typically ~0.85 after reserved UMA) from weights + live topology  
- One large model at a time; trust `free -h` available more than `docker stats` on UMA  

## Phase A + B Workbench (retired)

Removed: the Workbench page, `components/workbench/*`, the backend agent runtime / patches / context packer, and **Configure → Context budget** (`contextBudgetChars`). Hermes against `:8000` is the agent. Existing sqlite tables from that era are left in place, unused.

## Code map

```text
apps/web/components/layout/AppShell.tsx          # Sidebar inspo shell
apps/web/lib/ide-chrome.ts                       # Labels + groupTimeline (tested)
packages/backend/src/controller/settings.ts      # resolveModelId
packages/backend/src/controller/usage.ts         # Usage from engine /metrics counters
packages/serve-engine/                           # vLLM serve/bench
```

## Tests

```bash
export PATH="$HOME/.bun/bin:$PATH"
cd apps/web && bun test
cd packages/backend && bun test
# Serve auto-config (Python)
cd packages/serve-engine && PYTHONPATH=. python -m pytest tests/test_autoconfig.py -q
```

## Server section (2026-07-21)

- **Auto-configure** pulls the live HF card + config.json, then researches Unsloth / NVIDIA / GitHub / vLLM recipes even when the card has no links; scores recipes; strips unsafe `flashinfer_b12x` on mixed FP8 MoE; sizes util / max-len / VL from hardware.
- **UI** (`apps/web/app/server/page.tsx`): full recommend panel (warnings, card recipes, rationale, sources), live status, job log always visible across tabs.
- **HF token**: stale `hf_oauth_*` in `~/.cache/huggingface/token` 401s public fetches — code retries anonymous + warns. Re-login: `hf auth login` for gated models.
- Pin image still `vllm/vllm-openai:v0.27.1`.

## Related

- Full setup: [README.md](./README.md)  
- Design: [Phase A](./docs/superpowers/specs/2026-07-16-lail-cursor-ide-design.md) · [Phase B](./docs/superpowers/specs/2026-07-16-lail-phase-b-context-engine-design.md)  
- Plans: [Phase A](./docs/superpowers/plans/2026-07-16-lail-phase-a-agent-platform.md) · [Phase B](./docs/superpowers/plans/2026-07-16-lail-phase-b-context-engine.md)  
- Env template: [.env.example](./.env.example)  
