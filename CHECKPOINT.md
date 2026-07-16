# Checkpoint — L.A.I.L (Local AI Lab)

Session handoff. Read this before continuing lab work.

## What this product is

**Cursor-style local agentic IDE** + lab tooling:

| Surface | Role |
|---------|------|
| **Workbench** | Primary — Composer agent, file tabs, Status rail |
| **Server** | vLLM Lab Safe / Workflow Max, smoke, perf, agentic, history |
| **Models** | HF search + download for vLLM/llama.cpp |
| **Usage / Configure / Status** | Metering, backends, health |

Path: `~/projects/ai-lab/local-ai-lab`  
Legacy GUI: `~/projects/ai-lab/local-ai-lab-legacy`

## Start

```bash
export PATH="$HOME/.bun/bin:$PATH"
cd ~/projects/ai-lab/local-ai-lab
source .venv/bin/activate   # if using venv for serve-engine
bun run dev
```

| Service | Port |
|---------|------|
| Web | **3000** → `/workbench` |
| Controller | **8787** |
| Serve-engine | **8765** |

Mac tunnel: `ssh -L 3000:127.0.0.1:3000 -L 8787:127.0.0.1:8787 sfxnz@spark1`

## Model gotcha

If Composer says model `default` does not exist:

1. Serve the model (Server tab).  
2. **Configure** → set Default model to the real id (e.g. `unsloth/Qwen3.6-35B-A3B-NVFP4`).  
3. Or use `auto` and ensure `/v1/models` returns something.  
4. Prefer **New chat** after fixing (old sessions may still show prior errors in history; stream filters error noise).  

## GPU util (Spark UMA)

Same as legacy lab:

- **0.4** Lab Safe · **0.7–0.85** Workflow Max with free RAM buffer  
- One large model at a time; trust `free -h` available more than `docker stats` on UMA  

## Code map

```text
apps/web/components/layout/AppShell.tsx   # Sidebar inspo shell
apps/web/app/workbench/page.tsx           # IDE composer + editor + status
apps/web/lib/ide-chrome.ts                # Labels + groupTimeline (tested)
packages/backend/src/controller/agent.ts  # Composer agent loop
packages/backend/src/controller/settings.ts  # resolveModelId
packages/serve-engine/                    # vLLM serve/bench
```

## Tests

```bash
cd apps/web && bun test lib/ide-chrome.test.ts lib/shell-source.test.ts
```

## Related

- Full setup: [README.md](./README.md)  
- Env template: [.env.example](./.env.example)  
