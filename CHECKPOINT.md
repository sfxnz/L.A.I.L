# Checkpoint — L.A.I.L (Local AI Lab)

Session handoff. Read this before continuing lab work.

## What this product is

**Cursor-style local agentic IDE** + lab tooling. **Phase A** (agent platform) is in: modes, review-first patches, streaming/cancel, risky shell approval.

| Surface | Role |
|---------|------|
| **Workbench** | Primary — Composer (Plan / Ask / Agent), patch review Accept/Reject, shell approval, cancel, file tabs, Status rail |
| **Server** | vLLM Lab Safe / Workflow Max, smoke, perf, agentic, history |
| **Models** | HF search + download for vLLM/llama.cpp |
| **Usage / Configure / Status** | Metering, backends, health |

Path: `~/projects/ai-lab/local-ai-lab`  
Legacy GUI: `~/projects/ai-lab/local-ai-lab-legacy`  
Design: [docs/superpowers/specs/2026-07-16-lail-cursor-ide-design.md](./docs/superpowers/specs/2026-07-16-lail-cursor-ide-design.md)

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

## Phase A Workbench (handoff)

- **Modes**: Plan (no edits) · Ask (read-only) · Agent (tools + pending patches)  
- **Patches**: review-first; Accept / Reject / Accept all before disk write  
- **Streaming** + mid-run **Cancel**  
- **Risky shell**: approval banner Allow / Deny  
- UI: `ModeToggle`, `PatchReviewPanel`, `ShellApprovalBanner` under `apps/web/components/workbench/`  
- Backend agent modules: `packages/backend/src/agent/` (runtime, patch-store, tool-policy, approvals)  
- Phases B–E (Monaco, context index, terminal, …) still planned — see design spec  

## Code map

```text
apps/web/components/layout/AppShell.tsx          # Sidebar inspo shell
apps/web/app/workbench/page.tsx                  # IDE composer + editor + status + Phase A UI
apps/web/components/workbench/ModeToggle.tsx     # Plan | Ask | Agent
apps/web/components/workbench/PatchReviewPanel.tsx
apps/web/components/workbench/ShellApprovalBanner.tsx
apps/web/lib/ide-chrome.ts                       # Labels + groupTimeline (tested)
apps/web/lib/store.ts                            # agentMode, pendingPatches
packages/backend/src/agent/                      # Runtime, PatchStore, ToolPolicy, approvals
packages/backend/src/controller/agent.ts         # Composer agent HTTP/WS
packages/backend/src/controller/patches.ts       # Accept / reject patches
packages/backend/src/controller/settings.ts      # resolveModelId
packages/serve-engine/                           # vLLM serve/bench
```

## Tests

```bash
export PATH="$HOME/.bun/bin:$PATH"
cd apps/web && bun test
cd packages/backend && bun test
```

## Related

- Full setup: [README.md](./README.md)  
- Design: [docs/superpowers/specs/2026-07-16-lail-cursor-ide-design.md](./docs/superpowers/specs/2026-07-16-lail-cursor-ide-design.md)  
- Phase A plan: [docs/superpowers/plans/2026-07-16-lail-phase-a-agent-platform.md](./docs/superpowers/plans/2026-07-16-lail-phase-a-agent-platform.md)  
- Env template: [.env.example](./.env.example)  
