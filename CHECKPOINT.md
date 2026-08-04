# Checkpoint — L.A.I.L (Local AI Lab)

Session handoff. Read this before continuing lab work.

## What this product is

**Cursor-style local agentic IDE** + lab tooling. **Phase A** (agent platform) is in: modes, review-first patches, streaming/cancel, risky shell approval. **Phase B** (context engine) is in: `EditorSnapshot`, `@file`/`@folder`/`@search` mentions, open-tab packing, Configure context budget.

| Surface | Role |
|---------|------|
| **Workbench** | Primary — Composer (Plan / Ask / Agent), @mentions + context chips, patch review Accept/Reject, shell approval, cancel, file tabs, Status rail |
| **Server** | vLLM Lab Safe / Workflow Max, smoke, perf, agentic, history |
| **Models** | HF search + download for vLLM/llama.cpp |
| **Usage / Configure / Status** | Metering, backends, context budget, health |

Path: `~/projects/ai-lab/local-ai-lab`  
Legacy GUI: `~/projects/ai-lab/local-ai-lab-legacy`  
Design: [Phase A](./docs/superpowers/specs/2026-07-16-lail-cursor-ide-design.md) · [Phase B](./docs/superpowers/specs/2026-07-16-lail-phase-b-context-engine-design.md)

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

## Phase A + B Workbench (handoff)

- **Modes**: Plan (no edits) · Ask (read-only) · Agent (tools + pending patches)  
- **Patches**: review-first; Accept / Reject / Accept all before disk write  
- **Streaming** + mid-run **Cancel**  
- **Risky shell**: approval banner Allow / Deny  
- **Context (Phase B)**: client builds `EditorSnapshot` (open tabs, active path, selection, mentions); server `ContextPacker` re-reads disk, runs rg for `@search`, applies ignore + budget  
- Composer UI: type `@` → `MentionPopup` inserts `@file path`; `ContextChips` show mentions + open tab count  
- **Configure → Context budget (chars)** (`contextBudgetChars`, default 32000)  
- UI: `ModeToggle`, `PatchReviewPanel`, `ShellApprovalBanner`, `MentionPopup`, `ContextChips` under `apps/web/components/workbench/`  
- Backend: `packages/backend/src/agent/` + `agent/context/` (mentions, packer, budget, ignore, search)  
- Phases C–E (Monaco, terminal, …) still planned — see design specs  

## Code map

```text
apps/web/components/layout/AppShell.tsx          # Sidebar inspo shell
apps/web/app/workbench/page.tsx                  # IDE composer + editor + status + Phase A/B UI
apps/web/components/workbench/ModeToggle.tsx     # Plan | Ask | Agent
apps/web/components/workbench/MentionPopup.tsx   # @ path popup
apps/web/components/workbench/ContextChips.tsx   # mention + open-tab chips
apps/web/components/workbench/PatchReviewPanel.tsx
apps/web/components/workbench/ShellApprovalBanner.tsx
apps/web/lib/mentions.ts                         # client parseMentions
apps/web/lib/ide-chrome.ts                       # Labels + groupTimeline (tested)
apps/web/lib/store.ts                            # agentMode, pendingPatches
packages/backend/src/agent/                      # Runtime, PatchStore, ToolPolicy, approvals
packages/backend/src/agent/context/              # ContextPacker, mentions, budget, rg search
packages/backend/src/controller/agent.ts         # Composer agent HTTP/WS (+ editorSnapshot)
packages/backend/src/controller/patches.ts       # Accept / reject patches
packages/backend/src/controller/settings.ts      # resolveModelId + contextBudgetChars
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

- **Auto-configure** pulls live HF card + config.json; scores recipes; strips unsafe `flashinfer_b12x` on mixed FP8 MoE; Lab Safe util≤0.4 envelope.
- **UI** (`apps/web/app/server/page.tsx`): full recommend panel (warnings, card recipes, rationale, sources), live status, job log always visible across tabs.
- **HF token**: stale `hf_oauth_*` in `~/.cache/huggingface/token` 401s public fetches — code retries anonymous + warns. Re-login: `hf auth login` for gated models.
- Pin image still `vllm/vllm-openai:v0.26.0`.

## Related

- Full setup: [README.md](./README.md)  
- Design: [Phase A](./docs/superpowers/specs/2026-07-16-lail-cursor-ide-design.md) · [Phase B](./docs/superpowers/specs/2026-07-16-lail-phase-b-context-engine-design.md)  
- Plans: [Phase A](./docs/superpowers/plans/2026-07-16-lail-phase-a-agent-platform.md) · [Phase B](./docs/superpowers/plans/2026-07-16-lail-phase-b-context-engine.md)  
- Env template: [.env.example](./.env.example)  
