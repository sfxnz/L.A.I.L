# L.A.I.L → Full Cursor-Style Agentic IDE

**Date:** 2026-07-16  
**Status:** Approved (brainstorm)  
**Product:** L.A.I.L (Local AI Lab) — `~/projects/ai-lab/local-ai-lab`  
**Constraint:** Retain serving and eval (Server, benches, Models, Usage, Configure, OpenAI `/v1` proxy)

---

## 1. Goal

Turn L.A.I.L from a Cursor-*inspired* MVP into a **full Cursor-style agentic browser IDE for local models**, without removing or regressing lab tooling for **vLLM / llama.cpp** serve, performance benches, and model/usage management.

**North star:** A developer on a lab host (e.g. DGX Spark) opens the browser IDE, works a real workspace with Plan / Ask / Agent, streams a local model, reviews multi-file patches like Cursor, and still manages serve + eval from the same app shell.

---

## 2. Current state (baseline)

| Layer | Today |
|-------|--------|
| **UI** | Next.js App Router, dark IDE chrome, Workbench Composer + simple file tabs + Status rail |
| **Agent** | Bun `agent.ts` tool loop (max ~12 steps), non-streaming LLM, tools write disk immediately |
| **Tools** | `list_dir`, `read_file`, `write_file`, `grep`, `run_shell`, `plan` |
| **Lab** | Python serve-engine (Lab Safe / Workflow Max, benches, history), HF models, usage metering |
| **Gap** | No review-first apply, no modes, no full streaming, no modular agent platform, no context engine, no Monaco, no terminal/git IDE surface |

---

## 3. Product decisions (locked)

| Decision | Choice |
|----------|--------|
| Scope | **All** Cursor-class subsystems (not a thin vertical slice only) |
| Build order | **A** Agent+Diff → **B** Context → **C** Editor → **D** Terminal/git → **E** Chrome polish |
| Architecture | **Modular agent platform** inside LabController (not greenfield rewrite, not deepen-only god-files) |
| Product shell | **Browser IDE** (Next.js primary; SSH-tunnel friendly) |
| File edits | **Review-first** Accept / Reject (nothing hits disk until accept) |
| Edit representation | **Search/replace** patches (`path` + `old_string` + `new_string`) |
| Shell policy | **Approve risky only**; free reads; writes via patch review |
| Modes (Phase A) | **Plan · Ask · Agent** |
| Streaming | **Full** token stream + live tool/patch events over WebSocket |
| Lab | **Retained every phase** — serve/eval never blocked or removed |

**UX bar:** Cursor-style interaction is a hard requirement — streaming Composer, tool/run cards, reviewable multi-file changes, mode switcher, approval for dangerous shell — adapted for **local** backends and first-class lab tabs.

---

## 4. Architecture

### 4.1 Principle

Keep **LabController (Bun + Hono)** and **serve-engine (Python)** as public boundaries. Grow an **IDE agent platform** as explicit modules with stable interfaces so later phases plug in without rewriting serve/eval.

```text
┌──────────────────────────────────────────────────────────────┐
│  apps/web  Browser IDE                                       │
│  AppShell · Workbench (Composer + patch review) · Lab pages  │
└────────────────────────────┬─────────────────────────────────┘
                             │ REST + WS :8787
┌────────────────────────────▼─────────────────────────────────┐
│  LabController                                               │
│  AgentRuntime · PatchStore · ToolPolicy · EventBus           │
│  ContextProvider* · EditorBridge* · TerminalSession*         │
│  sessions · workspaces · models · usage · /v1 · serve/* proxy│
└──────────────┬──────────────────────────────┬────────────────┘
               │ OpenAI /v1                   │ proxy
               ▼                              ▼
        vLLM / llama.cpp              packages/serve-engine
```

\* Interface present in Phase A; full implementation in later phases.

### 4.2 Module responsibilities

| Module | Responsibility |
|--------|----------------|
| **AgentRuntime** | Plan / Ask / Agent modes, streaming tool loop, cancel, step limits, mode-specific system prompts |
| **PatchStore** | Pending search/replace hunks, Accept / Reject / Accept all, apply to disk, history of applied patches |
| **ToolPolicy** | Workspace sandbox, mode tool allowlists, risky-shell detection and approval gate |
| **EventBus** | Typed WebSocket events for Composer UI |
| **ContextProvider** | Build model context (Phase A thin; Phase B full) |
| **EditorBridge** | Open/reveal files (Phase A: existing tabs; Phase C: Monaco) |
| **Lab surfaces** | Server, Models, Usage, Configure, Status, Integrations — product role unchanged |

### 4.3 Package layout (Phase A)

Prefer modules under `packages/backend/src/` (e.g. `agent/runtime.ts`, `agent/patch-store.ts`, `agent/tool-policy.ts`, `agent/events.ts`) rather than a greenfield `packages/agent-core` package. Extract a separate package only if Phase A boundaries prove painful.

Do **not** move agent logic into Next.js route handlers or `page.tsx`.

---

## 5. Phase A detailed design

Phase A is the first implementation target and the focus of the first implementation plan.

### 5.1 Modes

| Mode | Tools | Disk | Shell |
|------|--------|------|--------|
| **Plan** | Read-only: `list_dir`, `read_file`, `grep`, `plan` | No | No |
| **Ask** | Read-only: `list_dir`, `read_file`, `grep` (no `plan` tool required) | No | No |
| **Agent** | Full set including patch + shell tools | Propose only → PatchStore | Yes; risky needs approval |

**Plan vs Ask (Phase A):** Same safety class (no writes, no shell). Differ by **system prompt and UX**, not by a large tool gap:

- **Plan** — produce a numbered implementation plan; prefer the `plan` tool; do not start coding; end with clear steps and risks.  
- **Ask** — answer questions about the codebase; use reads to cite files; no multi-step implementation plan unless asked.  
- **Agent** — implement: explore, propose patches, run shell as needed.

- Mode selected on the Composer input bar; applies to the **next** run (not mid-flight mutation of an active run).
- Plan/Ask attempting write/shell tools: tool result error `"not available in this mode"` (enforced by ToolPolicy, not model honor system).

### 5.2 Tools

| Tool | Behavior |
|------|----------|
| `list_dir`, `read_file`, `grep` | Immediate, workspace-scoped |
| `search_replace` | Creates **pending** patch; **does not write disk** |
| `create_file` | Pending create (`old_string` empty / create flag) |
| `delete_file` | Pending delete (explicit flag; review shows deletion) |
| `run_shell` | ToolPolicy → execute or `shell_approval_required` |
| `plan` | Timeline structured steps (all modes) |

**Primary edit path:** `search_replace`, not free-form `write_file` to disk.  
Internal apply used only when the user Accepts a patch.

### 5.3 Patch model

```ts
type Patch = {
  id: string;
  runId: string;
  sessionId: string;
  path: string;           // workspace-relative
  old_string: string;
  new_string: string;
  op: "replace" | "create" | "delete";
  status: "pending" | "accepted" | "rejected" | "failed";
  reason?: string;        // failed: no match | ambiguous | path escape | race
  createdAt: string;
  resolvedAt?: string;
};
```

**Apply rules:**

1. Resolve path under workspace root only.  
2. Re-read file immediately before apply.  
3. Exact `old_string` must match **exactly once**; 0 or >1 → `failed`, disk unchanged.  
4. Create: fail if file exists (unless explicitly overwrite policy later).  
5. Delete: fail if file missing.  
6. Accept all: apply pending in stable order; report per-patch failures; already-accepted patches in that batch are **not** rolled back (document in UI).  

**Persistence:** SQLite tables for patches (and agent runs) so refresh does not lose pending review state.

### 5.4 ToolPolicy — risky shell

**Always blocked (no prompt):** path escape outside workspace root.

**Needs approval (Agent mode):** heuristic denylist including, but not limited to:

- `rm -rf`, `sudo`, `mkfs`, `dd if=`, `curl … | sh`, `wget … | sh`
- `git push`, `git reset --hard`, force-push patterns
- Writes clearly targeting absolute paths outside workspace

**Safe by default:** read-oriented and normal project commands (`ls`, `cat`, `rg`, `npm test`, `bun test`, etc.) within cwd = workspace.

Approval: per command, tied to `runId` + `approvalId`; UI Allow / Deny; timeout (e.g. 120s) = Deny; tool result communicates denial so the model can adapt.

### 5.5 AgentRuntime loop

1. `POST /agent/runs` with `{ sessionId, message, mode, workspaceId? }` → `{ runId }`.  
2. Subscribe WS channel `agent:{runId}`.  
3. Build messages: system (mode-specific) + history (filter poison LLM errors) + `ContextProvider.buildContext()`.  
4. Stream chat completions to configured vLLM/llama.cpp OpenAI base.  
5. On tool calls: policy → execute or wait for approval → publish events → append tool results.  
6. Cap steps (default **32**, configurable).  
7. End with `done` | `error` | `cancelled`.  
8. Meter usage via existing usage recorder when tokens available.

**Cancel:** `POST /agent/runs/:runId/cancel` aborts LLM stream; run status `cancelled`; **pending patches remain** for user review.

**Retire or lab-flag:** offline “survival guide” auto-write fallback must not masquerade as Agent success when the model is down.

### 5.6 EventBus events (typed)

| Event | Purpose |
|-------|---------|
| `token` | Streaming assistant/thought text deltas |
| `thought` | Coarse thought markers when not using token channel |
| `status` | Working / progress strings |
| `tool_start` / `tool_end` | Tool lifecycle → UI “Ran N commands” |
| `patch_proposed` | New pending patch |
| `patch_updated` | Status change (accepted/rejected/failed) |
| `shell_approval_required` | Payload: command, approvalId |
| `error` | Recoverable or terminal errors |
| `done` / `cancelled` | Terminal run states |

UI mapping continues to honor inspo chrome: **Thought**, **Ran N command(s)**, file/patch labels, Composer placeholder **“Ask for follow-up changes”**.

### 5.7 Workbench UI (Phase A)

- Mode toggle: **Plan | Ask | Agent** on input bar  
- Streaming Composer column (Cursor-style cards)  
- **Patch review panel**: per-file list, hunk preview (old→new), Accept / Reject / Accept all  
- **Shell approval** inline or modal: Allow once / Deny  
- **Cancel** control on active run  
- Existing simple editor tabs remain until Phase C  
- Lab navigation (Status, Models, Server, …) unchanged  

### 5.8 ContextProvider (Phase A stub)

Returns:

- Recent session messages (existing windowing)  
- Optional user-attached paths if UI supports attach later in A  
- **No** semantic index, **no** automatic `@` (Phase B)

### 5.9 API surface (illustrative)

```text
POST   /agent/runs
GET    /agent/runs/:runId
POST   /agent/runs/:runId/cancel
POST   /agent/runs/:runId/shell-approvals/:approvalId  { decision: "allow" | "deny" }

GET    /patches?sessionId=&status=pending
POST   /patches/:id/accept
POST   /patches/:id/reject
POST   /patches/accept-all  { sessionId | runId }

WS     /ws  channels: agent, agent:{runId}
```

Existing sessions, workspaces, models, usage, serve proxy routes remain.

### 5.10 Errors and safety

| Class | Behavior |
|-------|----------|
| Model 404 | Re-resolve model id once; then error card |
| Stream timeout | `error` / cancel; no silent fake success |
| Bad tool JSON | tool_end with parse error; continue loop |
| Patch match fail | `failed`, disk unchanged, visible in review |
| Path escape | Hard fail at proposal/execute time |
| Shell deny/timeout | Tool result denied; agent continues |
| Agent crash | Must not take down serve-engine or `/v1` proxy |

Never show “Wrote file” unless Accept apply succeeded.

### 5.11 Testing (Phase A)

| Layer | Coverage |
|-------|----------|
| **Unit** | PatchStore match/apply/ambiguous; ToolPolicy risky/safe/escape; mode allowlists; timeline/stream mappers |
| **Integration** | Mock streaming LLM with tool_calls → pending patches → accept writes under temp workspace; shell deny path |
| **UI contract** | Mode labels, placeholders, Ran N / Thought / patch strings (extend `ide-chrome` / shell-source tests) |
| **Lab regression** | serve-engine autoconfig tests remain green; health + `/v1` smoke |

No real GPU required for unit/integration; mock OpenAI-compatible stream.

### 5.12 Phase A definition of done

In the browser against a real workspace and local model:

1. Switch **Plan / Ask / Agent**  
2. See **streaming** tokens and tool cards  
3. Receive **pending** search/replace patches (no silent disk writes)  
4. **Accept / Reject** with correct on-disk results  
5. Get prompted only for **risky** shell  
6. **Server / Models / Usage** (and related lab pages) still work as today  

---

## 6. Master roadmap (Phases B–E)

Nothing is cut from the end goal. Each phase gets its own detailed spec before implementation.

### Phase B — Context engine

- `@file`, `@folder`, `@code` (or equivalent) mentions in Composer  
- Open tabs + current selection as automatic context  
- Workspace search (ripgrep-backed initially; optional embeddings later)  
- `.lailignore` / respect `.gitignore`  
- Token budgets and truncation strategy tuned for local context windows  
- Implements full `ContextProvider`

### Phase C — Editor surface

- Monaco (or equivalent) multi-tab editor  
- Split panes, find-in-file, syntax highlighting, dirty/save state  
- Jump-to from patch review / agent file events via `EditorBridge`  
- Replace Phase A textarea tabs

### Phase D — Terminal + git

- Integrated terminal panel (cwd workspace)  
- Agent-linked shell visibility (commands from tools surface in terminal when appropriate)  
- Git status, diff, commit assist (still subject to ToolPolicy for destructive ops)

### Phase E — IDE chrome polish

- Command palette  
- Keyboard map (Cursor-like bindings where they don’t fight browser)  
- Pinned chats polish, multi-workspace UX  
- Mode UX refinements (if any deferred from A)

### Lab (continuous)

- vLLM Lab Safe / Workflow Max, stop, agent-restore  
- Smoke / perf / agentic benches, run history  
- HF model search/download  
- Usage metering and OpenAI-compatible proxy  
- Never gated on IDE phase completion  

---

## 7. Key decisions (summary)

1. **Full product, phased delivery** — all Cursor-class capabilities; implement A→E so each phase is shippable.  
2. **Modular agent platform in LabController** — stable Runtime / PatchStore / ToolPolicy / EventBus; avoid god-file growth and avoid greenfield rewrite of lab stack.  
3. **Review-first search/replace patches** — local-model-friendly edit contract + Cursor-like human gate.  
4. **Plan / Ask / Agent from Phase A** — mode allowlists enforced in ToolPolicy.  
5. **Risky-shell approval only** — lab-speed defaults without silent `rm -rf`.  
6. **Full streaming over WS** — Cursor-feel Composer with cancel support.  
7. **Browser IDE shell** — Next.js remains primary; desktop wrapper is out of scope for this roadmap.  
8. **Serve/eval retained** — first-class lab surfaces and serve-engine stay; agent metering continues to feed Usage.  

---

## 8. Non-goals (this roadmap)

- Electron/Tauri desktop app (optional later packaging)  
- Cloud multi-tenant auth / collaboration  
- Replacing vLLM/llama.cpp with Ollama  
- Full LSP language intelligence in Phase A–B (editor may add later)  
- Guaranteeing cloud-Cursor quality on tiny local models (design optimizes protocol and UX; model quality remains hardware/serve-dependent)

---

## 9. Open questions

None blocking Phase A design. Defer to phase-specific specs:

- Embeddings backend for Phase B semantic search (local vs none)  
- Monaco vs CodeMirror for Phase C  
- PTY provider for Phase D terminal  

---

## 10. Implementation note

Next step after user review of this spec: **writing-plans** skill → detailed implementation plan for **Phase A** (with roadmap checklist for B–E). No application code until that plan is approved and execution begins.
