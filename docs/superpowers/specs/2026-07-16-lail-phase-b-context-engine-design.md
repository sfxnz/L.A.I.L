# L.A.I.L Phase B — Context Engine Design

**Date:** 2026-07-16  
**Status:** Approved (brainstorm)  
**Product:** L.A.I.L (Local AI Lab) — `~/projects/ai-lab/local-ai-lab`  
**Depends on:** Phase A agent platform (modes, PatchStore, streaming Runtime)  
**Spec parent:** `docs/superpowers/specs/2026-07-16-lail-cursor-ide-design.md` §6 Phase B

---

## 1. Goal

Give local models **Cursor-style project awareness** without embeddings:

- Explicit **`@file` / `@folder` / `@search`** (alias `@code`) in Composer  
- **Automatic** context from **open tabs + active selection** only (no auto-related-file retrieval)  
- **ripgrep-backed** workspace search for `@search`  
- **Configurable character budget** with priority-based truncation  
- **Ignore rules** (`.gitignore` + `.lailignore`)  
- Full **`ContextProvider` / packer** used by `AgentRuntime` on every run  

Retain Phase A behavior (Plan/Ask/Agent, review-first patches, shell approval, lab serve/eval).

---

## 2. Product decisions (locked)

| Decision | Choice |
|----------|--------|
| Auto context | Open tabs + selection only (not whole-repo, not auto-related) |
| Search engine | **ripgrep only** (no embeddings in Phase B) |
| Explicit attach | **`@` mentions** in Composer input + popup |
| Token/char budget | **Configurable** + smart priority truncation |
| Who assembles | **Client snapshot** each run; server validates, re-reads, packs |
| Architecture | **B1 — Context pack module** under `packages/backend/src/agent/context/` |
| Injection style | Context as **system/context message blocks**, not fake tool results |

---

## 3. Architecture

```text
apps/web Workbench
  @ parser + mention popup
  editorSnapshot = { openFiles, activePath, selection, mentions }
        │
        ▼  POST /api/agent/run  { message, mode, editorSnapshot }
LabController
  runAgent → startAgentRun
        │
        ▼
  ContextPacker (context/*)
    validate paths · load files · rg search · ignore · budget
        │
        ▼
  AgentRuntime
    systemPrompt(mode) + packer.systemExtra + packer.contextMessage
    + history (budget-aware) + user message
```

### Module map

| Path | Responsibility |
|------|----------------|
| `agent/context/types.ts` | `EditorSnapshot`, `ContextChunk`, `ContextPack`, mention types |
| `agent/context/mentions.ts` | Parse `@file` / `@folder` / `@search` / `@code` from text (server-side validate + optional client mirror) |
| `agent/context/ignore.ts` | Load `.gitignore` + `.lailignore`; default skips (`node_modules`, `.git`, …) |
| `agent/context/search.ts` | ripgrep workspace search; soft-fail if `rg` missing |
| `agent/context/budget.ts` | Priority queue + truncation to `contextBudgetChars` |
| `agent/context/packer.ts` | Orchestrate → `ContextPack` |
| `agent/context/index.ts` | Public `buildContextPack(opts)` replacing thin Phase A `buildContext` for runtime |
| `apps/web/...` | Mention UI, snapshot builder, context chips, Configure budget field |

Phase A `agent/context.ts` becomes a thin re-export or is folded into `context/index.ts`.

---

## 4. Contracts

### 4.1 `@` syntax

| Form | Meaning |
|------|---------|
| `@file <path>` | Include file contents (server re-read) |
| `@folder <path>` | List tree + optionally include small text files under folder (budget-limited; skip ignored) |
| `@search <query>` | ripgrep hits (paths + line snippets); max N hits (e.g. 30) |
| `@code <query>` | Alias of `@search` |

Paths are workspace-relative. Popup on `@` suggests files (from workspace tree API).

**Stripping:** Mentions remain visible in the user message for the model; packer still attaches structured context so the model has content without needing tools first.

### 4.2 `EditorSnapshot` (client → server)

```ts
type EditorSnapshot = {
  openFiles: Array<{
    path: string;
    /** Optional; server prefers re-read from disk for freshness */
    content?: string;
  }>;
  activePath?: string | null;
  selection?: {
    path: string;
    startLine: number;
    endLine: number;
    text: string;
  } | null;
  mentions: Array<
    | { type: "file"; path: string }
    | { type: "folder"; path: string }
    | { type: "search"; query: string }
  >;
};
```

**Client duties:**

1. Parse `@` from composer text → `mentions` (and keep raw message).  
2. Fill `openFiles` paths (+ optional content from open tabs).  
3. Fill `selection` if the simple editor has a selection (Phase B: textarea selection range if implementable; otherwise omit until Phase C Monaco).  
4. POST with `mode` + `message` + `editorSnapshot`.

**Server duties:**

1. `assertWorkspaceRelativePath` on every path.  
2. Re-read file contents from workspace disk (do not trust client content for packing, except **selection.text** which is ephemeral).  
3. Apply ignore + size caps.  
4. Run `@search` via rg.  
5. Budget-pack and inject.

### 4.3 Priority (high → low)

When over budget, drop or truncate **lowest** first:

1. **Selection** (highest — never drop first; truncate only if absurdly large)  
2. **`@file` / `@folder` / `@search` results** (explicit user intent)  
3. **Active open file**  
4. **Other open tabs**  
5. **Chat history** window (shrink message count / length last among history)

Within a priority tier, prefer head+tail truncation for long files (keep first ~60% and last ~20% of allowed slice with a middle marker).

### 4.4 Budget settings

| Setting | Default | Notes |
|---------|---------|--------|
| `contextBudgetChars` | `32000` | Total for packed context chunks (not including mode system prompt core) |
| `contextMaxFileChars` | `200000` | Skip or hard-truncate single files above this before packing |
| `contextMaxSearchHits` | `30` | `@search` cap |

Expose `contextBudgetChars` in **Configure** UI; others may be env/settings with defaults only in Phase B.

### 4.5 Ignore

- Parse `.gitignore` and `.lailignore` at workspace root (simple glob / path rules; need not be full gitignore parity).  
- Always skip: `node_modules`, `.git`, `.venv`, `dist`, `build`, binary extensions (e.g. `.png`, `.woff`, `.sqlite`).  
- Ignored paths never enter open-tab auto context or `@folder` expansion.

### 4.6 `ContextPack` → Runtime

```ts
type ContextChunk = {
  kind: "selection" | "mention_file" | "mention_folder" | "mention_search" | "open_tab" | "note";
  path?: string;
  label: string;
  body: string;
  priority: number; // lower number = higher priority
};

type ContextPack = {
  chunks: ContextChunk[];
  truncated: boolean;
  droppedLabels: string[];
  /** Injected as additional system content after mode system prompt */
  systemExtra: string;
  /** Optional single user-role or system "context" block before history */
  contextMessage: { role: "system"; content: string } | null;
};
```

`AgentRuntime` message order:

1. Mode `systemPrompt`  
2. `systemExtra` (budget summary + rules: “context below is attached; prefer it over guessing”)  
3. `contextMessage` (concatenated chunks) if non-empty  
4. History (from existing session messages, still poison-filtered; history length may shrink under budget)  
5. Current user message  

Publish WS `status` (or `context_truncated`) when `truncated` is true so UI can show “Context truncated to budget”.

---

## 5. UI (Workbench)

1. **Composer `@` UX:** On typing `@`, show filtered file/folder suggestions; Tab/Enter inserts `@file path` (or `@folder`). For search, insert `@search ` and leave cursor for query.  
2. **Context chips:** Above input or under mode bar — show active mentions + “N open tabs” summary before send; after send, optional status line from server truncation event.  
3. **No change** to patch review / shell approval / modes from Phase A.  
4. **Configure:** number input for context budget (chars or approximate “k tokens” label with chars under the hood).

---

## 6. Data flow

```text
User composes message with @file foo.ts
  → client parseMentions + snapshot
  → POST /api/agent/run
  → runAgent validates session/workspace
  → startAgentRun({ ..., editorSnapshot })
  → buildContextPack({ workspaceId, rootPath, snapshot, budget })
  → LLM stream with packed context
  → tools/patches as Phase A
```

`app.ts` / `runAgent` accept optional `editorSnapshot` (default empty → packer only history + empty auto tabs).

---

## 7. Errors & edge cases

| Case | Behavior |
|------|----------|
| `@file` missing | Chunk `kind: "note"` “File not found: …”; run continues |
| Path escape | Drop chunk; never read outside workspace |
| Binary / huge file | Skip with note |
| `rg` not installed | `@search` note “ripgrep unavailable”; no crash |
| Empty snapshot | Phase A-like history only |
| Over budget | Drop low priority; `truncated: true`; status event |
| Folder with thousands of files | Cap entries (e.g. 100 files listed); include only small text files under remaining budget |

---

## 8. Testing

| Layer | Cases |
|-------|--------|
| **Unit** | mention parse; ignore match; budget priority order; head-tail truncate |
| **Packer integration** | temp workspace files; snapshot with open tab + @file; assert pack includes content; assert budget drops open tab before @file |
| **Runtime** | mock LLM; pass snapshot; assert messages include packed file body |
| **UI contract** | ide-chrome / shell-source: `@` affordance strings, context budget label if in Configure |
| **Lab** | no serve-engine changes required |

---

## 9. Definition of done

1. Typing `@file path/to.ts` and sending includes that file’s content in the model context (observable via packer unit test + optional debug status).  
2. Open editor tabs are auto-included without `@`.  
3. Over-budget packs truncate lower-priority chunks first.  
4. `@search` returns rg snippets when `rg` is available.  
5. Configure can change `contextBudgetChars`.  
6. Phase A: modes, patches Accept/Reject, shell approval, Server/Models/Usage still work.  

---

## 10. Non-goals (Phase B)

- Local embeddings / vector index  
- Auto “related files” from the user message  
- Full gitignore engine parity  
- Monaco selection UX (Phase C) — selection included when cheaply available from current editor  
- Changing patch/tool policy  
- Desktop app  

---

## 11. Key decisions

1. **Client snapshot + server re-read** — browser IDE has truth for open tabs; disk is truth for file body.  
2. **Pack into system/context, not tools** — local models get context even if they under-call tools.  
3. **rg-only search** — YAGNI vs embeddings until proven necessary.  
4. **Minimal auto context** — open tabs + selection; explicit `@` for more.  
5. **Char budget with priority** — tunable for small vs large local windows.  

---

## 12. Open questions

None blocking. Defer embeddings and Monaco selection polish to later phases.

---

## 13. Implementation note

Next: **writing-plans** → `docs/superpowers/plans/2026-07-16-lail-phase-b-context-engine.md`, then subagent-driven or inline execution. No application code in this brainstorm step beyond this doc.
