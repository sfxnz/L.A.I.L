# L.A.I.L Phase B — Context Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Cursor-style `@file` / `@folder` / `@search` mentions, open-tab + selection auto context, rg-backed search, ignore rules, and configurable budget packing into AgentRuntime — without embeddings and without breaking Phase A agent/lab features.

**Architecture:** Client builds `EditorSnapshot` each run; server `ContextPacker` validates paths, re-reads disk, runs rg for `@search`, applies ignore + priority budget, injects system/context blocks before history. Modules live under `packages/backend/src/agent/context/`. Workbench adds `@` popup, context chips, and Configure budget field.

**Tech Stack:** Bun + Hono + existing agent platform · ripgrep CLI · Next.js Workbench · bun:test · SQLite settings (LabSettings)

**Spec:** `docs/superpowers/specs/2026-07-16-lail-phase-b-context-engine-design.md`

**Always:** `export PATH="$HOME/.bun/bin:$PATH"` before bun commands. Work from `/home/sfxnz/projects/ai-lab/local-ai-lab` on branch `feat/phase-a-agent-platform` (or a new `feat/phase-b-context` branched from it).

---

## File map

| Path | Responsibility |
|------|----------------|
| `packages/shared/src/index.ts` | `EditorSnapshot`, context settings fields on `LabSettings`, optional context event types |
| `packages/backend/src/agent/context/types.ts` | Backend chunk/pack types (or re-export shared) |
| `packages/backend/src/agent/context/mentions.ts` | Parse `@file` / `@folder` / `@search` / `@code` from message text |
| `packages/backend/src/agent/context/mentions.test.ts` | Mention parse tests |
| `packages/backend/src/agent/context/ignore.ts` | gitignore/lailignore + default denylist |
| `packages/backend/src/agent/context/ignore.test.ts` | Ignore tests |
| `packages/backend/src/agent/context/budget.ts` | Priority pack + head/tail truncate |
| `packages/backend/src/agent/context/budget.test.ts` | Budget tests |
| `packages/backend/src/agent/context/search.ts` | ripgrep search |
| `packages/backend/src/agent/context/search.test.ts` | Search tests (skip if no rg with note) |
| `packages/backend/src/agent/context/packer.ts` | Orchestrate pack |
| `packages/backend/src/agent/context/packer.test.ts` | Temp workspace integration |
| `packages/backend/src/agent/context/index.ts` | `buildContextPack` public API |
| `packages/backend/src/agent/context.ts` | Re-export packer; keep history helper or move to `history.ts` |
| `packages/backend/src/agent/runtime.ts` | Accept `editorSnapshot`; inject pack |
| `packages/backend/src/controller/agent.ts` | Pass snapshot through |
| `packages/backend/src/controller/settings.ts` | `contextBudgetChars` defaults |
| `packages/backend/src/app.ts` | Body `editorSnapshot` |
| `apps/web/lib/api.ts` | Types + `agentRun` snapshot arg |
| `apps/web/lib/mentions.ts` | Client parse (mirror server rules) |
| `apps/web/lib/mentions.test.ts` | Client parse tests |
| `apps/web/lib/store.ts` | Context chips state if needed |
| `apps/web/components/workbench/MentionPopup.tsx` | `@` suggestions |
| `apps/web/components/workbench/ContextChips.tsx` | Show mentions / open tabs summary |
| `apps/web/app/workbench/page.tsx` | Wire parse, snapshot, popup, chips |
| `apps/web/app/configure/page.tsx` | Budget field |
| `apps/web/lib/ide-chrome.ts` | Context UI labels if any |
| `README.md` / `CHECKPOINT.md` | Phase B docs |

---

### Task 1: Shared types — EditorSnapshot + settings

**Files:**
- Modify: `packages/shared/src/index.ts`

- [ ] **Step 1: Extend LabSettings and add snapshot types**

```ts
// Add to LabSettings:
export type LabSettings = {
  defaultBackend: BackendKind;
  defaultModel: string;
  backends: Record<BackendKind, { url: string; enabled: boolean; label: string }>;
  hfToken?: string;
  /** Max chars for packed context chunks (default 32000) */
  contextBudgetChars?: number;
  contextMaxFileChars?: number;
  contextMaxSearchHits?: number;
};

export type ContextMention =
  | { type: "file"; path: string }
  | { type: "folder"; path: string }
  | { type: "search"; query: string };

export type EditorSelection = {
  path: string;
  startLine: number;
  endLine: number;
  text: string;
};

export type EditorSnapshot = {
  openFiles: Array<{ path: string; content?: string }>;
  activePath?: string | null;
  selection?: EditorSelection | null;
  mentions: ContextMention[];
};

// Optional AgentEvent:
// | { type: "context_truncated"; runId: string; dropped: string[] }
```

- [ ] **Step 2: Commit**

```bash
git add packages/shared/src/index.ts
git commit -m "feat(shared): EditorSnapshot and context budget settings types"
```

---

### Task 2: Settings defaults for context budget

**Files:**
- Modify: `packages/backend/src/controller/settings.ts`

- [ ] **Step 1: Defaults + sanitize**

In `defaults()`:

```ts
contextBudgetChars: 32_000,
contextMaxFileChars: 200_000,
contextMaxSearchHits: 30,
```

In `sanitize` / `putSettings` merge: coerce numbers with `Number.isFinite` and clamp budget to min 2000, max 500_000.

- [ ] **Step 2: Quick smoke**

```bash
cd packages/backend
bun -e "import { getSettings } from './src/controller/settings.ts'; console.log(getSettings().contextBudgetChars)"
```

Expected: `32000` (or stored value).

- [ ] **Step 3: Commit** `feat(settings): context budget defaults`

---

### Task 3: Mention parser — TDD

**Files:**
- Create: `packages/backend/src/agent/context/mentions.ts`
- Create: `packages/backend/src/agent/context/mentions.test.ts`

- [ ] **Step 1: Failing tests**

```ts
import { describe, expect, test } from "bun:test";
import { parseMentions } from "./mentions";

describe("parseMentions", () => {
  test("parses @file path", () => {
    expect(parseMentions("see @file src/a.ts please")).toEqual([
      { type: "file", path: "src/a.ts" },
    ]);
  });
  test("parses @folder", () => {
    expect(parseMentions("@folder packages/backend")).toEqual([
      { type: "folder", path: "packages/backend" },
    ]);
  });
  test("parses @search and @code alias", () => {
    expect(parseMentions('@search "foo bar" and @code baz')).toEqual([
      { type: "search", query: "foo bar" },
      { type: "search", query: "baz" },
    ]);
  });
  test("dedupes identical mentions", () => {
    const m = parseMentions("@file a.ts @file a.ts");
    expect(m).toEqual([{ type: "file", path: "a.ts" }]);
  });
});
```

- [ ] **Step 2: Implement**

```ts
import type { ContextMention } from "@lail/shared";

/** Parse @file/@folder/@search/@code from composer text. */
export function parseMentions(text: string): ContextMention[] {
  const out: ContextMention[] = [];
  const seen = new Set<string>();
  const add = (m: ContextMention) => {
    const key =
      m.type === "search" ? `search:${m.query}` : `${m.type}:${m.path}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(m);
  };

  // @file path  |  @folder path
  const fileFolder =
    /@(file|folder)\s+([^\s@]+)/gi;
  let m: RegExpExecArray | null;
  while ((m = fileFolder.exec(text))) {
    add({ type: m[1].toLowerCase() as "file" | "folder", path: m[2] });
  }

  // @search "query" or @search query | @code same
  const search =
    /@(search|code)\s+(?:"([^"]+)"|'([^']+)'|([^\s@]+))/gi;
  while ((m = search.exec(text))) {
    const q = (m[2] ?? m[3] ?? m[4] ?? "").trim();
    if (q) add({ type: "search", query: q });
  }
  return out;
}
```

- [ ] **Step 3: `bun test src/agent/context/mentions.test.ts` — PASS**
- [ ] **Step 4: Commit** `feat(context): parse @file @folder @search mentions`

---

### Task 4: Ignore rules — TDD

**Files:**
- Create: `packages/backend/src/agent/context/ignore.ts`
- Create: `packages/backend/src/agent/context/ignore.test.ts`

- [ ] **Step 1: Tests**

```ts
import { describe, expect, test, beforeEach, afterEach } from "bun:test";
import { mkdirSync, writeFileSync, rmSync } from "fs";
import { join } from "path";
import { loadIgnore, isIgnored } from "./ignore";

const TMP = `/tmp/lail-ignore-${process.pid}`;

describe("ignore", () => {
  beforeEach(() => {
    rmSync(TMP, { recursive: true, force: true });
    mkdirSync(TMP, { recursive: true });
    writeFileSync(join(TMP, ".gitignore"), "dist/\n*.log\n");
    writeFileSync(join(TMP, ".lailignore"), "secrets/\n");
  });
  afterEach(() => rmSync(TMP, { recursive: true, force: true }));

  test("default skips node_modules and .git", () => {
    const ig = loadIgnore(TMP);
    expect(isIgnored(ig, "node_modules/x")).toBe(true);
    expect(isIgnored(ig, ".git/config")).toBe(true);
    expect(isIgnored(ig, "src/a.ts")).toBe(false);
  });

  test("respects gitignore and lailignore", () => {
    const ig = loadIgnore(TMP);
    expect(isIgnored(ig, "dist/out.js")).toBe(true);
    expect(isIgnored(ig, "foo.log")).toBe(true);
    expect(isIgnored(ig, "secrets/key")).toBe(true);
  });
});
```

- [ ] **Step 2: Implement simple matcher**

YAGNI: support trailing `/` for dirs, `*.ext` globs, exact path prefixes. Defaults always applied:

```ts
const DEFAULTS = ["node_modules/", ".git/", ".venv/", "dist/", "build/"];
const BINARY_EXT = new Set([".png", ".jpg", ".jpeg", ".gif", ".woff", ".woff2", ".sqlite", ".wasm", ".pdf"]);

export type IgnoreSet = { patterns: string[] };

export function loadIgnore(rootPath: string): IgnoreSet {
  // read .gitignore and .lailignore lines, skip comments/empty
  // return { patterns: [...DEFAULTS, ...fromFiles] }
}

export function isIgnored(ig: IgnoreSet, relPath: string): boolean {
  const p = relPath.replace(/\\/g, "/").replace(/^\.\//, "");
  const ext = p.includes(".") ? p.slice(p.lastIndexOf(".")) : "";
  if (BINARY_EXT.has(ext.toLowerCase())) return true;
  for (const pat of ig.patterns) {
    // implement prefix / suffix *.ext / dir/ matching
  }
  return false;
}
```

- [ ] **Step 3: Tests PASS · Commit** `feat(context): gitignore and lailignore for context pack`

---

### Task 5: Budget packer — TDD

**Files:**
- Create: `packages/backend/src/agent/context/budget.ts`
- Create: `packages/backend/src/agent/context/budget.test.ts`
- Create: `packages/backend/src/agent/context/types.ts` (ContextChunk, priorities)

- [ ] **Step 1: Types**

```ts
export type ContextChunk = {
  kind: "selection" | "mention_file" | "mention_folder" | "mention_search" | "open_tab" | "note";
  path?: string;
  label: string;
  body: string;
  /** lower = higher priority */
  priority: number;
};

export const PRIORITY = {
  selection: 10,
  mention: 20,
  active_tab: 30,
  open_tab: 40,
  note: 50,
} as const;
```

- [ ] **Step 2: Tests for `applyBudget(chunks, budgetChars)`**

```ts
test("keeps high priority when over budget", () => {
  const chunks = [
    { kind: "open_tab", label: "tab", body: "x".repeat(1000), priority: 40 },
    { kind: "mention_file", label: "m", body: "y".repeat(100), priority: 20 },
  ];
  const r = applyBudget(chunks, 200);
  expect(r.chunks.some((c) => c.kind === "mention_file")).toBe(true);
  expect(r.truncated).toBe(true);
});

test("head-tail truncates long body", () => {
  const r = truncateBody("a".repeat(1000), 100);
  expect(r.length).toBeLessThanOrEqual(120); // marker overhead
  expect(r.includes("…")).toBe(true);
});
```

- [ ] **Step 3: Implement `applyBudget` + `truncateBody`**
  - Sort by priority ascending  
  - Greedily take chunks until budget; for partial last chunk use truncateBody  
  - Return `{ chunks, truncated, droppedLabels }`

- [ ] **Step 4: PASS · Commit** `feat(context): priority budget and truncation`

---

### Task 6: ripgrep search

**Files:**
- Create: `packages/backend/src/agent/context/search.ts`
- Create: `packages/backend/src/agent/context/search.test.ts`

- [ ] **Step 1: Implement**

```ts
export async function ripgrepSearch(opts: {
  rootPath: string;
  query: string;
  maxHits?: number;
}): Promise<{ ok: boolean; output: string; hits: number }> {
  const max = opts.maxHits ?? 30;
  try {
    const proc = Bun.spawn(
      ["rg", "-n", "--max-count", "5", "--max-filesize", "200K", "-m", String(max), opts.query, "."],
      { cwd: opts.rootPath, stdout: "pipe", stderr: "pipe" },
    );
    const out = await new Response(proc.stdout).text();
    await proc.exited;
    if (proc.exitCode === 1 && !out) return { ok: true, output: "(no matches)", hits: 0 };
    if (proc.exitCode !== 0 && proc.exitCode !== 1) {
      return { ok: false, output: "ripgrep failed or unavailable", hits: 0 };
    }
    const lines = out.split("\n").filter(Boolean).slice(0, max);
    return { ok: true, output: lines.join("\n") || "(no matches)", hits: lines.length };
  } catch {
    return { ok: false, output: "ripgrep unavailable", hits: 0 };
  }
}
```

- [ ] **Step 2: Test with temp dir + file containing unique string** (skip soft if rg missing: `if (!Bun.which("rg")) return;`)

- [ ] **Step 3: Commit** `feat(context): ripgrep workspace search for @search`

---

### Task 7: ContextPacker orchestration — TDD

**Files:**
- Create: `packages/backend/src/agent/context/packer.ts`
- Create: `packages/backend/src/agent/context/packer.test.ts`
- Create: `packages/backend/src/agent/context/index.ts`
- Modify: `packages/backend/src/agent/context.ts` → re-export history + pack

- [ ] **Step 1: API**

```ts
import type { EditorSnapshot } from "@lail/shared";
import type { ContextChunk } from "./types";

export type ContextPack = {
  chunks: ContextChunk[];
  truncated: boolean;
  droppedLabels: string[];
  systemExtra: string;
  contextMessage: { role: "system"; content: string } | null;
};

export async function buildContextPack(opts: {
  rootPath: string;
  snapshot: EditorSnapshot;
  budgetChars: number;
  maxFileChars?: number;
  maxSearchHits?: number;
  /** merge client mentions with parseMentions(message) */
  message?: string;
}): Promise<ContextPack>
```

Algorithm:

1. `loadIgnore(rootPath)`  
2. Mentions = dedupe(snapshot.mentions + parseMentions(message||""))  
3. If selection → chunk priority selection (use selection.text)  
4. For each file mention: resolve path, if missing note chunk; else read (cap maxFileChars), ignore check  
5. For each folder mention: list up to 100 non-ignored entries; include small files if budget allows  
6. For each search mention: ripgrepSearch → mention_search chunk  
7. For openFiles: if not ignored, re-read; activePath gets PRIORITY.active_tab else open_tab  
8. `applyBudget`  
9. Format `contextMessage.content` as markdown sections `# path\n```\nbody\n```  
10. `systemExtra` = short rules + "Attached N context chunks" + truncation note  

- [ ] **Step 2: Integration test**

```ts
// temp ws with a.txt "hello-unique"
// snapshot openFiles: [{path: "a.txt"}], mentions: [{type:"file", path:"a.txt"}]
// pack = await buildContextPack(...)
// expect contextMessage.content includes "hello-unique"
// budget tiny → truncated true when many open tabs
```

- [ ] **Step 3: history helper**

Keep poison-filtered last N messages as `loadHistory(sessionId)` in `context.ts` or `history.ts`.

- [ ] **Step 4: Commit** `feat(context): ContextPacker builds budgeted packs from snapshot`

---

### Task 8: Wire AgentRuntime + API

**Files:**
- Modify: `packages/backend/src/agent/runtime.ts`
- Modify: `packages/backend/src/controller/agent.ts`
- Modify: `packages/backend/src/app.ts`
- Modify: `packages/backend/src/agent/runtime.test.ts` (add snapshot assertion)

- [ ] **Step 1: StartRunOpts**

```ts
import type { EditorSnapshot } from "@lail/shared";

export type StartRunOpts = {
  // ...existing
  editorSnapshot?: EditorSnapshot;
};
```

- [ ] **Step 2: After model resolve, before loop**

```ts
import { buildContextPack } from "./context";
import { getSettings } from "../controller/settings";

const settings = getSettings();
const pack = await buildContextPack({
  rootPath: ws.rootPath,
  snapshot: opts.editorSnapshot ?? { openFiles: [], mentions: [] },
  budgetChars: settings.contextBudgetChars ?? 32_000,
  maxFileChars: settings.contextMaxFileChars,
  maxSearchHits: settings.contextMaxSearchHits,
  message: opts.message,
});

if (pack.truncated) {
  publish(runId, {
    type: "status",
    text: `Context truncated to budget (${pack.droppedLabels.length} lower-priority chunks dropped)`,
  });
  // optional: type context_truncated
}

const history = await loadHistory(sessionId); // former buildContext history part
const messages: ChatMsg[] = [
  { role: "system", content: systemPrompt(mode, ws.rootPath) + "\n\n" + pack.systemExtra },
];
if (pack.contextMessage) messages.push(pack.contextMessage);
messages.push(...history);
// ensure user message
```

- [ ] **Step 3: Facade + app pass editorSnapshot**

```ts
// agent.ts runAgent opts.editorSnapshot
// app.ts body.editorSnapshot
```

- [ ] **Step 4: Runtime test** — mock LLM; pass snapshot with file; assert fetch body messages system/context includes file text (inspect JSON in mock fetchImpl).

- [ ] **Step 5: `bun test src/agent` PASS · Commit** `feat(agent): inject ContextPack into AgentRuntime`

---

### Task 9: Web API + client mention parse

**Files:**
- Modify: `apps/web/lib/api.ts`
- Create: `apps/web/lib/mentions.ts`
- Create: `apps/web/lib/mentions.test.ts`

- [ ] **Step 1: api.agentRun**

```ts
agentRun: (
  sessionId: string,
  message: string,
  workspaceId?: string,
  mode: AgentMode = "agent",
  editorSnapshot?: EditorSnapshot,
) =>
  req<{ runId: string }>("/api/agent/run", {
    method: "POST",
    body: JSON.stringify({ sessionId, message, workspaceId, mode, editorSnapshot }),
  }),
```

Export `EditorSnapshot`, `ContextMention` types (mirror shared or import if web resolves shared — currently web may duplicate; prefer copy types in api.ts matching shared).

- [ ] **Step 2: Client `parseMentions`** — same regex as server (or shared package import if web can import `@lail/shared` — check package.json; if not, duplicate in `lib/mentions.ts`).

- [ ] **Step 3: `cd apps/web && bun test lib/mentions.test.ts`**  
- [ ] **Step 4: Commit** `feat(web): EditorSnapshot on agentRun and client mention parse`

---

### Task 10: Workbench UI — @ popup, chips, snapshot send

**Files:**
- Create: `apps/web/components/workbench/MentionPopup.tsx`
- Create: `apps/web/components/workbench/ContextChips.tsx`
- Modify: `apps/web/app/workbench/page.tsx`

- [ ] **Step 1: MentionPopup**

When input contains `@` at end or incomplete token, show list from `api.workspaces.tree` flattened paths (filter by suffix after `@`). On select, insert `@file path ` into textarea.

Minimal UX:

- Detect last `@` without space after  
- Filter tree paths by query string after `@`  
- Keyboard: ArrowDown/Up/Enter optional; mouse click required minimum  

- [ ] **Step 2: ContextChips**

Show chips for `parseMentions(input)` + “N open tabs”.

- [ ] **Step 3: send()**

```ts
const mentions = parseMentions(message);
// merge with any chip-only mentions if needed
const editorSnapshot: EditorSnapshot = {
  openFiles: openFiles.map((f) => ({ path: f.path, content: f.content })),
  activePath: activeFile,
  selection: null, // Phase B: optional — if textarea selection available, set it
  mentions,
};
await api.agentRun(session.id, message, workspace?.id, agentMode, editorSnapshot);
```

Selection optional enhancement: if `taRef` has selectionStart/End and active file, slice content for selection.text.

- [ ] **Step 4: WS status** — already shows status events including truncation.

- [ ] **Step 5: Manual layout check · Commit** `feat(web): Composer @mentions, context chips, snapshot on run`

---

### Task 11: Configure UI — context budget

**Files:**
- Modify: `apps/web/app/configure/page.tsx`

- [ ] **Step 1:** Load/save `contextBudgetChars` via existing configure get/put. Number input label “Context budget (chars)” default 32000, min 2000.

- [ ] **Step 2: Commit** `feat(web): Configure context budget`

---

### Task 12: UI contracts + docs

**Files:**
- Modify: `apps/web/lib/shell-source.test.ts` and/or `ide-chrome.ts`
- Modify: `README.md`, `CHECKPOINT.md`

- [ ] **Step 1: Assert workbench source contains MentionPopup or parseMentions and editorSnapshot**

- [ ] **Step 2: Document Phase B in README Workbench section**

- [ ] **Step 3: `cd apps/web && bun test` PASS**

- [ ] **Step 4: Commit** `test+docs: Phase B context engine contracts`

---

### Task 13: Full verification

- [ ] **Step 1: Automated**

```bash
export PATH="$HOME/.bun/bin:$PATH"
cd /home/sfxnz/projects/ai-lab/local-ai-lab/packages/backend && bun test
cd /home/sfxnz/projects/ai-lab/local-ai-lab/apps/web && bun test
cd /home/sfxnz/projects/ai-lab/local-ai-lab && bun run typecheck
```

Expected: all PASS.

- [ ] **Step 2: Manual checklist (when model served)**

1. Open two files as tabs; ask “what files are open?” — model sees tab context.  
2. `@file` a specific path — answer cites content.  
3. `@search UniqueString` — hits returned in pack.  
4. Lower budget in Configure to 3000; attach many large files — status mentions truncation.  
5. Phase A patch Accept still works.  

- [ ] **Step 3: Commit any fixes** `fix: Phase B verification follow-ups`

---

## Spec coverage self-review

| Spec requirement | Task |
|------------------|------|
| @file @folder @search/@code | 3, 7, 9, 10 |
| Open tabs + selection auto | 7, 10 |
| rg only | 6 |
| Ignore rules | 4 |
| Configurable budget + priority | 2, 5, 11 |
| Client snapshot + server re-read | 7, 8, 10 |
| Inject system/context not tools | 8 |
| Packer modules under context/ | 3–7 |
| Tests | 3–7, 8, 9, 12, 13 |
| Phase A retained | 8 non-regression + 13 |
| No embeddings | Not in plan |

**Type names:** `EditorSnapshot`, `ContextMention`, `ContextPack`, `buildContextPack` — consistent across tasks.

---

## Out of scope

- Embeddings / vector index  
- Auto-related files  
- Monaco (Phase C)  
- Changing patch/shell policy  

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-16-lail-phase-b-context-engine.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute in this session with checkpoints  

**Which approach?**
