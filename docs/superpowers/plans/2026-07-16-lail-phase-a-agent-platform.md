# L.A.I.L Phase A — Agent Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Cursor-style Plan/Ask/Agent modes, full token+tool streaming, review-first search/replace patches, and risky-shell approval — while keeping serve/eval lab surfaces unchanged.

**Architecture:** Extract a modular agent platform under `packages/backend/src/agent/` (`ToolPolicy`, `PatchStore`, `AgentRuntime`, event helpers) behind LabController HTTP/WS. Web Workbench gains mode toggle, patch review panel, shell approval UI, and cancel. No Monaco, no context index, no terminal panel (Phases B–E).

**Tech Stack:** Bun + Hono + bun:sqlite (controller) · Next.js 16 + React 19 + Zustand (web) · existing WS hub · vLLM/llama.cpp OpenAI-compatible `/v1` · bun:test

**Spec:** `docs/superpowers/specs/2026-07-16-lail-cursor-ide-design.md`

**Note on git:** This tree may not be a git repository yet. Before the first commit step, run `git init` (and a sensible `.gitignore` if missing) if `git status` fails. If the user forbids init, skip commit steps and continue.

---

## File map

| Path | Responsibility |
|------|----------------|
| `packages/shared/src/index.ts` | Shared types: `AgentMode`, `Patch`, extended `AgentEvent` |
| `packages/backend/src/db/schema.ts` | Migrate `agent_runs`, `patches` tables |
| `packages/backend/src/agent/types.ts` | Backend-local run/approval types |
| `packages/backend/src/agent/tool-policy.ts` | Mode allowlists + risky shell + path checks |
| `packages/backend/src/agent/tool-policy.test.ts` | Policy unit tests |
| `packages/backend/src/agent/patch-store.ts` | Propose / list / accept / reject / apply |
| `packages/backend/src/agent/patch-store.test.ts` | Patch apply unit tests |
| `packages/backend/src/agent/approvals.ts` | In-memory shell approval waiters |
| `packages/backend/src/agent/context.ts` | Thin `ContextProvider` (Phase A stub) |
| `packages/backend/src/agent/runtime.ts` | Streaming agent loop, cancel, modes |
| `packages/backend/src/agent/runtime.test.ts` | Mock-LLM integration tests |
| `packages/backend/src/agent/prompts.ts` | System prompts per mode |
| `packages/backend/src/tools/index.ts` | Tool defs + execute (no silent disk writes for edits) |
| `packages/backend/src/controller/agent.ts` | Thin facade → `AgentRuntime` (keep export `runAgent`) |
| `packages/backend/src/controller/patches.ts` | Patch list/accept/reject HTTP helpers |
| `packages/backend/src/app.ts` | New routes: cancel, shell approval, patches |
| `packages/backend/package.json` | Add `"test": "bun test"` |
| `apps/web/lib/api.ts` | Client methods for new endpoints |
| `apps/web/lib/store.ts` | `agentMode`, `pendingPatches`, helpers |
| `apps/web/lib/ide-chrome.ts` | Mode labels, patch stream markers |
| `apps/web/lib/ide-chrome.test.ts` | Contract tests for modes/patches |
| `apps/web/components/workbench/PatchReviewPanel.tsx` | Accept/Reject UI |
| `apps/web/components/workbench/ShellApprovalBanner.tsx` | Allow/Deny UI |
| `apps/web/components/workbench/ModeToggle.tsx` | Plan \| Ask \| Agent |
| `apps/web/app/workbench/page.tsx` | Wire stream events, mode, cancel, panels |
| `README.md` / `CHECKPOINT.md` | Document Phase A behavior |

**Do not modify (except accidental breakage fixes):** `packages/serve-engine/**`, Server/Models/Usage pages’ core flows, `llm-proxy.ts` metering behavior (agent still calls `recordUsage`).

---

### Task 1: Shared types — modes, patches, events

**Files:**
- Modify: `packages/shared/src/index.ts`

- [ ] **Step 1: Extend shared types**

Add (or replace the existing `AgentEvent` union carefully) these exports in `packages/shared/src/index.ts`:

```ts
export type AgentMode = "plan" | "ask" | "agent";

export type PatchOp = "replace" | "create" | "delete";

export type PatchStatus = "pending" | "accepted" | "rejected" | "failed";

export type Patch = {
  id: string;
  runId: string;
  sessionId: string;
  path: string;
  oldString: string;
  newString: string;
  op: PatchOp;
  status: PatchStatus;
  reason?: string;
  createdAt: string;
  resolvedAt?: string;
};

export type AgentRunStatus = "running" | "done" | "error" | "cancelled";

export type AgentEvent =
  | { type: "thought"; runId: string; text: string }
  | { type: "token"; runId: string; text: string; channel?: "assistant" | "thought" }
  | { type: "status"; runId: string; text: string }
  | { type: "tool_start"; runId: string; tool: string; args: Record<string, unknown> }
  | { type: "tool_end"; runId: string; tool: string; summary: string; output?: string }
  | { type: "patch_proposed"; runId: string; patch: Patch }
  | { type: "patch_updated"; runId: string; patch: Patch }
  | { type: "shell_approval_required"; runId: string; approvalId: string; command: string }
  | { type: "file_write"; runId: string; path: string; bytes: number } // after accept only
  | { type: "assistant"; runId: string; text: string; delta?: boolean }
  | { type: "done"; runId: string; usage?: { prompt: number; completion: number } }
  | { type: "cancelled"; runId: string }
  | { type: "error"; runId: string; message: string };
```

Keep existing non-agent exports (`Workspace`, `LabSettings`, etc.) unchanged.

- [ ] **Step 2: Typecheck shared + backend**

Run:

```bash
cd <repo-root>
bun run --filter @lail/backend typecheck
```

Expected: PASS (or only pre-existing errors unrelated to AgentEvent consumers — fix any breakages in backend that import old event shapes).

- [ ] **Step 3: Commit**

```bash
cd <repo-root>
git add packages/shared/src/index.ts
git commit -m "feat(shared): AgentMode, Patch, and streaming AgentEvent types"
```

---

### Task 2: Database migration — agent_runs + patches

**Files:**
- Modify: `packages/backend/src/db/schema.ts`

- [ ] **Step 1: Add tables to `migrate()`**

Inside the existing `database.exec(\`...\`)` block in `packages/backend/src/db/schema.ts`, append:

```sql
CREATE TABLE IF NOT EXISTS agent_runs (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  workspace_id TEXT NOT NULL,
  mode TEXT NOT NULL,
  status TEXT NOT NULL,
  message TEXT NOT NULL,
  error TEXT,
  prompt_tokens INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS patches (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  path TEXT NOT NULL,
  old_string TEXT NOT NULL,
  new_string TEXT NOT NULL,
  op TEXT NOT NULL,
  status TEXT NOT NULL,
  reason TEXT,
  created_at TEXT NOT NULL,
  resolved_at TEXT,
  FOREIGN KEY (run_id) REFERENCES agent_runs(id),
  FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_patches_session_status ON patches(session_id, status);
CREATE INDEX IF NOT EXISTS idx_patches_run ON patches(run_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_session ON agent_runs(session_id);
```

- [ ] **Step 2: Smoke open DB**

Run:

```bash
cd <repo-root>/packages/backend
bun -e "import { getDb } from './src/db/schema.ts'; getDb(); console.log('ok')"
```

Expected: prints `ok` with no throw.

- [ ] **Step 3: Commit**

```bash
git add packages/backend/src/db/schema.ts
git commit -m "feat(db): agent_runs and patches tables for Phase A"
```

---

### Task 3: ToolPolicy — TDD

**Files:**
- Create: `packages/backend/src/agent/tool-policy.ts`
- Create: `packages/backend/src/agent/tool-policy.test.ts`
- Create: `packages/backend/src/agent/types.ts`

- [ ] **Step 1: Write failing tests**

Create `packages/backend/src/agent/types.ts`:

```ts
import type { AgentMode } from "@lail/shared";

export type { AgentMode };

export const READ_TOOLS = ["list_dir", "read_file", "grep", "plan"] as const;
export const ASK_TOOLS = ["list_dir", "read_file", "grep"] as const;
export const AGENT_TOOLS = [
  "list_dir",
  "read_file",
  "grep",
  "plan",
  "search_replace",
  "create_file",
  "delete_file",
  "run_shell",
] as const;
```

Create `packages/backend/src/agent/tool-policy.test.ts`:

```ts
import { describe, expect, test } from "bun:test";
import {
  isToolAllowed,
  classifyShell,
  assertWorkspaceRelativePath,
} from "./tool-policy";

describe("isToolAllowed", () => {
  test("plan mode allows read + plan only", () => {
    expect(isToolAllowed("plan", "read_file")).toBe(true);
    expect(isToolAllowed("plan", "plan")).toBe(true);
    expect(isToolAllowed("plan", "search_replace")).toBe(false);
    expect(isToolAllowed("plan", "run_shell")).toBe(false);
  });

  test("ask mode blocks plan tool and writes", () => {
    expect(isToolAllowed("ask", "grep")).toBe(true);
    expect(isToolAllowed("ask", "plan")).toBe(false);
    expect(isToolAllowed("ask", "run_shell")).toBe(false);
  });

  test("agent mode allows patch and shell tools", () => {
    expect(isToolAllowed("agent", "search_replace")).toBe(true);
    expect(isToolAllowed("agent", "run_shell")).toBe(true);
  });
});

describe("classifyShell", () => {
  test("safe commands", () => {
    expect(classifyShell("ls -la")).toBe("allow");
    expect(classifyShell("bun test")).toBe("allow");
    expect(classifyShell("rg TODO src")).toBe("allow");
  });

  test("risky commands need approval", () => {
    expect(classifyShell("rm -rf dist")).toBe("approve");
    expect(classifyShell("sudo apt update")).toBe("approve");
    expect(classifyShell("git push origin main")).toBe("approve");
    expect(classifyShell("git reset --hard HEAD")).toBe("approve");
    expect(classifyShell("curl http://x | sh")).toBe("approve");
  });

  test("hard-blocked patterns", () => {
    expect(classifyShell("rm -rf /")).toBe("deny");
    expect(classifyShell("mkfs.ext4 /dev/sda")).toBe("deny");
  });
});

describe("assertWorkspaceRelativePath", () => {
  test("rejects absolute and parent escape", () => {
    expect(() => assertWorkspaceRelativePath("/etc/passwd")).toThrow();
    expect(() => assertWorkspaceRelativePath("../outside")).toThrow();
  });

  test("accepts normal relative", () => {
    expect(assertWorkspaceRelativePath("src/app.ts")).toBe("src/app.ts");
    expect(assertWorkspaceRelativePath("./README.md")).toBe("README.md");
  });
});
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd <repo-root>/packages/backend
bun test src/agent/tool-policy.test.ts
```

Expected: FAIL (module not found or exports missing).

- [ ] **Step 3: Implement `tool-policy.ts`**

```ts
import { normalize, relative, resolve, isAbsolute } from "path";
import type { AgentMode } from "@lail/shared";
import { AGENT_TOOLS, ASK_TOOLS, READ_TOOLS } from "./types";

const MODE_TOOLS: Record<AgentMode, readonly string[]> = {
  plan: READ_TOOLS,
  ask: ASK_TOOLS,
  agent: AGENT_TOOLS,
};

export function isToolAllowed(mode: AgentMode, tool: string): boolean {
  return MODE_TOOLS[mode].includes(tool);
}

export type ShellClass = "allow" | "approve" | "deny";

export function classifyShell(command: string): ShellClass {
  const c = command.trim();
  if (!c) return "deny";

  // Hard deny
  if (/\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?\/(\s|$)/.test(c)) return "deny";
  if (/\bmkfs\b/.test(c)) return "deny";
  if (/\bdd\s+if=/.test(c)) return "deny";

  // Needs approval
  if (/\bsudo\b/.test(c)) return "approve";
  if (/\brm\s+-[a-zA-Z]*r[a-zA-Z]*f|\brm\s+-[a-zA-Z]*f[a-zA-Z]*r/.test(c)) return "approve";
  if (/\bgit\s+push\b/.test(c)) return "approve";
  if (/\bgit\s+reset\s+--hard\b/.test(c)) return "approve";
  if (/curl\b.*\|\s*(ba)?sh/.test(c) || /wget\b.*\|\s*(ba)?sh/.test(c)) return "approve";
  if (/\bchmod\s+-R\s+777\b/.test(c)) return "approve";

  return "allow";
}

/** Normalize and ensure path is workspace-relative (no abs, no .. escape). */
export function assertWorkspaceRelativePath(path: string): string {
  const raw = String(path || "").trim();
  if (!raw) throw Object.assign(new Error("Empty path"), { code: "PATH_EMPTY" });
  if (isAbsolute(raw)) {
    throw Object.assign(new Error("Absolute paths not allowed"), { code: "PATH_ESCAPE" });
  }
  const norm = normalize(raw).replace(/^\.\/+/, "");
  if (norm === ".." || norm.startsWith("../") || norm.includes("/../")) {
    throw Object.assign(new Error("Path escapes workspace"), { code: "PATH_ESCAPE" });
  }
  // resolve against fake root to detect remaining escapes
  const fakeRoot = "/__ws__";
  const abs = resolve(fakeRoot, norm);
  const rel = relative(fakeRoot, abs);
  if (rel.startsWith("..") || isAbsolute(rel)) {
    throw Object.assign(new Error("Path escapes workspace"), { code: "PATH_ESCAPE" });
  }
  return rel.split("\\").join("/"); // windows safety
}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd <repo-root>/packages/backend
bun test src/agent/tool-policy.test.ts
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/backend/src/agent/
git commit -m "feat(agent): ToolPolicy for modes, risky shell, path sandbox"
```

---

### Task 4: PatchStore — TDD

**Files:**
- Create: `packages/backend/src/agent/patch-store.ts`
- Create: `packages/backend/src/agent/patch-store.test.ts`

- [ ] **Step 1: Write failing tests**

```ts
import { describe, expect, test, beforeEach, afterEach } from "bun:test";
import { mkdirSync, writeFileSync, readFileSync, rmSync, existsSync } from "fs";
import { join } from "path";
import { Database } from "bun:sqlite";

// Test against a temp workspace + inject db helpers.
// Implementation should export pure apply helpers + store methods.

import {
  applySearchReplaceToContent,
  createPatchStore,
} from "./patch-store";

const TMP = `/tmp/lail-patch-test-${process.pid}`;

describe("applySearchReplaceToContent", () => {
  test("replaces exactly once", () => {
    const r = applySearchReplaceToContent("hello world", "world", "there");
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.content).toBe("hello there");
  });

  test("fails on zero matches", () => {
    const r = applySearchReplaceToContent("abc", "zzz", "q");
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe("no_match");
  });

  test("fails on ambiguous matches", () => {
    const r = applySearchReplaceToContent("aa aa", "aa", "b");
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe("ambiguous");
  });
});

describe("PatchStore accept", () => {
  beforeEach(() => {
    rmSync(TMP, { recursive: true, force: true });
    mkdirSync(TMP, { recursive: true });
    writeFileSync(join(TMP, "a.txt"), "line1\nfoo\nline3\n");
  });
  afterEach(() => {
    rmSync(TMP, { recursive: true, force: true });
  });

  test("propose then accept writes disk", () => {
    // Use real getDb() only if tests can isolate — prefer createPatchStore with rootPath override.
    const store = createPatchStore();
    const patch = store.propose({
      runId: "run1",
      sessionId: "sess1",
      path: "a.txt",
      oldString: "foo",
      newString: "bar",
      op: "replace",
      workspaceRoot: TMP,
    });
    expect(patch.status).toBe("pending");
    const applied = store.accept(patch.id, TMP);
    expect(applied.status).toBe("accepted");
    expect(readFileSync(join(TMP, "a.txt"), "utf8")).toContain("bar");
  });

  test("reject leaves file unchanged", () => {
    const store = createPatchStore();
    const patch = store.propose({
      runId: "run1",
      sessionId: "sess1",
      path: "a.txt",
      oldString: "foo",
      newString: "bar",
      op: "replace",
      workspaceRoot: TMP,
    });
    store.reject(patch.id);
    expect(readFileSync(join(TMP, "a.txt"), "utf8")).toContain("foo");
  });
});
```

- [ ] **Step 2: Run — expect FAIL**

```bash
cd <repo-root>/packages/backend
bun test src/agent/patch-store.test.ts
```

- [ ] **Step 3: Implement `patch-store.ts`**

Implement at minimum:

```ts
import { randomUUID } from "crypto";
import { existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from "fs";
import { dirname, join } from "path";
import type { Patch, PatchOp, PatchStatus } from "@lail/shared";
import { getDb } from "../db/schema";
import { assertWorkspaceRelativePath } from "./tool-policy";

export function applySearchReplaceToContent(
  content: string,
  oldString: string,
  newString: string,
): { ok: true; content: string } | { ok: false; reason: "no_match" | "ambiguous" } {
  if (oldString === "") {
    return { ok: false, reason: "no_match" };
  }
  let count = 0;
  let idx = 0;
  while (true) {
    const found = content.indexOf(oldString, idx);
    if (found === -1) break;
    count++;
    idx = found + oldString.length;
    if (count > 1) return { ok: false, reason: "ambiguous" };
  }
  if (count === 0) return { ok: false, reason: "no_match" };
  return { ok: true, content: content.replace(oldString, newString) };
}

function rowToPatch(r: Record<string, unknown>): Patch {
  return {
    id: String(r.id),
    runId: String(r.run_id),
    sessionId: String(r.session_id),
    path: String(r.path),
    oldString: String(r.old_string),
    newString: String(r.new_string),
    op: String(r.op) as PatchOp,
    status: String(r.status) as PatchStatus,
    reason: r.reason != null ? String(r.reason) : undefined,
    createdAt: String(r.created_at),
    resolvedAt: r.resolved_at != null ? String(r.resolved_at) : undefined,
  };
}

export type ProposeInput = {
  runId: string;
  sessionId: string;
  path: string;
  oldString: string;
  newString: string;
  op: PatchOp;
  workspaceRoot?: string; // unused at propose; validate path only
};

export function createPatchStore() {
  return {
    propose(input: ProposeInput): Patch {
      const path = assertWorkspaceRelativePath(input.path);
      const id = randomUUID();
      const ts = new Date().toISOString();
      getDb()
        .query(
          `INSERT INTO patches
           (id, run_id, session_id, path, old_string, new_string, op, status, reason, created_at, resolved_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL, ?, NULL)`,
        )
        .run(
          id,
          input.runId,
          input.sessionId,
          path,
          input.oldString,
          input.newString,
          input.op,
          ts,
        );
      return this.get(id)!;
    },

    get(id: string): Patch | null {
      const row = getDb().query("SELECT * FROM patches WHERE id = ?").get(id) as
        | Record<string, unknown>
        | null;
      return row ? rowToPatch(row) : null;
    },

    list(opts: { sessionId?: string; runId?: string; status?: PatchStatus }): Patch[] {
      // Build simple filtered query — implement with optional WHERE clauses
      let sql = "SELECT * FROM patches WHERE 1=1";
      const params: string[] = [];
      if (opts.sessionId) {
        sql += " AND session_id = ?";
        params.push(opts.sessionId);
      }
      if (opts.runId) {
        sql += " AND run_id = ?";
        params.push(opts.runId);
      }
      if (opts.status) {
        sql += " AND status = ?";
        params.push(opts.status);
      }
      sql += " ORDER BY created_at ASC";
      const rows = getDb().query(sql).all(...params) as Record<string, unknown>[];
      return rows.map(rowToPatch);
    },

    reject(id: string): Patch {
      const ts = new Date().toISOString();
      getDb()
        .query(
          `UPDATE patches SET status = 'rejected', resolved_at = ? WHERE id = ? AND status = 'pending'`,
        )
        .run(ts, id);
      return this.get(id)!;
    },

    accept(id: string, workspaceRoot: string): Patch {
      const patch = this.get(id);
      if (!patch) throw new Error("Patch not found");
      if (patch.status !== "pending") return patch;

      const abs = join(workspaceRoot, patch.path);
      try {
        if (patch.op === "create") {
          if (existsSync(abs)) {
            return this.fail(id, "exists");
          }
          mkdirSync(dirname(abs), { recursive: true });
          writeFileSync(abs, patch.newString, "utf8");
        } else if (patch.op === "delete") {
          if (!existsSync(abs)) return this.fail(id, "no_match");
          unlinkSync(abs);
        } else {
          if (!existsSync(abs)) return this.fail(id, "no_match");
          const cur = readFileSync(abs, "utf8");
          const result = applySearchReplaceToContent(cur, patch.oldString, patch.newString);
          if (!result.ok) return this.fail(id, result.reason);
          writeFileSync(abs, result.content, "utf8");
        }
      } catch (e) {
        return this.fail(id, e instanceof Error ? e.message : "apply_error");
      }

      const ts = new Date().toISOString();
      getDb()
        .query(
          `UPDATE patches SET status = 'accepted', resolved_at = ? WHERE id = ?`,
        )
        .run(ts, id);
      return this.get(id)!;
    },

    fail(id: string, reason: string): Patch {
      const ts = new Date().toISOString();
      getDb()
        .query(
          `UPDATE patches SET status = 'failed', reason = ?, resolved_at = ? WHERE id = ?`,
        )
        .run(reason, ts, id);
      return this.get(id)!;
    },

    acceptAll(opts: { sessionId?: string; runId?: string }, workspaceRoot: string): Patch[] {
      const pending = this.list({ ...opts, status: "pending" });
      return pending.map((p) => this.accept(p.id, workspaceRoot));
    },
  };
}

export const patchStore = createPatchStore();
```

**Important:** `propose` requires `agent_runs` row for FK — either:

1. Make FK optional in migration (drop FK on patches.run_id), **or**
2. Ensure tests insert a dummy `agent_runs` / `sessions` row first, **or**
3. Remove FK constraints from patches for simplicity in Phase A.

**Preferred for Phase A:** remove `FOREIGN KEY (run_id)` and `FOREIGN KEY (session_id)` from `patches` create SQL (and agent_runs session FK if it blocks tests) so unit tests and partial runs stay simple. Update Task 2 migration if not already applied; if DB already migrated, document `CREATE TABLE IF NOT EXISTS` won't alter — use a new migrate step:

```ts
// after initial exec, optional: no-op if tables exist
```

For greenfield installs without prior lail.sqlite, just omit FKs in Task 2 SQL. **If Task 2 already ran with FKs**, adjust Task 2 before continuing, or insert stub session/run in tests.

- [ ] **Step 4: Ensure schema has no blocking FKs (adjust if needed)**

In `schema.ts` patch table definition, use **no foreign keys** on `patches` and `agent_runs` for Phase A simplicity.

- [ ] **Step 5: Run tests — PASS**

```bash
cd <repo-root>/packages/backend
bun test src/agent/patch-store.test.ts
```

If `getDb()` points at real `data/lail.sqlite`, tests still OK if isolated IDs used. Prefer setting `LAIL_DATA_DIR` to a temp dir in tests:

```ts
// at top of test file before imports of getDb — hard with ESM.
// Alternative: export setDbForTests — skip unless needed.
```

If pollution is an issue, add `packages/backend/src/db/schema.ts` helper:

```ts
export function _resetDbForTests(path: string) {
  if (db) { db.close(); db = null; }
  // set config.dbPath via env LAIL_DB_PATH if config supports it
}
```

Check `config.ts` — if `dbPath` is fixed, extend config to honor `process.env.LAIL_DB_PATH` for tests.

- [ ] **Step 6: Commit**

```bash
git add packages/backend/src/agent/patch-store.ts packages/backend/src/agent/patch-store.test.ts packages/backend/src/db/schema.ts packages/backend/src/config.ts
git commit -m "feat(agent): PatchStore with exact search/replace apply"
```

---

### Task 5: Shell approval waiters

**Files:**
- Create: `packages/backend/src/agent/approvals.ts`
- Create: `packages/backend/src/agent/approvals.test.ts`

- [ ] **Step 1: Failing test**

```ts
import { describe, expect, test } from "bun:test";
import { approvalHub } from "./approvals";

describe("approvalHub", () => {
  test("resolves allow", async () => {
    const p = approvalHub.wait("a1", 2000);
    queuePromise.resolve().then(() => approvalHub.decide("a1", "allow"));
    await expect(p).resolves.toBe("allow");
  });

  test("timeout denies", async () => {
    const p = approvalHub.wait("a2", 50);
    await expect(p).resolves.toBe("deny");
  });
});
```

- [ ] **Step 2: Implement**

```ts
type Decision = "allow" | "deny";

const pending = new Map<
  string,
  { resolve: (d: Decision) => void; timer: ReturnType<typeof setTimeout> }
>();

export const approvalHub = {
  wait(approvalId: string, timeoutMs = 120_000): Promise<Decision> {
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        pending.delete(approvalId);
        resolve("deny");
      }, timeoutMs);
      pending.set(approvalId, {
        resolve: (d) => {
          clearTimeout(timer);
          pending.delete(approvalId);
          resolve(d);
        },
        timer,
      });
    });
  },
  decide(approvalId: string, decision: Decision): boolean {
    const p = pending.get(approvalId);
    if (!p) return false;
    p.resolve(decision);
    return true;
  },
};
```

- [ ] **Step 3: `bun test src/agent/approvals.test.ts` — PASS**
- [ ] **Step 4: Commit** `feat(agent): shell approval hub with timeout deny`

---

### Task 6: Tools — search_replace proposes, no silent write_file

**Files:**
- Modify: `packages/backend/src/tools/index.ts`
- Create: `packages/backend/src/tools/tools.test.ts` (optional light tests)

- [ ] **Step 1: Replace tool definitions**

Update `toolDefinitions` so edit tools are:

- `search_replace` — path, old_string, new_string  
- `create_file` — path, content  
- `delete_file` — path  
- Remove `write_file` from definitions (or keep alias that maps to create/replace only via PatchStore in runtime — **prefer remove**)

- [ ] **Step 2: Change `runTool` contract**

Extend `ToolResult`:

```ts
export type ToolResult = {
  ok: boolean;
  output: string;
  summary: string;
  fileWrite?: { path: string; bytes: number };
  /** When set, runtime should create a pending patch instead of treating as disk write */
  patchProposal?: {
    path: string;
    oldString: string;
    newString: string;
    op: "replace" | "create" | "delete";
  };
  needsShellApproval?: boolean;
  command?: string;
};
```

Implement:

- `search_replace` / `create_file` / `delete_file` → validate path via `assertWorkspaceRelativePath`, return `patchProposal` (do **not** write disk).  
- `run_shell` → call `classifyShell`; if `deny`, return fail; if `approve`, return `{ ok: false, needsShellApproval: true, command, output: "approval required", summary: "Awaiting approval" }` **or** let runtime handle classification before call — **prefer runtime handles classification** and only call `runTool` when allowed/approved.

Runtime-owned shell policy is cleaner:

- `run_shell` in tools only executes (soft deny hard-blocked patterns as last resort).  
- Runtime: `classifyShell` → approve wait → `runTool`.

- [ ] **Step 3: Manual sanity**

```bash
cd <repo-root>/packages/backend
bun -e "
import { toolDefinitions } from './src/tools/index.ts';
console.log(toolDefinitions.map(t => t.function.name).join(','))
"
```

Expected names include `search_replace,create_file,delete_file` and **not** `write_file`.

- [ ] **Step 4: Commit** `feat(tools): review-first patch tools replace write_file`

---

### Task 7: Prompts + thin ContextProvider

**Files:**
- Create: `packages/backend/src/agent/prompts.ts`
- Create: `packages/backend/src/agent/context.ts`

- [ ] **Step 1: Implement prompts**

```ts
import type { AgentMode } from "@lail/shared";

export function systemPrompt(mode: AgentMode, rootPath: string): string {
  const base = `You are Composer in L.A.I.L (Local AI Lab), a Cursor-style coding agent for local models.
Workspace root: ${rootPath}
Stay inside the workspace. Prefer tools over guessing file contents.
Never claim a file was written to disk until the user accepts a patch (the UI reviews patches).
`;

  if (mode === "plan") {
    return (
      base +
      `MODE=PLAN. Explore with read tools if needed. Produce a clear multi-step plan. Use the plan tool. Do not propose file edits or run shell.`
    );
  }
  if (mode === "ask") {
    return (
      base +
      `MODE=ASK. Answer questions using read tools. Cite paths. Do not propose edits or run shell. Do not invent a full implementation plan unless asked.`
    );
  }
  return (
    base +
    `MODE=AGENT. Implement changes via search_replace / create_file / delete_file tools (these become pending patches). Use run_shell when needed. Think briefly, act with tools, summarize what you proposed.`
  );
}
```

- [ ] **Step 2: Context stub**

```ts
import { listMessages } from "../controller/sessions";

export async function buildContext(sessionId: string): Promise<
  Array<{ role: string; content: string }>
> {
  return listMessages(sessionId)
    .filter((m) => m.role === "user" || m.role === "assistant")
    .filter((m) => {
      if (m.role === "assistant" && /^Error:\s*LLM error/i.test(m.content)) return false;
      if (m.role === "assistant" && /model `default` does not exist/i.test(m.content)) return false;
      return true;
    })
    .slice(-20)
    .map((m) => ({ role: m.role, content: m.content }));
}
```

- [ ] **Step 3: Commit** `feat(agent): mode system prompts and thin context provider`

---

### Task 8: AgentRuntime — mock LLM streaming + tools

**Files:**
- Create: `packages/backend/src/agent/runtime.ts`
- Create: `packages/backend/src/agent/runtime.test.ts`
- Modify: `packages/backend/src/controller/agent.ts` (thin wrapper)
- Modify: `packages/backend/package.json` (test script)

- [ ] **Step 1: Add test script**

In `packages/backend/package.json`:

```json
"test": "bun test"
```

- [ ] **Step 2: Write integration-style test with mock fetch**

```ts
import { describe, expect, test, mock, beforeAll, afterAll } from "bun:test";
import { mkdirSync, writeFileSync, readFileSync, rmSync } from "fs";
import { join } from "path";

// This test will call startAgentRun with dependency injection for fetch + workspace.
// Design runtime to accept optional deps for testing.
```

**Runtime public API:**

```ts
export type StartRunOpts = {
  sessionId: string;
  message: string;
  workspaceId: string;
  mode: AgentMode;
  /** test seam */
  fetchImpl?: typeof fetch;
  maxSteps?: number;
};

export function startAgentRun(opts: StartRunOpts): { runId: string };
export function cancelAgentRun(runId: string): boolean;
export function getAgentRun(runId: string): { id: string; status: string; mode: string } | null;
```

**Test scenario (non-stream first if stream mock is hard):**

1. Create temp workspace + real workspace row in DB **or** pass `rootPath` override.  
2. Mock `fetchImpl` to return one completion with `tool_calls: [search_replace]` then a final text message.  
3. Assert patch pending in DB, file unchanged.  
4. `patchStore.accept` → file changed.

For streaming: mock body as async iterator of SSE lines — if too heavy for first PR, implement stream parser but test non-stream path with `stream: true` fallback to JSON when mock returns JSON.

**Minimum viable runtime loop:**

1. Insert `agent_runs` row status=running.  
2. Publish status/thought.  
3. Loop up to maxSteps (default 32):  
   - Call LLM with tools filtered by mode (pass only allowed tool defs).  
   - Prefer `stream: true`; parse SSE `data: {choices...}`; accumulate content + tool_calls; publish `token` deltas.  
   - If tool_calls: for each, policy check → patch propose / shell approve / execute → publish events.  
   - If final text: break.  
4. Save assistant message, recordUsage, publish done.  
5. On cancel flag / AbortController: publish cancelled.

**Reuse** useful pieces from current `controller/agent.ts` (history filter, model resolve, offline fallback **removed** or lab-flagged off).

- [ ] **Step 3: Implement runtime until test passes**

Keep file focused; helper `parseChatCompletionStream(response): AsyncGenerator<...>` in same file or `agent/openai-stream.ts`.

- [ ] **Step 4: Point `controller/agent.ts` at runtime**

```ts
import { startAgentRun } from "../agent/runtime";
import type { AgentMode } from "@lail/shared";

export async function runAgent(opts: {
  sessionId: string;
  message: string;
  workspaceId?: string | null;
  mode?: AgentMode;
}): Promise<{ runId: string }> {
  // validate session + workspace (same as today)
  // ...
  return startAgentRun({
    sessionId: opts.sessionId,
    message: opts.message,
    workspaceId,
    mode: opts.mode ?? "agent",
  });
}

export { cancelAgentRun, getAgentRun } from "../agent/runtime";
```

Delete inlined old loop from `agent.ts` once runtime covers behavior (keep file thin).

- [ ] **Step 5: Run all backend agent tests**

```bash
cd <repo-root>/packages/backend
bun test src/agent
```

Expected: PASS.

- [ ] **Step 6: Commit** `feat(agent): streaming AgentRuntime with modes and patch proposals`

---

### Task 9: HTTP routes — cancel, approvals, patches

**Files:**
- Create: `packages/backend/src/controller/patches.ts`
- Modify: `packages/backend/src/app.ts`
- Modify: `packages/backend/src/controller/agent.ts` (export cancel)

- [ ] **Step 1: Implement patch controller helpers**

```ts
// packages/backend/src/controller/patches.ts
import { patchStore } from "../agent/patch-store";
import { getWorkspace } from "./workspaces";
import { getSession } from "./sessions";
import { wsHub } from "../ws/hub";

export function listPatches(q: { sessionId?: string; runId?: string; status?: string }) {
  return patchStore.list(q as any);
}

export function acceptPatch(id: string) {
  const patch = patchStore.get(id);
  if (!patch) throw Object.assign(new Error("not found"), { code: "NOT_FOUND" });
  const session = getSession(patch.sessionId);
  if (!session?.workspaceId) throw new Error("No workspace");
  const ws = getWorkspace(session.workspaceId);
  if (!ws) throw new Error("Workspace missing");
  const updated = patchStore.accept(id, ws.rootPath);
  wsHub.publish(`agent:${patch.runId}`, { runId: patch.runId, type: "patch_updated", patch: updated });
  if (updated.status === "accepted") {
    wsHub.publish(`agent:${patch.runId}`, {
      runId: patch.runId,
      type: "file_write",
      path: updated.path,
      bytes: Buffer.byteLength(updated.newString),
    });
  }
  return updated;
}

// rejectPatch, acceptAll similarly
```

- [ ] **Step 2: Wire routes in `app.ts`**

Replace/extend agent routes:

```ts
app.post("/api/agent/run", async (c) => {
  const body = await c.req.json();
  const result = await runAgent({
    sessionId: body.sessionId,
    message: body.message,
    workspaceId: body.workspaceId,
    mode: body.mode, // "plan" | "ask" | "agent"
  });
  return c.json(result);
});

app.post("/api/agent/runs/:runId/cancel", (c) => {
  const ok = cancelAgentRun(c.req.param("runId"));
  return c.json({ ok });
});

app.post("/api/agent/runs/:runId/shell-approvals/:approvalId", async (c) => {
  const body = await c.req.json();
  const decision = body.decision === "allow" ? "allow" : "deny";
  const ok = approvalHub.decide(c.req.param("approvalId"), decision);
  return c.json({ ok });
});

app.get("/api/patches", (c) => {
  return c.json(
    listPatches({
      sessionId: c.req.query("sessionId") || undefined,
      runId: c.req.query("runId") || undefined,
      status: c.req.query("status") || undefined,
    }),
  );
});

app.post("/api/patches/:id/accept", (c) => {
  try {
    return c.json(acceptPatch(c.req.param("id")));
  } catch (e) {
    return c.json({ error: (e as Error).message }, 400);
  }
});

app.post("/api/patches/:id/reject", (c) => c.json(rejectPatch(c.req.param("id"))));

app.post("/api/patches/accept-all", async (c) => {
  const body = await c.req.json();
  return c.json(acceptAllPatches(body));
});
```

Keep legacy path `POST /api/agent/run` (already exists) — do not break old clients; add `mode` optional default `"agent"`.

- [ ] **Step 3: Manual route smoke with bun**

```bash
# with API running OR app.request:
cd <repo-root>/packages/backend
bun -e "
import { createApp } from './src/app.ts';
const app = createApp();
const r = await app.request('/api/health');
console.log(await r.json());
"
```

Expected: `{ status: 'ok', ... }`.

- [ ] **Step 4: Commit** `feat(api): agent cancel, shell approvals, patch accept/reject routes`

---

### Task 10: Web API client + store

**Files:**
- Modify: `apps/web/lib/api.ts`
- Modify: `apps/web/lib/store.ts`

- [ ] **Step 1: Extend api**

```ts
export type AgentMode = "plan" | "ask" | "agent";
export type Patch = {
  id: string;
  runId: string;
  sessionId: string;
  path: string;
  oldString: string;
  newString: string;
  op: "replace" | "create" | "delete";
  status: "pending" | "accepted" | "rejected" | "failed";
  reason?: string;
  createdAt: string;
  resolvedAt?: string;
};

// in api object:
agentRun: (sessionId: string, message: string, workspaceId?: string, mode: AgentMode = "agent") =>
  req<{ runId: string }>("/api/agent/run", {
    method: "POST",
    body: JSON.stringify({ sessionId, message, workspaceId, mode }),
  }),
cancelAgentRun: (runId: string) =>
  req<{ ok: boolean }>(`/api/agent/runs/${runId}/cancel`, { method: "POST" }),
shellApproval: (runId: string, approvalId: string, decision: "allow" | "deny") =>
  req<{ ok: boolean }>(`/api/agent/runs/${runId}/shell-approvals/${approvalId}`, {
    method: "POST",
    body: JSON.stringify({ decision }),
  }),
patches: {
  list: (q: { sessionId?: string; runId?: string; status?: string }) => {
    const sp = new URLSearchParams();
    if (q.sessionId) sp.set("sessionId", q.sessionId);
    if (q.runId) sp.set("runId", q.runId);
    if (q.status) sp.set("status", q.status);
    return req<Patch[]>(`/api/patches?${sp}`);
  },
  accept: (id: string) => req<Patch>(`/api/patches/${id}/accept`, { method: "POST" }),
  reject: (id: string) => req<Patch>(`/api/patches/${id}/reject`, { method: "POST" }),
  acceptAll: (body: { sessionId?: string; runId?: string }) =>
    req<Patch[]>("/api/patches/accept-all", { method: "POST", body: JSON.stringify(body) }),
},
```

- [ ] **Step 2: Extend zustand store**

Add:

```ts
agentMode: AgentMode; // default "agent"
setAgentMode: (m: AgentMode) => void;
pendingPatches: Patch[];
setPendingPatches: (p: Patch[]) => void;
upsertPatch: (p: Patch) => void;
shellApproval: null | { runId: string; approvalId: string; command: string };
setShellApproval: (v: LabStore["shellApproval"]) => void;
streamingText: string;
appendStreamingText: (t: string) => void;
clearStreamingText: () => void;
```

- [ ] **Step 3: Commit** `feat(web): API client and store for modes, patches, approvals`

---

### Task 11: ide-chrome contracts

**Files:**
- Modify: `apps/web/lib/ide-chrome.ts`
- Modify: `apps/web/lib/ide-chrome.test.ts`

- [ ] **Step 1: Add exports**

```ts
export const AGENT_MODES = ["plan", "ask", "agent"] as const;
export const AGENT_MODE_LABELS = { plan: "Plan", ask: "Ask", agent: "Agent" } as const;

export const STREAM_MARKERS = {
  thought: "Thought",
  working: "Working",
  ran: "Ran",
  creating: "Creating",
  proposed: "Proposed",
  status: "Status",
} as const;
```

Extend `groupTimeline` / `TimelineKind` to support `patch` kind → stream block `{ type: "patch"; path: string }`.

- [ ] **Step 2: Tests**

```ts
test("mode labels Plan Ask Agent", () => {
  expect(AGENT_MODE_LABELS.plan).toBe("Plan");
  expect(AGENT_MODE_LABELS.ask).toBe("Ask");
  expect(AGENT_MODE_LABELS.agent).toBe("Agent");
});

test("proposed patch marker", () => {
  expect(STREAM_MARKERS.proposed).toBe("Proposed");
});
```

- [ ] **Step 3: `cd apps/web && bun test lib/ide-chrome.test.ts` — PASS**
- [ ] **Step 4: Commit** `test(web): Cursor-style mode and patch chrome contracts`

---

### Task 12: Workbench UI components

**Files:**
- Create: `apps/web/components/workbench/ModeToggle.tsx`
- Create: `apps/web/components/workbench/PatchReviewPanel.tsx`
- Create: `apps/web/components/workbench/ShellApprovalBanner.tsx`
- Modify: `apps/web/app/workbench/page.tsx`

- [ ] **Step 1: ModeToggle**

Segmented control Plan | Ask | Agent using `AGENT_MODE_LABELS`; calls `setAgentMode`. Disabled while `busy` optional (spec: mode applies to next run — can allow change while busy for next message only; simplest: allow always, send uses current mode).

- [ ] **Step 2: PatchReviewPanel**

List `pendingPatches` (and failed); show path, op, truncated old/new; buttons Accept / Reject; footer Accept all. On Accept: `api.patches.accept` → `upsertPatch` → reload file tab via `openFile`.

- [ ] **Step 3: ShellApprovalBanner**

When `shellApproval` non-null, show command + Allow / Deny → `api.shellApproval` → clear state.

- [ ] **Step 4: Wire `page.tsx`**

- `send()` passes `agentMode`.  
- WS handler:  
  - `token` → `appendStreamingText`  
  - `patch_proposed` → `upsertPatch`  
  - `patch_updated` → `upsertPatch`  
  - `shell_approval_required` → `setShellApproval`  
  - `cancelled` → clear busy  
  - keep existing tool_end / thought / error / done  
- **Do not** open editor on patch_proposed; open/refresh on `file_write` (post-accept) only.  
- Add Cancel button when busy → `api.cancelAgentRun(activeRunId)`.  
- Layout: Composer column + optional PatchReviewPanel (right of composer or replace Status when patches pending). Keep Status rail; patch panel can sit above Status or as a third column ~280px.

- [ ] **Step 5: Visual smoke**

```bash
cd <repo-root>
bun run dev
# open /workbench — mode toggle visible, placeholder still "Ask for follow-up changes"
```

- [ ] **Step 6: Commit** `feat(web): Workbench modes, patch review, shell approval, cancel`

---

### Task 13: shell-source / regression tests

**Files:**
- Modify: `apps/web/lib/shell-source.test.ts`

- [ ] **Step 1: Assert Workbench source contains ModeToggle usage or AGENT_MODE labels and patch panel import**

```ts
test("workbench has mode toggle and patch review", () => {
  expect(wb).toMatch(/ModeToggle|agentMode|Plan/);
  expect(wb).toMatch(/PatchReview|pendingPatches|patches\.accept/);
});
```

- [ ] **Step 2: `bun test` in apps/web — PASS**
- [ ] **Step 3: Commit** `test(web): workbench Phase A surface contracts`

---

### Task 14: Docs + CHECKPOINT

**Files:**
- Modify: `README.md`
- Modify: `CHECKPOINT.md`

- [ ] **Step 1: Document Phase A behavior**

Update Workbench section:

- Modes Plan / Ask / Agent  
- Review-first patches (Accept/Reject)  
- Streaming + cancel  
- Risky shell approval  
- Note Phases B–E still planned (link spec)

- [ ] **Step 2: Commit** `docs: Phase A Cursor-style agent platform`

---

### Task 15: End-to-end verification checklist

**Files:** none (manual + automated)

- [ ] **Step 1: Automated**

```bash
cd <repo-root>/packages/backend && bun test
cd <repo-root>/apps/web && bun test
cd <repo-root> && bun run typecheck
```

Expected: all PASS.

- [ ] **Step 2: Manual with local model (when available)**

1. Serve model (Server tab) / Configure real model id.  
2. Workbench → **Plan**: ask for a plan to add a README section — no patches.  
3. **Ask**: “what files are in this workspace?” — reads only.  
4. **Agent**: “Add a line to README via search_replace” — pending patch appears; file unchanged until Accept.  
5. Reject → still unchanged; re-run or new patch; Accept → disk updates; tab refreshes.  
6. Agent: `run_shell` with `rm -rf something` → approval UI; Deny → agent continues.  
7. Cancel mid-run → stream stops.  
8. Server / Models / Usage still load.

- [ ] **Step 3: Final commit if any fixes** `fix: Phase A verification follow-ups`

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|------------------|------|
| Modular Runtime / PatchStore / ToolPolicy / EventBus | 3, 4, 5, 8 (EventBus = existing wsHub) |
| Plan / Ask / Agent modes | 3, 7, 8, 12 |
| Review-first search/replace | 4, 6, 9, 12 |
| Risky shell approval | 3, 5, 8, 9, 12 |
| Full streaming tokens + tools | 8, 12 |
| Cancel | 8, 9, 12 |
| Persist patches / runs | 2, 4 |
| Thin ContextProvider | 7 |
| Lab surfaces retained | File map non-goals; Task 15 checks |
| Tests unit + integration + UI contract | 3, 4, 5, 8, 11, 13, 15 |
| Browser IDE only | No desktop tasks |
| No Monaco / context index / terminal | Not in tasks |

**Gaps addressed in plan:** FK flexibility for patches; test DB isolation note; legacy `/api/agent/run` kept.

**Type names:** `oldString`/`newString` in TS Patch; SQL `old_string`/`new_string` — consistent mapping in `rowToPatch`.

---

## Out of scope (do not implement in this plan)

- Phase B `@` context / embeddings  
- Phase C Monaco  
- Phase D terminal/git UI  
- Phase E command palette  
- Desktop shell  
- serve-engine rewrites  

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-16-lail-phase-a-agent-platform.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration  
2. **Inline Execution** — execute tasks in this session using executing-plans, batch with checkpoints  

**Which approach?**
