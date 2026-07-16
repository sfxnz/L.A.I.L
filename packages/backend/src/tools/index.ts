import { existsSync, readFileSync, readdirSync, statSync } from "fs";
import { join, relative } from "path";
import { resolveInWorkspace, getWorkspace } from "../controller/workspaces";
import { assertWorkspaceRelativePath } from "../agent/tool-policy";

export const toolDefinitions = [
  {
    type: "function" as const,
    function: {
      name: "list_dir",
      description: "List files and directories under a path in the workspace",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Relative path (default .)" },
        },
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "read_file",
      description: "Read a text file from the workspace",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string" },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "search_replace",
      description:
        "Propose a search-and-replace edit in a workspace file (pending review; does not write disk)",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string" },
          old_string: { type: "string" },
          new_string: { type: "string" },
        },
        required: ["path", "old_string", "new_string"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "create_file",
      description:
        "Propose creating a new text file in the workspace (pending review; does not write disk)",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string" },
          content: { type: "string" },
        },
        required: ["path", "content"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "delete_file",
      description:
        "Propose deleting a file in the workspace (pending review; does not write disk)",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string" },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "grep",
      description: "Search for a pattern in workspace files",
      parameters: {
        type: "object",
        properties: {
          pattern: { type: "string" },
          path: { type: "string", description: "Subpath to search" },
        },
        required: ["pattern"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "run_shell",
      description: "Run a shell command with cwd set to the workspace root (ls, cat, grep, find, etc.)",
      parameters: {
        type: "object",
        properties: {
          command: { type: "string" },
        },
        required: ["command"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "plan",
      description: "Record a short multi-step plan for the user timeline",
      parameters: {
        type: "object",
        properties: {
          steps: { type: "array", items: { type: "string" } },
        },
        required: ["steps"],
      },
    },
  },
];

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

export async function runTool(
  workspaceId: string,
  name: string,
  args: Record<string, unknown>,
): Promise<ToolResult> {
  const ws = getWorkspace(workspaceId);
  if (!ws) return { ok: false, output: "Workspace not found", summary: "Failed" };

  try {
    switch (name) {
      case "list_dir": {
        const rel = String(args.path || ".");
        const abs = resolveInWorkspace(workspaceId, rel);
        if (!existsSync(abs)) return { ok: false, output: "Not found", summary: "list_dir failed" };
        const entries = readdirSync(abs).map((n) => {
          const st = statSync(join(abs, n));
          return `${st.isDirectory() ? "d" : "f"} ${n}`;
        });
        return {
          ok: true,
          output: entries.join("\n") || "(empty)",
          summary: `Listed ${entries.length} entries`,
        };
      }
      case "read_file": {
        const abs = resolveInWorkspace(workspaceId, String(args.path));
        const buf = readFileSync(abs);
        if (buf.length > 512_000) {
          return {
            ok: true,
            output: buf.subarray(0, 512_000).toString("utf8") + "\n…(truncated)",
            summary: `Read ${args.path} (truncated)`,
          };
        }
        return { ok: true, output: buf.toString("utf8"), summary: `Read ${args.path}` };
      }
      case "search_replace": {
        const path = assertWorkspaceRelativePath(String(args.path));
        const oldString = String(args.old_string ?? "");
        const newString = String(args.new_string ?? "");
        return {
          ok: true,
          output: `Proposed replace in ${path} (${oldString.length} → ${newString.length} chars)`,
          summary: `Proposed replace in ${path}`,
          patchProposal: { path, oldString, newString, op: "replace" },
        };
      }
      case "create_file": {
        const path = assertWorkspaceRelativePath(String(args.path));
        const content = String(args.content ?? "");
        return {
          ok: true,
          output: `Proposed create ${path} (${content.length} bytes)`,
          summary: `Proposed create ${path}`,
          patchProposal: {
            path,
            oldString: "",
            newString: content,
            op: "create",
          },
        };
      }
      case "delete_file": {
        const path = assertWorkspaceRelativePath(String(args.path));
        return {
          ok: true,
          output: `Proposed delete ${path}`,
          summary: `Proposed delete ${path}`,
          patchProposal: {
            path,
            oldString: "",
            newString: "",
            op: "delete",
          },
        };
      }
      case "grep": {
        const pattern = String(args.pattern);
        const rel = String(args.path || ".");
        const abs = resolveInWorkspace(workspaceId, rel);
        const proc = Bun.spawn(["rg", "-n", "--max-count", "50", pattern, abs], {
          cwd: ws.rootPath,
          stdout: "pipe",
          stderr: "pipe",
        });
        const out = await new Response(proc.stdout).text();
        const err = await new Response(proc.stderr).text();
        await proc.exited;
        if (proc.exitCode !== 0 && !out) {
          // fallback simple search
          return simpleGrep(ws.rootPath, abs, pattern);
        }
        return {
          ok: true,
          output: out || err || "(no matches)",
          summary: `grep ${pattern}`,
        };
      }
      case "run_shell": {
        const command = String(args.command || "");
        if (!command.trim()) return { ok: false, output: "Empty command", summary: "Failed" };
        // Soft guardrails (hard-blocked patterns); runtime owns classifyShell/approval
        if (/\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?\/\b/.test(command) || command.includes("mkfs")) {
          return { ok: false, output: "Command blocked by safety policy", summary: "Blocked" };
        }
        const proc = Bun.spawn(["bash", "-lc", command], {
          cwd: ws.rootPath,
          stdout: "pipe",
          stderr: "pipe",
          env: { ...process.env, PWD: ws.rootPath },
        });
        const timer = setTimeout(() => proc.kill(), 30_000);
        const stdout = await new Response(proc.stdout).text();
        const stderr = await new Response(proc.stderr).text();
        await proc.exited;
        clearTimeout(timer);
        const combined = [stdout, stderr].filter(Boolean).join("\n").slice(0, 40_000);
        return {
          ok: proc.exitCode === 0,
          output: combined || `(exit ${proc.exitCode})`,
          summary: `Ran shell: ${command.slice(0, 60)}`,
        };
      }
      case "plan": {
        const steps = (args.steps as string[]) || [];
        return {
          ok: true,
          output: steps.map((s, i) => `${i + 1}. ${s}`).join("\n"),
          summary: `Planned ${steps.length} steps`,
        };
      }
      default:
        return { ok: false, output: `Unknown tool ${name}`, summary: "Unknown tool" };
    }
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { ok: false, output: msg, summary: `${name} error` };
  }
}

function simpleGrep(root: string, start: string, pattern: string): ToolResult {
  const re = new RegExp(pattern, "i");
  const hits: string[] = [];
  const walk = (dir: string) => {
    if (hits.length >= 50) return;
    let names: string[];
    try {
      names = readdirSync(dir);
    } catch {
      return;
    }
    for (const n of names) {
      if (n === "node_modules" || n === ".git") continue;
      const full = join(dir, n);
      let st;
      try {
        st = statSync(full);
      } catch {
        continue;
      }
      if (st.isDirectory()) walk(full);
      else if (st.isFile() && st.size < 200_000) {
        try {
          const text = readFileSync(full, "utf8");
          const lines = text.split("\n");
          lines.forEach((line, i) => {
            if (re.test(line) && hits.length < 50) {
              hits.push(`${relative(root, full)}:${i + 1}:${line.slice(0, 200)}`);
            }
          });
        } catch {
          /* binary */
        }
      }
    }
  };
  walk(start);
  return {
    ok: true,
    output: hits.join("\n") || "(no matches)",
    summary: `grep ${pattern}`,
  };
}
