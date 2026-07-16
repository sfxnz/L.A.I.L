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
