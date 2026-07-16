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
