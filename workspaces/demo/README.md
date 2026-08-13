# Demo workspace

Default project root under `LAIL_WORKSPACES_DIR`. First-run path is **Status → Serve → Auto-configure → Start**, then **`/connect`** for Hermes.

`/workbench` is retired (Hermes is the agent). If you still hit the old Composer tools, they sandbox here.

## Notes

- Shell tools run with **cwd = this workspace**.
- Paths cannot escape the workspace root.
- Point **Configure → Default model** at your live vLLM (or llama.cpp) model id.

## Quick Start

Run `ls` to list files or `cat <file>` to read them.
