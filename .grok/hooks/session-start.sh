#!/usr/bin/env bash
# Install versioned git hooks for this clone. Non-fatal.
set -euo pipefail
root=${GROK_WORKSPACE_ROOT:-${PWD}}
if [ -x "$root/scripts/install-git-hooks.sh" ]; then
  "$root/scripts/install-git-hooks.sh" >/dev/null 2>&1 || true
fi
exit 0
