#!/usr/bin/env bash
# Point this clone at the versioned hooks. Safe to re-run.
set -euo pipefail
root=$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)
git -C "$root" config core.hooksPath scripts/git-hooks
chmod +x "$root"/scripts/git-hooks/pre-commit "$root"/scripts/git-hooks/pre-push "$root"/scripts/git-hooks/commit-msg
echo "core.hooksPath=scripts/git-hooks"
