#!/usr/bin/env python3
"""Deny git commit/push/merge on main, deny gh pr merge, and deny staging secrets.

Reads a Grok PreToolUse JSON envelope from stdin. Fail-open on parse errors.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys


def allow() -> None:
    print('{"decision":"allow"}')
    sys.exit(0)


def deny(reason: str) -> None:
    print(json.dumps({"decision": "deny", "reason": reason}))
    sys.exit(0)


def head_branch(cwd: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def looks_like_git(cmd: str) -> bool:
    return bool(re.search(r"(?:^|[;&|(\n])\s*(?:sudo\s+)?git\b", cmd))


COMMIT_RE = re.compile(r"\bgit\b(?:\s+-[^\s]+)*\s+commit\b")
PUSH_RE = re.compile(r"\bgit\b(?:\s+-[^\s]+)*\s+push\b")
MERGE_RE = re.compile(r"\bgit\b(?:\s+-[^\s]+)*\s+merge\b")
GH_PR_MERGE_RE = re.compile(r"\bgh\b(?:\s+\S+)*?\s+pr\s+merge\b")
PROTECTED = {"main", "master"}
SECRET_RE = re.compile(
    r"(?:^|[\s'\"])((?:\.env(?:\.local)?)|(?:data/\S+\.sqlite(?:-shm|-wal)?)|(?:data/(?:multinode_serve|cluster)\.json)|(?:apps/web/next-env\.d\.ts))"
)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        allow()

    inp = data.get("toolInput") or {}
    cmd = inp.get("command") or inp.get("cmd") or ""
    if not isinstance(cmd, str) or not cmd.strip():
        allow()

    cwd = data.get("cwd") or data.get("workspaceRoot") or os.getcwd()
    branch = head_branch(cwd)

    if GH_PR_MERGE_RE.search(cmd):
        deny(
            "Refusing gh pr merge. Open the PR and stop — only the human merges."
        )

    if not looks_like_git(cmd):
        allow()

    if COMMIT_RE.search(cmd) and branch in PROTECTED:
        deny(
            f"Refusing git commit on '{branch}'. "
            "Create a feat/fix/chore/docs branch and open a PR. "
            "main is protected; committing here is not allowed."
        )

    if PUSH_RE.search(cmd):
        pushes_main = bool(re.search(r"\b(?:main|master)\b", cmd))
        if branch in PROTECTED or pushes_main:
            deny(
                "Refusing git push to main/master. Push a topic branch and open a PR."
            )

    if MERGE_RE.search(cmd) and branch in PROTECTED:
        deny(
            f"Refusing git merge while on '{branch}'. "
            "Open a PR; only the human merges."
        )

    if re.search(r"\bgit\b(?:\s+-[^\s]+)*\s+add\b", cmd):
        hits = SECRET_RE.findall(cmd)
        if hits:
            deny(
                "Refusing to stage secret/runtime paths: "
                + ", ".join(hits)
                + ". See AGENTS.md / SECURITY.md."
            )

    allow()


if __name__ == "__main__":
    main()
