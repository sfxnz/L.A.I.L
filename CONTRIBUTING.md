# Contributing

Thanks for helping with L.A.I.L (Local AI Lab). Keep the product name; do not commit `.env`, `data/*.sqlite`, or run logs.

## Setup

```bash
git clone https://github.com/sfxnz/L.A.I.L.git
cd <repo>
python3 -m venv .venv
source .venv/bin/activate
pip install -e "packages/serve-engine[dev]"
bun install
cp .env.example .env
./scripts/install-git-hooks.sh
```

The last line points this clone at `scripts/git-hooks/` so commits on `main` and secret files are rejected locally.

## Checks

```bash
export PATH="$HOME/.bun/bin:$PATH"
bun run typecheck
bun test apps/web packages/backend
.venv/bin/python -m pytest packages/serve-engine/tests -q
```

CI (`.github/workflows/ci.yml`) runs the same three.

## Git

`main` is protected: pull request required, linear history, no force-push, CI `test` must pass.

1. Branch from latest `main`. Names: `feat/…`, `fix/…`, `chore/…`, `docs/…`, `test/…`, `ci/…`.
2. Conventional Commits on the first line: `type(scope): summary`
   - types: `feat` `fix` `test` `chore` `docs` `refactor` `ci` `perf` `revert`
   - scopes (optional): `web` `backend` `serve-engine` `autoconfig` `security` `workflows`
3. Open a PR against `main`. Do not commit or push to `main`.
4. Do not merge the PR. Only the maintainer merges (`gh pr merge` and the GitHub merge button are both off-limits to agents).
5. Do not use `--no-verify` to skip hooks.

Agents follow the same rules — see [AGENTS.md](./AGENTS.md). Surgical diffs; no drive-by refactors.
