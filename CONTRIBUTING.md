# Contributing

Thanks for helping with L.A.I.L (Local AI Lab). Keep the product name; do not commit `.env`, `data/*.sqlite`, or run logs.

## Setup

```bash
git clone <this-repo>
cd <repo>
python3 -m venv .venv
source .venv/bin/activate
pip install -e "packages/serve-engine[dev]"
bun install
cp .env.example .env
```

## Checks

```bash
bun run typecheck
bun test apps/web packages/backend
python3 -m pytest packages/serve-engine/tests -q
```

## Coding

See [AGENTS.md](./AGENTS.md). Surgical diffs; no drive-by refactors.

## Pull requests

Open a PR against the default branch. CI (`.github/workflows/ci.yml`) must pass.
