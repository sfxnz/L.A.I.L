---
name: verify-lail
description: Use when about to claim L.A.I.L work is done, before opening a PR, or after changing UI, controller, serve-engine, or auth/bind behavior.
---

# Verify L.A.I.L

Do not claim done until the matching checks have been run and their output inspected.

## Always

```bash
export PATH="$HOME/.bun/bin:$PATH"
bun run typecheck
bun test apps/web packages/backend
.venv/bin/python -m pytest packages/serve-engine/tests/ -q
```

Add `bun run build` when the change can break the production build.

## UI / CSS / client state

1. Open the changed route in Chrome DevTools MCP (`http://127.0.0.1:3000/...`).
2. Exercise it as a user. Check other pages that share the state.
3. A fresh profile getting `401` / `LAIL_TOKEN required` is expected, not "controller down".
4. Layout/CSS: desktop and a narrow viewport. Do not hand-write `-webkit-` prefixes (see `apps/web/AGENTS.md`).

## Serve-engine / placement

Run the pytest line above. Add or update a corpus/fixture if a recipe or placement rule changed.

## Git

Never commit or push on `main`. Branch, PR, CI green. Do not merge the PR — only the human merges.
