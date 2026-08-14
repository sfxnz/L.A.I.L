# AGENTS.md — L.A.I.L (Local AI Lab)

Serve & eval console for local vLLM / llama.cpp. Hermes (or any OpenAI client) is the coding agent — wire it to the live `:8000` endpoint after Serve. `/` redirects to Status. `/workbench` is retired.

## Working rules

- **Think first.** State assumptions. Ask on real ambiguity. Do not pick an interpretation silently.
- **Simplicity.** Minimum code that solves the request. No speculative features or abstractions.
- **Surgical.** Every changed line traces to the request. No drive-by refactors or reformatting. Clean up only your own orphans.
- **Plan Mode first** for anything non-trivial: multi-file work, auth/bind, serve-engine placement, CSS architecture, infra, or unclear requirements. Propose a plan, wait for approval, then implement.
- **Verify before done.** Run the checks below and show real output. UI changes also need a browser pass (load `verify-lail` when claiming UI/serve/eval work is done).

## Build, test, run

`bun` is often missing from tmux PATH:

```bash
export PATH="$HOME/.bun/bin:$PATH"
```

Dev stack: tmux session `lail` — web `:3000`, controller `:8787`, serve-engine `:8765`.

```bash
bun run typecheck                              # web + backend — required
bun test apps/web packages/backend             # required (this is what CI runs)
.venv/bin/python -m pytest packages/serve-engine/tests/ -q   # required (not system python3)
bun run build                                  # when the change can break production build
```

Do not claim done if any of the required commands were skipped or failed.

## Never touch / never commit

- `.env`, `.env.local`, live tokens (`HF_TOKEN`, `LAIL_TOKEN`, API keys)
- `data/*.sqlite*`, `data/multinode_serve.json`, `data/cluster.json`, run logs
- `apps/web/next-env.d.ts` (generated, gitignored on purpose)
- Cluster inventory and anything that exposes Docker on the host

These are gitignored for a reason. Re-tracking them dirties every serve/eval run. See [SECURITY.md](./SECURITY.md).

## Invariants

- Serve flags come from the placement engine (`packages/serve-engine/app/services/autoconfig.py`). Add model recipes to `data/serve_overlays.json`, do not hardcode.
- A headless multi-node TP worker reports `serving_worker` (no `/v1/models` by design). Treat that as serving in any UI check.
- **HTTP 401 with `LAIL_TOKEN required` is not "controller down."** Fresh browsers have no token. Loopback (`127.0.0.1`) needs no token; off-loopback bind requires `LAIL_TOKEN`. Paste it in the UI banner (`sessionStorage`). The web app does not inject the operator secret.
- `bun run dev` defaults to `LAIL_HOST=127.0.0.1`. Do not bind `0.0.0.0` without a token.

## Git (non-negotiable)

`main` is protected. **Never commit, merge, or push to `main`.**

1. Branch from latest `main`: `feat/`, `fix/`, `chore/`, `docs/`, `test/`, `ci/`.
2. Conventional Commits: `type(scope): summary` — types `feat` `fix` `test` `chore` `docs` `refactor` `ci` `perf`. Scopes: `web` `backend` `serve-engine` `autoconfig` `security` `workflows`.
3. Open a PR against `main`. CI (`.github/workflows/ci.yml`) must pass.
4. **Do not merge the PR** — not `gh pr merge`, not the GitHub UI, not squash/rebase/merge. Only the human merges.
5. Do not `--no-verify`, force-push to `main`, or amend other people's commits.
6. Commit only after the required checks above have been run.

Local hooks live in `scripts/git-hooks/` (`./scripts/install-git-hooks.sh` once per clone). Grok hooks in `.grok/hooks/` deny `git commit` / `git push` on `main` even if you forget.

## UI changes

Exercise the changed route the way a user would (Chrome DevTools MCP is configured on this lab). Confirm 401-without-token vs a real outage. Check other pages that share the state you touched. Layout/CSS: desktop and a narrow viewport.

## Pointers

- Humans: [CONTRIBUTING.md](./CONTRIBUTING.md), [README.md](./README.md), [SECURITY.md](./SECURITY.md)
- Package rules: `apps/web/AGENTS.md`, `packages/serve-engine/AGENTS.md`
- Large audits: `.grok/workflows/` (do not start the `lail` tmux server from an agent sandbox)
