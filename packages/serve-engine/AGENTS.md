# AGENTS.md — `packages/serve-engine`

Python FastAPI: vLLM Lab Safe / Workflow Max, smoke, benches, run history.

## Placement

- Flags are generated in `app/services/autoconfig.py`. Do not hardcode model recipes in Python.
- Add future model recipes to repo-root `data/serve_overlays.json`.
- Corpus tests live in `tests/corpus/` and `tests/fixtures/`. A placement change without a corpus or unit test is incomplete.

## Verify

```bash
.venv/bin/python -m pytest packages/serve-engine/tests/ -q
```

A headless multi-node TP worker reports `serving_worker` and has no `/v1/models` endpoint by design — treat it as serving.
