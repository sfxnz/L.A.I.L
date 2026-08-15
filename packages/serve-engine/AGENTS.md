# AGENTS.md — `packages/serve-engine`

Python FastAPI: vLLM auto-configure, smoke, benches, run history.

## Placement

- Flags are generated in `app/services/autoconfig.py`. Do not hardcode model recipes in Python.
- Add future model recipes to repo-root `data/serve_overlays.json`.
- Corpus tests live in `tests/corpus/` and `tests/fixtures/`. A placement change without a corpus or unit test is incomplete.

## Verify

```bash
.venv/bin/python -m pytest packages/serve-engine/tests/ -q
```

A headless multi-node TP worker reports `serving_worker` and has no `/v1/models` endpoint by design — treat it as serving.

## Cluster

Default topology is this host plus QSFP/RoCE peers that answer ping (`app/services/cluster.py`). `LAIL_CLUSTER_JSON` or gitignored `data/cluster.json` still override. Do not bake lab hostnames or interconnect IPs into the default.

An empty `ip neigh` after reboot is not “single node.” The ARP table is cold until something talks on the link — scan the QSFP prefix (/24 or tighter) when neigh is empty. Do not “fix” a missing peer by writing inventory into the repo.
