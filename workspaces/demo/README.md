# Demo artifacts

Sample files for the lab gallery. Import one with:

```bash
curl -sS -X POST http://127.0.0.1:8787/api/lab/runs/import \
  -H 'Content-Type: application/json' \
  -d '{"title":"Geometry Dash–like runner","from":"workspaces/demo/geometry-dash-like.html"}'
```

`/workbench` and its workspace tools are retired — Hermes is the agent, wired to the served `:8000` endpoint.
