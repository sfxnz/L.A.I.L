# Security

## Reporting

Report vulnerabilities privately via GitHub Security Advisories on this repository. Do not file a public issue for an unpatched security problem.

## Secrets and lab access

Never commit:

- `.env` or live tokens (`HF_TOKEN`, API keys)
- Cluster inventory (`LAIL_CLUSTER_JSON` values, `data/cluster.json`, `data/multinode_serve.json`)
- Anything that exposes Docker on the host (`docker.sock`, remote Docker URLs, or credentials that can reach them)
- SQLite databases and run logs under `data/`

Use [`.env.example`](./.env.example) as the template. Treat `HF_TOKEN`, `LAIL_TOKEN`, cluster JSON, and Docker access as operator secrets.

## Bind policy

`bun run dev` and the controller default to `LAIL_HOST=127.0.0.1`. Anyone on the LAN must not be able to start/stop Docker or hit shell tools.

- Loopback bind: no token required.
- Off-loopback bind (`0.0.0.0`, `::`, a LAN IP): set `LAIL_TOKEN` or the process refuses to start.
- Send the token as `Authorization: Bearer <token>` or `X-Lail-Token`. WebSocket and EventSource: `?token=`.
- The web UI does **not** inject the operator secret. If a request returns 401, paste `LAIL_TOKEN` into the banner; it is kept in `sessionStorage` and sent as `X-Lail-Token` / `?token=`.
- Docker Compose publishes `127.0.0.1:PORT:PORT` and sets `LAIL_INSECURE_BIND=1` for the container-internal `0.0.0.0` listen. If you publish those ports on `0.0.0.0`, set `LAIL_TOKEN` and enter it in the UI. Do not rely on the escape hatch.
