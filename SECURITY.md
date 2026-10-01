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
- Send the token as `Authorization: Bearer <token>` or `X-Lail-Token`. EventSource (job logs, stream-run events) only: `?token=`.
- Lab artifacts play from `/api/lab/play/<id>/<key>/…`: the key is an HMAC of the run id under `LAIL_TOKEN`, so iframes need no token, a link opens only that run, and rotating the token revokes links. Artifacts are served with a strict CSP and `sandbox allow-scripts` (opaque origin), so a game can never read the dashboard's storage or token. Consequences: artifact responses carry `Access-Control-Allow-Origin: *` (no credentials) so the game's own module scripts, `fetch()`ed data, WASM and fonts load; and HTML artifacts get a small in-memory `localStorage`/`sessionStorage` injected ahead of their own scripts (real storage throws in an opaque origin), so a game that keeps a high score still runs; it is lost on reload and IndexedDB/cookies stay unavailable. Non-web files in a private run (e.g. `.py`) download as `application/octet-stream`; public shares serve web assets only.
- CORS trusts only the web origin (`:3000`, `LAIL_WEB_PORT`, `LAIL_CORS_ORIGINS`). With or without a token, a cross-site non-GET is refused unless it is JSON (preflighted), by the controller and by the serve-engine alike. curl and Hermes send no `Origin` and are unaffected.
- Without a token, the controller and the serve-engine also refuse a request whose `Host` (or `X-Forwarded-Host`) is a dotted name other than `localhost` or a `LAIL_CORS_ORIGINS` host (DNS rebinding). IP literals and single-label names (`spark1`, docker service names) pass.
- The `/v1` proxy never forwards `LAIL_TOKEN` or cookies to the model backend.
- The web UI does **not** inject the operator secret. If a request returns 401, paste `LAIL_TOKEN` into the banner; it is kept in `sessionStorage` and sent as `X-Lail-Token` / `?token=`.
- Docker Compose publishes `127.0.0.1:PORT:PORT` and sets `LAIL_INSECURE_BIND=1` for the container-internal `0.0.0.0` listen. If you publish those ports on `0.0.0.0`, set `LAIL_TOKEN` and enter it in the UI. Do not rely on the escape hatch.
