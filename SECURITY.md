# Security

## Reporting

Report vulnerabilities privately via GitHub Security Advisories on this repository. Do not file a public issue for an unpatched security problem.

## Secrets and lab access

Never commit:

- `.env` or live tokens (`HF_TOKEN`, API keys)
- Cluster inventory (`LAIL_CLUSTER_JSON` values, `data/cluster.json`, `data/multinode_serve.json`)
- Anything that exposes Docker on the host (`docker.sock`, remote Docker URLs, or credentials that can reach them)
- SQLite databases and run logs under `data/`

Use [`.env.example`](./.env.example) as the template. Treat `HF_TOKEN`, cluster JSON, and Docker access as operator secrets.
