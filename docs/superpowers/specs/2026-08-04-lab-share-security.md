# Lab public share — security model

## Goal

People on X can play a **model-built HTML game** without getting access to Spark admin, Hermes, models, or secrets.

## Architecture

```text
Internet
   │
   ▼
Tailscale Funnel (HTTPS)
   │
   ▼
127.0.0.1:8791  lab-public-server   ← ONLY process funneled
   │
   ▼
data/lab-public/<slug>/   static files only
```

**Never funnel** L.A.I.L `:3000`, controller `:8787`, Hermes `:8642`, glass `:8766`, or vLLM `:8000`.

## Guarantees (what we enforce)

| Control | How |
|--------|-----|
| Separate process | `scripts/lab-public-server.ts` |
| Loopback bind only | Hardcoded `127.0.0.1` |
| Funnel refuses admin ports | `lab-share-funnel.sh` exits if PORT is 3000/8787/8000/8642 |
| Funnel refuses non-loopback bind | `ss` check before enable |
| Path traversal blocked | `..`, null bytes, resolve-under-root |
| Metadata not served | `share.json` / `meta.json` / dotfiles blocked |
| Extension allowlist | html/js/css/images/fonts/audio/json data only |
| GET/HEAD only | POST etc → 405 |
| Secret scan at publish | refuses API keys / private keys in artifacts |
| Browser lockdown | CSP, nosniff, no referrer, permissions-policy |
| Unpublish | deletes public slug tree |

## Residual risk (honest)

1. **Published game JS runs in the visitor’s browser.** Same as any HTML game link. CSP blocks most outbound calls; it cannot make every model output “safe code.”
2. **Slug URL is unguessable but not a password.** Anyone with the link can play while published. Unpublish when done.
3. **Funnel is global to the node path you enable.** Misconfiguration (funneling `:3000`) would be dangerous — scripts refuse known bad ports; don’t hand-roll `tailscale funnel 3000`.
4. **Full lab still on your network.** Tailnet/LAN users who can reach `:3000` already could before; Funnel does not change that if you only funnel `:8791`.
5. **No formal pentest.** This is defense-in-depth for a lab product, not a SOC2 claim.

## Operator checklist before X posts

- [ ] `ss -ltnp | grep 8791` shows **127.0.0.1 only**
- [ ] `tailscale funnel status` points at **http://127.0.0.1:8791** only
- [ ] Share URL is `https://…ts.net/s/<slug>/index.html` (not `:3000`)
- [ ] Game has no secrets (publish already scans)
- [ ] Unpublish after the thread cools if you care about long-lived links

## Off switches

```bash
tailscale funnel reset
# optional
kill "$(cat ~/projects/ai-lab/local-ai-lab/data/lab-public-server.pid)" 2>/dev/null || true
```
