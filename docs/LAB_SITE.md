# DGX Lab static site (Wesche-style)

Public demos are **static files on GitHub Pages** (or any static host).  
The Spark never needs to be on the internet.

## One-time setup

1. Create a **public** GitHub repo, e.g. `YOU/your-lab-site` (empty is fine).
2. Enable **Settings → Pages → Deploy from branch `gh-pages` / root**.
3. On the lab host:

```bash
cd <repo-root>

# SSH deploy key or gh auth must be able to push to that repo
export LAIL_SITE_REPO=git@github.com:YOU/dgx-lab.git
export LAIL_SITE_BASE=https://YOU.github.io/dgx-lab
# optional custom domain:
# export LAIL_SITE_BASE=https://lab.yourdomain.com
# export LAIL_SITE_CNAME=lab.yourdomain.com

# save for next time
grep -q '^LAIL_SITE_REPO=' .env 2>/dev/null || echo "LAIL_SITE_REPO=$LAIL_SITE_REPO" >> .env
grep -q '^LAIL_SITE_BASE=' .env 2>/dev/null || echo "LAIL_SITE_BASE=$LAIL_SITE_BASE" >> .env
```

## Everyday flow

```bash
# 1) Build something with a local model → publish into Lab
bun run lab:publish -- \
  --title "DNA helix" \
  --from ./out/index.html \
  --model nvidia/Qwen3.6-27B-NVFP4 \
  --brief "canvas DNA animation, no CDN" \
  --public

# 2) Build static site from all published lab-public slugs
bun run lab:site-build

# 3) Push to GitHub Pages
bun run lab:site-deploy
```

X link shape (like Wesche):

```text
https://YOU.github.io/dgx-lab/dgx/html-game/<slug>/index.html
```

Gallery home:

```text
https://YOU.github.io/dgx-lab/
```

## Compare models (same task)

Use the **same `--brief`** for each model, publish both, rebuild site.  
L.A.I.L Lab compare UI still works on Tailscale; the static site is the public play surface.

## Security

| On GitHub Pages | On Spark |
|-----------------|----------|
| Only exported HTML/JS/CSS | Private |
| No Hermes / vLLM / L.A.I.L APIs | Unchanged |
| Same class as wesche.com/dgx/* | |

Publish still runs secret scan before files enter `lab-public/`.

## Commands

| Script | Purpose |
|--------|---------|
| `bun run lab:site-build` | Generate `site/dist` |
| `bun run lab:site-deploy` | Build + force-push `gh-pages` |
| `bun run lab:publish` | Import artifact into Lab (+ lab-public) |

## Off / delete a demo

Unpublish in L.A.I.L Lab UI (removes lab-public slug), then `lab:site-deploy` again.
