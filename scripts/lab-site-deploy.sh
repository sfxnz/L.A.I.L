#!/usr/bin/env bash
# Publish L.A.I.L lab artifacts the Wesche way: static site → GitHub Pages.
#
# First time:
#   1. Create empty public repo (e.g. YOU/your-lab-site) with Pages from gh-pages branch
#   2. export LAIL_SITE_REPO=git@github.com:YOU/your-lab-site.git
#   3. export LAIL_SITE_BASE=https://YOU.github.io/your-lab-site   # or custom domain
#   4. bun run lab:site-deploy
#
# After that, each deploy rebuilds site/dist and force-pushes gh-pages.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.bun/bin:${PATH}"

SITE_REPO="${LAIL_SITE_REPO:-}"
SITE_BASE="${LAIL_SITE_BASE:-}"
BRANCH="${LAIL_SITE_BRANCH:-gh-pages}"
OUT="${LAIL_SITE_OUT:-$ROOT/site/dist}"

if [[ -z "$SITE_REPO" ]]; then
  echo "Set LAIL_SITE_REPO to your GitHub Pages repo, e.g.:" >&2
  echo "  export LAIL_SITE_REPO=git@github.com:YOU/your-lab-site.git" >&2
  echo "  export LAIL_SITE_BASE=https://YOU.github.io/your-lab-site" >&2
  echo "  # optional custom domain: LAIL_SITE_BASE=https://lab.example.com" >&2
  exit 2
fi

echo "→ build static site"
LAIL_SITE_BASE="$SITE_BASE" bun run scripts/lab-site-build.ts >/tmp/lab-site-build.json
echo "   out: $OUT"
python3 - <<'PY' 2>/dev/null || true
import json
d=json.load(open("/tmp/lab-site-build.json"))
print(f"   items: {d.get('count')}")
for e in (d.get("entries") or [])[:5]:
    print(f"   - {e.get('url') or e.get('path')}")
PY

# Optional CNAME from env
if [[ -n "${LAIL_SITE_CNAME:-}" ]]; then
  echo "$LAIL_SITE_CNAME" > "$OUT/CNAME"
  echo "   CNAME: $LAIL_SITE_CNAME"
fi

WORKDIR="$(mktemp -d /tmp/lail-site-XXXXXX)"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

echo "→ push $BRANCH → $SITE_REPO"
git clone --depth 1 --branch "$BRANCH" "$SITE_REPO" "$WORKDIR" 2>/dev/null \
  || git clone --depth 1 "$SITE_REPO" "$WORKDIR"

cd "$WORKDIR"
git checkout -B "$BRANCH" >/dev/null 2>&1 || true
# replace tree with dist
find . -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
cp -a "$OUT"/. .

git add -A
if git diff --cached --quiet; then
  echo "No site changes to deploy."
else
  git -c user.email="${LAIL_SITE_GIT_EMAIL:-lab@local}" \
      -c user.name="${LAIL_SITE_GIT_NAME:-L.A.I.L Lab}" \
      commit -m "deploy: lab site $(date -u +%Y-%m-%dT%H%MZ)"
  git push -u origin "HEAD:$BRANCH" --force
  echo "→ deployed"
fi

echo
echo "Public gallery:"
echo "  ${SITE_BASE:-$SITE_REPO}/"
echo "Per-build URLs are in catalog.json / lab:site-build output."
echo
echo "Persist in L.A.I.L .env:"
echo "  LAIL_SITE_REPO=$SITE_REPO"
echo "  LAIL_SITE_BASE=$SITE_BASE"
echo "  # restart bun run dev after setting"
