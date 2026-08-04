#!/usr/bin/env bash
# Expose ONLY the lab-public static server to the internet via Tailscale Funnel.
# NEVER funnels L.A.I.L :3000, controller :8787, Hermes, or vLLM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.bun/bin:${PATH}"

PORT="${LAIL_SHARE_PORT:-8791}"
HOST="127.0.0.1"

if [[ "${PORT}" == "3000" || "${PORT}" == "8787" || "${PORT}" == "8000" || "${PORT}" == "8642" ]]; then
  echo "REFUSING to funnel lab/admin port ${PORT}" >&2
  exit 2
fi

# Ensure public file server is up on LOOPBACK only
if ! curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "→ starting lab-public-server on ${HOST}:${PORT}"
  mkdir -p data/lab-public logs
  # Prefer managed restart via bun run dev; this is a fallback
  if command -v bun >/dev/null; then
    bun run scripts/lab-public-server.ts >>logs/lab-public-server.log 2>&1 &
    echo $! > data/lab-public-server.pid
  fi
  for _ in $(seq 1 30); do
    curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1 && break
    sleep 0.2
  done
fi

if ! curl -sf "http://${HOST}:${PORT}/health" >/dev/null; then
  echo "public server failed — start lab (bun run dev) or: bun run lab:share-server" >&2
  exit 1
fi

# Prove we are not pointing Funnel at the full lab by accident
SS_LINE="$(ss -ltnp 2>/dev/null | grep ":${PORT} " || true)"
if echo "$SS_LINE" | grep -qE '0\.0\.0\.0|:::' ; then
  echo "REFUSING: port ${PORT} is bound on all interfaces. Must be 127.0.0.1 only." >&2
  echo "$SS_LINE" >&2
  exit 2
fi

echo "→ Tailscale Funnel → http://${HOST}:${PORT} ONLY (artifacts)"
echo "   (resets any prior funnel/serve config first)"

if ! tailscale funnel reset >/dev/null 2>&1; then
  echo "Need operator rights. Run once:" >&2
  echo "  sudo tailscale set --operator=\$USER" >&2
  echo "  sudo tailscale funnel reset" >&2
  exit 1
fi
tailscale serve reset >/dev/null 2>&1 || true

if ! tailscale funnel --bg "http://${HOST}:${PORT}"; then
  echo "" >&2
  echo "Funnel failed. Often means Funnel is off for the tailnet." >&2
  echo "Enable here, then re-run: bun run lab:funnel" >&2
  echo "  https://login.tailscale.com/admin/acls" >&2
  echo "  (Funnel node approval / https://login.tailscale.com/f/funnel )" >&2
  exit 1
fi

echo
tailscale funnel status || true
echo

FUNNEL_HOST="$(tailscale status --json 2>/dev/null | bun -e '
  const j=JSON.parse(await Bun.stdin.text());
  console.log((j.Self?.DNSName||"").replace(/\.$/,""));
' 2>/dev/null || true)"

if [[ -n "${FUNNEL_HOST}" ]]; then
  BASE="https://${FUNNEL_HOST}"
  echo "Internet base:"
  echo "  LAIL_SHARE_PUBLIC_BASE=${BASE}"
  ENVF="${ROOT}/.env"
  touch "$ENVF"
  if grep -q '^LAIL_SHARE_PUBLIC_BASE=' "$ENVF" 2>/dev/null; then
    sed -i "s|^LAIL_SHARE_PUBLIC_BASE=.*|LAIL_SHARE_PUBLIC_BASE=${BASE}|" "$ENVF"
  else
    echo "LAIL_SHARE_PUBLIC_BASE=${BASE}" >> "$ENVF"
  fi
  echo "→ saved to .env — restart bun run dev so share links use Funnel HTTPS"
  echo
  echo "X play link shape:"
  echo "  ${BASE}/s/<slug>/index.html"
fi

echo
echo "Security checklist:"
echo "  ✓ Funnel target is loopback :${PORT} (artifacts only)"
echo "  ✗ Do NOT funnel :3000 or :8787"
echo "  Off:  tailscale funnel reset"
