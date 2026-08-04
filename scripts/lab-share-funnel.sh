#!/usr/bin/env bash
# Expose ONLY the lab-public static server to the internet via Tailscale Funnel.
# Does NOT funnel L.A.I.L :3000 / Hermes / serve-engine.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.bun/bin:${PATH}"

PORT="${LAIL_SHARE_PORT:-8791}"
HOST="127.0.0.1"

# Ensure public file server is up
if ! curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "→ starting lab-public-server on ${HOST}:${PORT}"
  mkdir -p data/lab-public logs
  nohup bun run scripts/lab-public-server.ts >>logs/lab-public-server.log 2>&1 &
  echo $! > data/lab-public-server.pid
  for i in $(seq 1 20); do
    curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1 && break
    sleep 0.25
  done
fi

if ! curl -sf "http://${HOST}:${PORT}/health" >/dev/null; then
  echo "public server failed to start — see logs/lab-public-server.log" >&2
  exit 1
fi

echo "→ configuring Tailscale Funnel → http://${HOST}:${PORT} (artifacts only)"
# Reset prior funnel/serve to avoid exposing whole lab by accident
tailscale funnel reset >/dev/null 2>&1 || true
tailscale serve reset >/dev/null 2>&1 || true

# Funnel the dedicated public port only
tailscale funnel --bg "http://${HOST}:${PORT}"

echo
tailscale funnel status || true
echo
# MagicDNS / funnel hostname
FUNNEL_HOST="$(tailscale status --json 2>/dev/null | bun -e '
  const j=JSON.parse(await Bun.stdin.text());
  const self=j.Self||{};
  const dns=self.DNSName||"";
  console.log(dns.replace(/\.$/, ""));
' 2>/dev/null || true)"

if [[ -n "${FUNNEL_HOST}" ]]; then
  BASE="https://${FUNNEL_HOST}"
  echo "Internet base (save this):"
  echo "  export LAIL_SHARE_PUBLIC_BASE=${BASE}"
  echo "  # add to L.A.I.L env / shell so gallery links use Funnel URLs"
  # Persist for controller if possible
  ENVF="${ROOT}/.env"
  if [[ -f "$ENVF" ]]; then
    if grep -q '^LAIL_SHARE_PUBLIC_BASE=' "$ENVF" 2>/dev/null; then
      sed -i "s|^LAIL_SHARE_PUBLIC_BASE=.*|LAIL_SHARE_PUBLIC_BASE=${BASE}|" "$ENVF"
    else
      echo "LAIL_SHARE_PUBLIC_BASE=${BASE}" >> "$ENVF"
    fi
    echo "→ wrote LAIL_SHARE_PUBLIC_BASE to .env (restart bun run dev to pick up)"
  fi
  echo
  echo "Example X link after publish:"
  echo "  ${BASE}/s/<slug>/index.html"
else
  echo "Funnel enabled. Check: tailscale funnel status"
fi

echo
echo "Off switch:"
echo "  tailscale funnel reset"
echo "  kill \$(cat data/lab-public-server.pid)  # optional"
