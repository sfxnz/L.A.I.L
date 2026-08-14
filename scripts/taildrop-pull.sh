#!/usr/bin/env bash
# Pull Tailscale Taildrops into the lab inbox agents can read.
# Once per machine (password prompt): sudo tailscale set --operator=$USER
set -euo pipefail

INBOX="${LAIL_TAILDROP_INBOX:-$HOME/projects/ai-lab/incoming-screenshots}"
mkdir -p "$INBOX"

# Screenshots the Mac already dropped into ~/Downloads (often root-owned, world-readable).
if [ -d "$HOME/Downloads" ]; then
  find "$HOME/Downloads" -maxdepth 1 -type f \( \
      -iname 'Screenshot *' -o -iname 'screenshot*' \
    \) -print0 2>/dev/null \
    | while IFS= read -r -d '' f; do
        dest="$INBOX/$(basename "$f")"
        if [ ! -e "$dest" ]; then
          cp -n "$f" "$dest" 2>/dev/null || true
        fi
      done
fi

err=$(mktemp)
if ! tailscale file get --conflict=rename "$INBOX" 2>"$err"; then
  if grep -qiE 'access denied|file access denied' "$err"; then
    echo "Tailscale inbox is root-only. Run this once, then re-run $0:" >&2
    echo "  sudo tailscale set --operator=\$USER" >&2
  else
    cat "$err" >&2
  fi
  rm -f "$err"
else
  rm -f "$err"
fi

echo "Inbox: $INBOX"
ls -lt "$INBOX" | head -12
