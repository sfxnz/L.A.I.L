# AGENTS.md — `apps/web`

Next.js 16 App Router + React 19 + Tailwind v4. Product chrome lives in `lib/ide-chrome.ts`.

## CSS (hard)

- **Never hand-write a `-webkit-` prefix next to the standard property.** Tailwind v4 compiles through Lightning CSS, which collapses the pair and keeps only the prefixed one — the standard property is silently dropped and the effect dies in every browser. Write the standard property alone; the compiler emits both. Cost a debugging cycle on `backdrop-filter` (2026-08-08). Verify compiled output, not source: `curl -s $(curl -s http://127.0.0.1:3000/status | grep -o '/_next/static/chunks/[^"]*\.css' | head -1 | sed 's|^|http://127.0.0.1:3000|')`.
- **One `::after` per element.** `app/layout.tsx` stacks field classes on `<body>`; two classes both defining `::after` means the later one silently replaces the earlier. Grain owns `::after` (it needs its own `mix-blend-mode`); vignette + scanlines share `::before` as stacked background layers.

## Verify

UI edits are not done until the changed route has been exercised in the browser (or you state that no browser tools were available). Token 401 on a fresh profile is expected — see root `AGENTS.md`.
