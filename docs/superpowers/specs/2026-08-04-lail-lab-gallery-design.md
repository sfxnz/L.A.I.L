# L.A.I.L Lab Runs & Artifact Gallery

**Status:** design (2026-08-04)  
**Goal:** Serve local models → eval → real Hermes tasks (HTML games, animations, small apps) → **browse, compare, and share results** inside L.A.I.L.

---

## Product loop

```text
1. Server     serve model on Spark (vLLM :8000)
2. Evals      tool-eval / smoke / perf  → scores in /evals
3. Tasks      Hermes (local or via Connect) builds real artifacts
4. Gallery    /lab  — every run is a card: model + task + scores + playable output
5. Share      public or Tailscale link to play HTML / view animation
```

One principle: **the model is only “good” if it ships something you can open** — not just a number on a board.

---

## What already exists

| Piece | Today |
|--------|--------|
| Serve | `/server` Lab Safe / Workflow Max |
| Tool evals | `/evals/tool` leaderboard + compare 2–4 models |
| Run JSON | `data/runs/*.json` + `/api/runs` |
| Connect | `/connect` OpenAI base URL for Hermes |
| Workspaces | `workspaces/demo/` scratch files |

**Gap:** no first-class **task run** (prompt → Hermes → artifact folder) and no **gallery** that pairs model identity + eval scores + playable files.

---

## Target surfaces

### 1. `/lab` — Artifact gallery (primary home for “what did models do?”)

Card grid, filterable by:

- model id  
- task type (`html-game`, `animation`, `svg`, `mini-app`, …)  
- date  
- tags (e.g. `spark-safe`, `no-network`)

Each card shows:

- thumbnail / iframe preview (HTML) or static frame  
- model + quant + serve notes  
- optional eval summary (tool score if linked)  
- **Open** (fullscreen play) · **Compare** · **Files** · **Copy share link**

### 2. `/lab/[runId]` — Run detail

- Prompt / task brief (what Hermes was asked)  
- Model + Hermes session link if any  
- Artifact tree (`index.html`, assets, …)  
- **Live preview** (`iframe` sandbox)  
- Raw logs / tool timeline (optional)  
- Link to parent **tool-eval run** if this model was evaluated the same week  

### 3. `/lab/compare?ids=a,b,c`

Side-by-side iframes + same task brief — “three models, one game prompt.”

### 4. Share / play

| Mode | Audience | Mechanism |
|------|----------|-----------|
| **Private** | You on Tailscale | `http://<tailnet-host>:3000/lab/<id>` |
| **LAN play** | Friends on home network | same host, optional auth off for `/lab/public/*` only |
| **Public** | Internet | later: signed URL or Cloudflare tunnel to **static export only** (not full L.A.I.L) |

Public path must **never** expose serve-engine, Hermes API keys, or lab sqlite — only static artifact bundles.

---

## Data model

```text
data/lab-runs/<run_id>/
  meta.json          # model, task, timestamps, scores, hermes_session, tags
  brief.md           # task prompt
  artifacts/         # shipped files (index.html, …)
  preview.png        # optional screenshot
  logs/              # optional agent log excerpt
```

`meta.json` sketch:

```json
{
  "id": "20260804T153012Z_ab12",
  "kind": "hermes_task",
  "task_type": "html-game",
  "title": "Geometry Dash–like runner",
  "model_id": "nvidia/Qwen3.6-27B-NVFP4",
  "serve": { "util": 0.4, "max_model_len": 65536, "image": "v0.26.0" },
  "eval_run_id": "20260730T110350Z_1abffa",
  "hermes": { "session": "…", "source": "desktop|glass|matrix" },
  "created_at": "2026-08-04T15:30:12Z",
  "entry": "artifacts/index.html",
  "share": { "public": false, "slug": null },
  "tags": ["html", "game", "self-contained"]
}
```

Index in sqlite (or scan dirs) for `/api/lab/runs` list + filter.

---

## How Hermes produces a lab run

### A. Manual drop (v0 — ship first)

1. Hermes writes HTML under `workspaces/<task>/` or `/tmp`.  
2. You (or a script) run:

```bash
lail-lab import \
  --model nvidia/Qwen3.6-27B-NVFP4 \
  --task html-game \
  --title "Runner v1" \
  --entry index.html \
  --from ./workspaces/demo/geometry-dash-like.html
```

3. Appears in `/lab`.

### B. Hermes skill / tool (v1)

Hermes tool `lail_lab_publish`:

- args: title, task_type, entry path, tags  
- reads model id from env / Connect endpoint  
- copies artifacts into `data/lab-runs/<id>/`  
- returns gallery URL  

So after “build me a self-contained HTML game”, Hermes ends with:  
**Published → https://spark…/lab/20260804…**

### C. L.A.I.L Workbench task templates (v2)

Server tab or Lab tab: **Run task** presets:

- Self-contained HTML game (one file, no CDN)  
- CSS/JS animation (no external fonts)  
- SVG explainer  
- Mini dashboard  

Spawns Hermes (or in-process agent) with fixed brief + `lail_lab_publish` at end.

---

## Compare models fairly

Same **brief.md** hash → multiple runs:

```text
task_fingerprint = sha256(brief.md + task_type)
```

Gallery filter “this task” shows all models that attempted it.  
Compare view loads N iframes + model badges + optional tool-eval score column.

---

## Security / share rules

1. Preview iframe: `sandbox="allow-scripts allow-same-origin"` carefully — prefer `allow-scripts` only for untrusted model HTML if possible; games may need more — document risk.  
2. Public share = zip/static host of `artifacts/` only.  
3. No env files, no `.hermes`, no API keys in artifacts (scan on publish).  
4. Optional simple password on `/lab/public/*`.

---

## UI fit (dark Apple-minimal)

- Nav: **Lab** between Evals and Connect  
- Status page: “Latest lab runs” strip (3 cards)  
- Evals tool detail: “Artifacts from this model” if `eval_run_id` links  

---

## Implementation phases

| Phase | Deliverable | Effort |
|-------|-------------|--------|
| **P0** | `data/lab-runs/` layout + `POST/GET /api/lab/runs` + import CLI | small |
| **P1** | `/lab` grid + `/lab/[id]` preview player | medium |
| **P2** | Hermes `lail_lab_publish` skill/tool | small |
| **P3** | Compare + link to tool-eval board | medium |
| **P4** | Public static share (Tailscale funnels or export zip) | medium |

---

## Success criteria

1. Serve model in L.A.I.L → tool-eval → Hermes builds HTML game → **one click** later you reopen that exact game with model label.  
2. Two models, same brief → side-by-side play.  
3. Friend on Tailscale (or public link) can **play** without SSH.  
4. No confusion with tool-eval JSON-only runs — lab runs are **artifacts-first**.

---

## Non-goals (for now)

- Full multiplayer game hosting  
- App Store packaging  
- Training loops inside gallery  
- Replacing Hermes Desktop (Connect remains the pointer)

---

## Next build step

Implement **P0 + P1**: storage + API + `/lab` gallery with iframe preview, seed with existing `workspaces/demo/*.html` if present.
