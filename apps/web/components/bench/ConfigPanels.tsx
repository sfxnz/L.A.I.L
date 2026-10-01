"use client";

import { fmtSize } from "@/lib/bench/format";
import {
  PREFILL_SIZES,
  TOKENS_PER_STREAM,
  sortConcurrencies,
  sortSizes,
  toggleLevel,
  toggleSize,
  type DecodeConfig,
  type PrefillConfig,
} from "@/lib/bench/levels";
import type { StreamPack } from "@/lib/stream-run-types";
import { Btn, Eyebrow, Input } from "@/components/ui";
import { cn } from "@/lib/utils";
import { Chip, LevelGrid } from "./Chip";

/** The left column. Chips lock while a sync runs; Run/Stop carry their keys. */

function RunStop({
  running,
  canRun,
  reason,
  onRun,
  onStop,
  starting,
}: {
  running: boolean;
  canRun: boolean;
  reason?: string;
  onRun: () => void;
  onStop: () => void;
  starting: boolean;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {running ? (
        <Btn variant="danger" onClick={onStop} title="Stop (⌘.)">
          Stop
        </Btn>
      ) : (
        <Btn onClick={onRun} disabled={!canRun} loading={starting} title={canRun ? "Run (r · ⌘↩)" : reason}>
          Run
        </Btn>
      )}
      <Eyebrow className="lab-num text-[9px]">{running ? "⌘. stops" : "r · ⌘↩"}</Eyebrow>
      {!running && !canRun && reason ? <span className="text-[11px] text-lab-muted">{reason}</span> : null}
    </div>
  );
}

const PACK_ORDER = ["prose", "structured", "code", "json"];

export function DecodeConfigPanel({
  packs,
  cfg,
  onChange,
  locked,
  running,
  starting,
  canRun,
  reason,
  onRun,
  onStop,
  className,
}: {
  packs: StreamPack[];
  cfg: DecodeConfig;
  onChange: (next: DecodeConfig) => void;
  locked: boolean;
  running: boolean;
  starting: boolean;
  canRun: boolean;
  reason?: string;
  onRun: () => void;
  onStop: () => void;
  className?: string;
}) {
  const selected = new Set(cfg.levels);
  const shown = packs.length
    ? [...packs].sort((a, b) => (PACK_ORDER.indexOf(a.id) + 99) % 99 - ((PACK_ORDER.indexOf(b.id) + 99) % 99))
    : PACK_ORDER.map((id) => ({ id, label: id[0].toUpperCase() + id.slice(1), prompts: [], default_max_tokens: 256 }));
  return (
    <div className={cn("space-y-5", className)}>
      <section>
        <Eyebrow className="mb-2 block">Pack</Eyebrow>
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Prompt pack">
          {shown.map((p) => (
            <Chip key={p.id} on={cfg.pack === p.id} disabled={locked} onClick={() => onChange({ ...cfg, pack: p.id })} title={p.prompts[0]?.text.slice(0, 120)}>
              {p.label}
            </Chip>
          ))}
        </div>
      </section>

      <section>
        <div className="mb-2 flex items-center justify-between gap-2">
          <Eyebrow>Levels</Eyebrow>
          <span className="lab-num font-mono text-[10px] text-lab-muted">{cfg.levels.map((l) => `×${l}`).join(" → ")}</span>
        </div>
        <LevelGrid selected={selected} disabled={locked} onToggle={(n) => onChange({ ...cfg, levels: sortConcurrencies(toggleLevel(selected, n)) })} />
        <p className="mt-1.5 text-[10px] leading-snug text-lab-muted">Keys 1–6 toggle ×1 ×2 ×4 ×8 ×16 ×32. Levels run ascending after one warmup request, each with at least 3 strands (×1 three waves, ×2 two, ×4+ one); numbers are medians with min–max.</p>
      </section>

      <section>
        <Eyebrow className="mb-2 block">Tokens / stream</Eyebrow>
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Tokens per stream">
          {[...new Set([...TOKENS_PER_STREAM, cfg.maxTokens])]
            .sort((a, b) => a - b)
            .map((t) => (
              <Chip key={t} on={cfg.maxTokens === t} disabled={locked} onClick={() => onChange({ ...cfg, maxTokens: t })}>
                {t}
              </Chip>
            ))}
        </div>
      </section>

      <section className="grid grid-cols-2 gap-3">
        <label className="block min-w-0">
          <Eyebrow className="mb-1.5 block">Interactivity floor</Eyebrow>
          <div className="flex items-center gap-2">
            <Input
              type="number"
              min={0}
              step={1}
              inputMode="numeric"
              value={cfg.floor}
              disabled={locked}
              onChange={(e) => onChange({ ...cfg, floor: Math.max(0, Number(e.target.value) || 0) })}
              aria-label="Interactivity floor, tok/s per stream"
              className="lab-num h-8 min-w-0 flex-1 py-1 font-mono text-[12px]"
            />
            <span className="shrink-0 font-mono text-[10px] text-lab-muted">tok/s</span>
          </div>
        </label>
        <label className="block min-w-0">
          <Eyebrow className="mb-1.5 block">TTFT SLO</Eyebrow>
          <div className="flex items-center gap-2">
            <Input
              type="number"
              min={0}
              step={50}
              inputMode="numeric"
              value={cfg.sloMs}
              disabled={locked}
              onChange={(e) => onChange({ ...cfg, sloMs: Math.max(0, Number(e.target.value) || 0) })}
              aria-label="TTFT SLO, milliseconds"
              className="lab-num h-8 min-w-0 flex-1 py-1 font-mono text-[12px]"
            />
            <span className="shrink-0 font-mono text-[10px] text-lab-muted">ms</span>
          </div>
        </label>
        <p className="col-span-2 text-[10px] leading-snug text-lab-muted">
          Drive the interpretation and knee shading; change them any time, the curve recolours.
        </p>
      </section>

      <RunStop running={running} canRun={canRun} reason={reason} onRun={onRun} onStop={onStop} starting={starting} />
    </div>
  );
}

export function PrefillConfigPanel({
  cfg,
  onChange,
  locked,
  running,
  starting,
  canRun,
  reason,
  onRun,
  onStop,
  className,
}: {
  cfg: PrefillConfig;
  onChange: (next: PrefillConfig) => void;
  locked: boolean;
  running: boolean;
  starting: boolean;
  canRun: boolean;
  reason?: string;
  onRun: () => void;
  onStop: () => void;
  className?: string;
}) {
  const selected = new Set(cfg.sizes);
  return (
    <div className={cn("space-y-5", className)}>
      <section>
        <Eyebrow className="mb-2 block">Context sizes</Eyebrow>
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Context sizes">
          {PREFILL_SIZES.map((s) => (
            <Chip key={s} on={selected.has(s)} disabled={locked} onClick={() => onChange({ sizes: sortSizes(toggleSize(selected, s)) })} title={`${s.toLocaleString("en-US")} tokens`}>
              {fmtSize(s)}
            </Chip>
          ))}
        </div>
        <p className="mt-1.5 text-[10px] leading-snug text-lab-muted">
          Two requests per size after one warmup, each with its own unique-prefix prompt (no prefix-cache hits), max_tokens 1; the rate is the median. Sizes over max-model-len are reported skipped — never dropped.
        </p>
      </section>
      <RunStop running={running} canRun={canRun} reason={reason} onRun={onRun} onStop={onStop} starting={starting} />
    </div>
  );
}
