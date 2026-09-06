"use client";

import { useState } from "react";
import { Btn, CheckboxRow, Eyebrow, SegmentedControl, inputCls } from "@/components/ui";
import type { StreamArrival, StreamPack, StreamThinking } from "@/lib/stream-run-types";
import { cn } from "@/lib/utils";

export type StreamControls = {
  base_url: string;
  pack: string;
  n: number;
  arrival: StreamArrival;
  max_tokens: number;
  fill_to_max: boolean;
  thinking: StreamThinking;
};

export type Endpoint = { url: string; label: string };

export const STRAND_PRESETS = [1, 2, 4, 8, 16, 32] as const;
export const TOKEN_OPTIONS = [128, 256, 512, 1024, 2048] as const;
const LOOPBACK = "__loopback";
const CUSTOM = "__custom";

function loopbackPort(url: string): string {
  try {
    const u = new URL(url);
    return u.hostname === "127.0.0.1" ? u.port : "";
  } catch {
    return "";
  }
}

function Ctl({ label, children, className }: { label: string; children: React.ReactNode; className?: string }) {
  return (
    <label className={cn("flex min-w-0 flex-col gap-1", className)}>
      <Eyebrow>{label}</Eyebrow>
      {children}
    </label>
  );
}

function Key({ k }: { k: string }) {
  return <kbd className="strand-kbd">{k}</kbd>;
}

/**
 * Load controls. Everything wraps at narrow widths; every action shows its key.
 * "Fill to max" is an explicit toggle (default off = natural EOS) because
 * forcing min_tokens distorts the text — the sparkDash default is our opt-in.
 */
export function ControlBar({
  controls,
  onChange,
  endpoints,
  packs,
  running,
  canRun,
  runDisabledReason,
  starting,
  hasRun,
  onRun,
  onStop,
  onCopyAll,
  onExportJson,
  onExportMarkdown,
  onWall,
  className,
}: {
  controls: StreamControls;
  onChange: (patch: Partial<StreamControls>) => void;
  endpoints: Endpoint[];
  packs: StreamPack[];
  running: boolean;
  canRun: boolean;
  runDisabledReason?: string;
  starting: boolean;
  hasRun: boolean;
  onRun: () => void;
  onStop: () => void;
  onCopyAll: () => void;
  onExportJson: () => void;
  onExportMarkdown: () => void;
  onWall: () => void;
  className?: string;
}) {
  const configured = endpoints.find((e) => e.url === controls.base_url);
  const selectValue = configured ? configured.url : LOOPBACK;
  const port = loopbackPort(controls.base_url) || "8000";
  const lock = running || starting;
  // "custom…" keeps the input open while the operator types a value that happens to be a preset.
  const [customTokens, setCustomTokens] = useState(false);
  const tokensPreset = !customTokens && TOKEN_OPTIONS.includes(controls.max_tokens as (typeof TOKEN_OPTIONS)[number]);

  return (
    <div className={cn("streams-controls flex flex-wrap items-end gap-x-4 gap-y-3", className)}>
      <Ctl label="Endpoint" className="w-[200px] max-w-full">
        <div className="flex gap-1">
          <select
            className={cn(inputCls, "h-8 py-0 text-[12px]")}
            value={selectValue}
            disabled={lock}
            onChange={(e) => {
              const v = e.target.value;
              onChange({ base_url: v === LOOPBACK ? `http://127.0.0.1:${port}` : v });
            }}
          >
            {endpoints.map((e) => (
              <option key={e.url} value={e.url}>
                {e.label} · {e.url.replace(/^https?:\/\//, "")}
              </option>
            ))}
            <option value={LOOPBACK}>127.0.0.1:port…</option>
          </select>
          {selectValue === LOOPBACK && (
            <input
              className={cn(inputCls, "lab-num h-8 w-[72px] py-0 font-mono text-[12px]")}
              inputMode="numeric"
              aria-label="Loopback port"
              value={port}
              disabled={lock}
              onChange={(e) => {
                const p = e.target.value.replace(/\D/g, "").slice(0, 5);
                onChange({ base_url: `http://127.0.0.1:${p || "8000"}` });
              }}
            />
          )}
        </div>
      </Ctl>

      <Ctl label="Pack" className="w-[150px]">
        <select
          className={cn(inputCls, "h-8 py-0 text-[12px]")}
          value={controls.pack}
          disabled={lock}
          onChange={(e) => onChange({ pack: e.target.value })}
        >
          {packs.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
      </Ctl>

      <div className="flex min-w-0 flex-col gap-1">
        <Eyebrow>Strands</Eyebrow>
        <div className="flex items-center gap-1.5">
          <div className="animus-chamfer-sm flex h-8 items-stretch border border-lab-border bg-lab-input">
            <button
              type="button"
              className="strand-step"
              aria-label="Fewer strands"
              disabled={lock || controls.n <= 1}
              onClick={() => onChange({ n: Math.max(1, controls.n - 1) })}
            >
              −
            </button>
            <input
              className="lab-num w-9 bg-transparent text-center font-mono text-[12px] text-lab-text outline-none"
              inputMode="numeric"
              aria-label="Strands"
              value={controls.n}
              disabled={lock}
              onChange={(e) => {
                const v = Number(e.target.value.replace(/\D/g, ""));
                if (v >= 1 && v <= 32) onChange({ n: v });
              }}
            />
            <button
              type="button"
              className="strand-step"
              aria-label="More strands"
              disabled={lock || controls.n >= 32}
              onClick={() => onChange({ n: Math.min(32, controls.n + 1) })}
            >
              +
            </button>
          </div>
          <SegmentedControl
            size="sm"
            ariaLabel="Strand presets"
            value={String(STRAND_PRESETS.includes(controls.n as (typeof STRAND_PRESETS)[number]) ? controls.n : "")}
            onChange={(v) => !lock && onChange({ n: Number(v) })}
            options={STRAND_PRESETS.map((p) => ({ id: String(p), label: String(p), disabled: lock }))}
          />
        </div>
      </div>

      <div className="flex flex-col gap-1">
        <Eyebrow>Arrival</Eyebrow>
        <SegmentedControl
          size="sm"
          ariaLabel="Arrival"
          value={controls.arrival}
          onChange={(v) => !lock && onChange({ arrival: v })}
          options={[
            { id: "burst", label: "burst", disabled: lock },
            { id: "staggered", label: "staggered", disabled: lock },
            { id: "poisson", label: "poisson", disabled: lock },
          ]}
        />
      </div>

      <Ctl label="Tokens" className="w-[150px]">
        <div className="flex gap-1">
          <select
            className={cn(inputCls, "lab-num h-8 py-0 font-mono text-[12px]")}
            value={tokensPreset ? controls.max_tokens : CUSTOM}
            disabled={lock}
            onChange={(e) => {
              const v = e.target.value;
              if (v === CUSTOM) setCustomTokens(true);
              else {
                setCustomTokens(false);
                onChange({ max_tokens: Number(v) });
              }
            }}
          >
            {TOKEN_OPTIONS.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
            <option value={CUSTOM}>custom…</option>
          </select>
          {!tokensPreset && (
            <input
              className={cn(inputCls, "lab-num h-8 w-[64px] py-0 font-mono text-[12px]")}
              inputMode="numeric"
              aria-label="Max tokens"
              value={controls.max_tokens}
              disabled={lock}
              onChange={(e) => {
                const v = Number(e.target.value.replace(/\D/g, ""));
                if (v >= 1 && v <= 65_536) onChange({ max_tokens: v });
              }}
            />
          )}
        </div>
      </Ctl>

      <div className="flex flex-col gap-1">
        <Eyebrow>Budget</Eyebrow>
        <div className="-mx-1.5 -my-1">
          <CheckboxRow checked={controls.fill_to_max} disabled={lock} onChange={(v) => onChange({ fill_to_max: v })} id="streams-fill">
            <span className="text-lab-text">Fill to max</span>
            <span className="block text-[10px] leading-tight text-lab-muted">forces min_tokens · distorts text · off = natural EOS</span>
          </CheckboxRow>
        </div>
      </div>

      <div className="flex flex-col gap-1">
        <Eyebrow>Thinking</Eyebrow>
        <SegmentedControl
          size="sm"
          ariaLabel="Thinking"
          value={controls.thinking}
          onChange={(v) => !lock && onChange({ thinking: v })}
          options={[
            { id: "auto", label: "auto", disabled: lock },
            { id: "on", label: "on", disabled: lock },
            { id: "off", label: "off", disabled: lock },
          ]}
        />
      </div>

      <div className="ml-auto flex flex-wrap items-center gap-2">
        <Btn variant="primary" size="sm" onClick={onRun} disabled={!canRun || lock} loading={starting} title={runDisabledReason ?? "Run (r)"}>
          Run <Key k="r" />
        </Btn>
        <Btn variant="danger" size="sm" onClick={onStop} disabled={!running} title="Stop (⌘.)">
          Stop <Key k="⌘." />
        </Btn>
        <Btn variant="secondary" size="sm" onClick={onCopyAll} disabled={!hasRun} title="Copy all transcripts as Markdown (⌘⇧C)">
          Copy all <Key k="⌘⇧C" />
        </Btn>
        <span className="flex items-center gap-0.5" role="group" aria-label="Export">
          <Btn variant="ghost" size="sm" onClick={onExportJson} disabled={!hasRun} title="Download the run snapshot as JSON">
            JSON
          </Btn>
          <Btn variant="ghost" size="sm" onClick={onExportMarkdown} disabled={!hasRun} title="Download a Markdown summary">
            MD
          </Btn>
        </span>
        <Btn variant="ghost" size="sm" onClick={onWall} title="Wall mode — chrome hidden, large type (?wall=1)">
          Wall
        </Btn>
      </div>
    </div>
  );
}
