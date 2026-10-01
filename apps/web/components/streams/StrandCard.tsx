"use client";

import { memo, useState, type CSSProperties } from "react";
import { Badge, Eyebrow, Tick } from "@/components/ui";
import type { StrandView } from "@/lib/use-stream-run";
import { fmtInt, fmtMs, fmtPct, fmtRate } from "@/lib/streams/format";
import { itlStats, type Desync } from "@/lib/streams/stalls";
import { strandColor, strandOpacity } from "@/lib/streams/strand-color";
import { cn } from "@/lib/utils";
import { ItlSparkline } from "./ItlSparkline";
import { StrandSyncBar } from "./StrandSyncBar";
import { Transcript } from "./Transcript";

export type StrandCardProps = {
  strand: StrandView;
  packLabel: string;
  maxTokens: number;
  fillToMax: boolean;
  /** False when the server sent no per-chunk usage: steps are then chunks, not tokens. */
  tokensExact: boolean;
  /** client-observed live rate (stack share), tok/s */
  liveRate: number;
  desync: Desync;
  focused: boolean;
  hovered: boolean;
  thinkingMuted: boolean;
  wall: boolean;
  onHover: (i: number | null) => void;
  onFocus: (i: number) => void;
  onExpand: (i: number) => void;
  onCopy: (i: number) => void;
  onToggleThinking: (i: number) => void;
};

function badgeTone(state: StrandView["state"], desync: boolean): "ok" | "warn" | "danger" | "muted" | "accent" {
  if (desync) return "danger";
  switch (state) {
    case "done":
      return "ok";
    case "decode":
      return "accent";
    case "prefill":
      return "warn";
    case "cancelled":
      return "warn";
    case "error":
      return "danger";
    default:
      return "muted";
  }
}

function KeyHint({ k }: { k: string }) {
  return <kbd className="strand-kbd">{k}</kbd>;
}

export const StrandCard = memo(function StrandCard({
  strand: s,
  packLabel,
  maxTokens,
  fillToMax,
  tokensExact,
  liveRate,
  desync,
  focused,
  hovered,
  thinkingMuted,
  wall,
  onHover,
  onFocus,
  onExpand,
  onCopy,
  onToggleThinking,
}: StrandCardProps) {
  const [thinkingOpen, setThinkingOpen] = useState(false);
  const live = s.state === "decode" && !desync.desync;
  const itl = itlStats(s.step_ms, s.step_tokens);
  const share = s.chunks ? s.reasoning_chunks / s.chunks : 0;
  const thinkingTokens = s.reasoning_tokens;
  const stateWord = desync.desync ? "desync" : s.state;
  const rate = live ? liveRate : s.tok_s;
  const showThinking = s.reasoning.length > 0 && !thinkingMuted;

  return (
    <article
      className={cn(
        "strand-card lab-card animus-bracketed flex flex-col overflow-hidden",
        "before:top-[3px]! before:left-[3px]! after:right-[3px]! after:bottom-[3px]!",
        focused && "strand-card-focused",
        hovered && "strand-card-hovered",
        s.state === "waiting" && "strand-card-waiting",
      )}
      style={{ "--strand-color": strandColor(s.i), "--strand-alpha": strandOpacity(s.i) } as CSSProperties}
      data-state={s.state}
      data-desync={desync.desync ? "true" : undefined}
      data-strand={s.i}
      onMouseEnter={() => onHover(s.i)}
      onMouseLeave={() => onHover(null)}
      aria-label={`Strand ${s.i + 1} · ${packLabel} · ${stateWord}`}
    >
      <span aria-hidden className="strand-spine" />

      <header className="flex shrink-0 items-center gap-2 border-b border-lab-border-subtle px-3 py-1.5" title={s.prompt}>
        <span className="lab-num shrink-0 font-mono text-[10px] text-lab-muted">{String(s.i + 1).padStart(2, "0")}</span>
        <Eyebrow className="shrink-0 text-lab-text-dim">{packLabel}</Eyebrow>
        <span className="min-w-0 flex-1 truncate font-mono text-[10px] text-lab-muted">{s.title}</span>
        <Badge tone={badgeTone(s.state, desync.desync)}>{stateWord}</Badge>
        <span className="lab-num flex shrink-0 items-baseline gap-1 font-mono text-[11px]">
          <span className={cn("text-lab-text", live && "strand-live")}>{fmtRate(rate)}</span>
          <span className="text-[9px] text-lab-muted">tok/s</span>
          {s.peak_tok_s ? <span className="text-[9px] text-lab-muted">· pk {fmtRate(s.peak_tok_s)}</span> : null}
        </span>
      </header>

      <StrandSyncBar
        className="shrink-0 px-3 pt-1.5"
        state={s.state}
        tokens={s.tokens}
        maxTokens={maxTokens}
        fillToMax={fillToMax}
        finishReason={s.finish_reason}
        desync={desync.desync}
      />

      <Transcript
        text={s.text}
        live={live}
        empty={
          <span className="text-lab-muted">
            {s.state === "waiting" ? "waiting for arrival…" : s.state === "prefill" ? "prefill — awaiting first token…" : desync.desync ? "" : "no output"}
          </span>
        }
      >
        {showThinking && (
          <div className={cn("strand-thinking mb-1.5", thinkingOpen && "strand-thinking-open")}>
            <button
              type="button"
              className="strand-chip"
              onClick={() => setThinkingOpen((v) => !v)}
              aria-expanded={thinkingOpen}
              title={thinkingOpen ? "Collapse thinking" : "Expand thinking"}
            >
              <span aria-hidden className="strand-chip-caret">{thinkingOpen ? "▾" : "▸"}</span>
              thinking {fmtPct(share)} · {fmtInt(thinkingTokens)} tok
            </button>
            {thinkingOpen && <div className="mt-1 whitespace-pre-wrap break-words text-lab-muted">{s.reasoning}</div>}
          </div>
        )}
        {!showThinking && s.reasoning.length > 0 && (
          <div className="mb-1.5">
            <span className="strand-chip strand-chip-muted" title="Thinking muted (t)">
              thinking {fmtPct(share)} · muted
            </span>
          </div>
        )}
      </Transcript>

      <footer className="flex shrink-0 items-center gap-2 border-t border-lab-border-subtle px-3 py-1.5">
        <ItlSparkline itl={s.step_ms} width={72} height={18} className="w-[72px] shrink-0" />
        <div className="lab-num flex min-w-0 flex-1 items-center gap-1.5 overflow-hidden font-mono text-[10px] whitespace-nowrap text-lab-muted">
          <span title="Tokens" className="text-lab-text-dim">{fmtInt(s.tokens ?? 0)} tok</span>
          <Tick className="h-2.5" />
          <span title="Time to first token">ttft {fmtMs(s.ttft_ms)}</span>
          <Tick className="h-2.5" />
          <span
            title={
              tokensExact
                ? "Per-token latency p50 / p95 (ms): each decode step's gap ÷ the tokens it carried"
                : "Inter-chunk latency p50 / p95 (ms): this server sends no per-chunk token counts"
            }
          >
            {tokensExact ? "itl" : "chunk"} {itl.p50 === null ? "—" : `${Math.round(itl.p50)}/${Math.round(itl.p95 ?? itl.p50)}`}
          </span>
          {itl.stalls > 0 && (
            <>
              <Tick className="h-2.5" />
              <span className="text-lab-warn" title="Gaps ≥ 2 s with no output">{itl.stalls} stall{itl.stalls === 1 ? "" : "s"}</span>
            </>
          )}
          {(s.finish_reason || desync.desync) && (
            <>
              <Tick className="h-2.5" />
              <span className={cn("truncate", desync.desync ? "text-lab-danger" : "text-lab-text-dim")} title={desync.reason ?? s.finish_reason}>
                {desync.desync ? desync.reason : s.finish_reason}
              </span>
            </>
          )}
        </div>
      </footer>

      {!wall && (
        <div className="strand-actions" role="toolbar" aria-label={`Strand ${s.i + 1} actions`}>
          <button type="button" onClick={() => onCopy(s.i)} title="Copy transcript (c)">
            Copy <KeyHint k="c" />
          </button>
          <button type="button" onClick={() => onExpand(s.i)} title="Open transcript sheet (↩)">
            Expand <KeyHint k="↩" />
          </button>
          <button type="button" onClick={() => onFocus(s.i)} title={focused ? "Unpin (f)" : "Pin / focus (f)"} aria-pressed={focused}>
            {focused ? "Unpin" : "Pin"} <KeyHint k="f" />
          </button>
          {s.reasoning.length > 0 && (
            <button type="button" onClick={() => onToggleThinking(s.i)} title="Mute thinking (t)" aria-pressed={thinkingMuted}>
              {thinkingMuted ? "Unmute" : "Mute"} <KeyHint k="t" />
            </button>
          )}
        </div>
      )}
    </article>
  );
});
