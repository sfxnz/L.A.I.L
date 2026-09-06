"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { Badge, Btn, Eyebrow, Tick } from "@/components/ui";
import type { StrandView } from "@/lib/use-stream-run";
import { fmtInt, fmtMs, fmtPct, fmtRate } from "@/lib/streams/format";
import { itlStats } from "@/lib/streams/stalls";
import { ItlSparkline } from "./ItlSparkline";

/** Full transcript of one strand: prompt, thinking (collapsed), output, metrics. */
export function TranscriptSheet({
  strand,
  packLabel,
  tokensPerChunk,
  open,
  onOpenChange,
  onCopy,
}: {
  strand: StrandView | null;
  packLabel: string;
  tokensPerChunk: number;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCopy: (i: number) => void;
}) {
  const s = strand;
  const itl = itlStats(s?.itl_ms);
  const share = s && s.chunks ? s.reasoning_chunks / s.chunks : 0;
  return (
    <Dialog.Root open={open && !!s} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="streams-sheet-overlay fixed inset-0 z-40" />
        <Dialog.Content
          className="streams-sheet lab-card animus-bracketed fixed inset-y-3 right-3 z-50 flex w-[min(720px,calc(100vw-24px))] flex-col overflow-hidden"
          aria-describedby={undefined}
        >
          {s && (
            <>
              <div className="flex shrink-0 items-center gap-3 border-b border-lab-border-subtle px-4 py-3">
                <span className="lab-num font-mono text-[11px] text-lab-muted">{String(s.i + 1).padStart(2, "0")}</span>
                <Dialog.Title className="font-[family-name:var(--font-display)] text-[13px] font-semibold uppercase tracking-[0.14em] text-lab-text">
                  {packLabel} <span className="text-lab-muted">· {s.title}</span>
                </Dialog.Title>
                <Badge tone={s.state === "done" ? "ok" : s.state === "error" ? "danger" : s.state === "decode" ? "accent" : "muted"}>{s.state}</Badge>
                <div className="ml-auto flex items-center gap-2">
                  <Btn variant="secondary" size="sm" onClick={() => onCopy(s.i)}>
                    Copy
                  </Btn>
                  <Dialog.Close asChild>
                    <Btn variant="ghost" size="sm" aria-label="Close transcript (esc)">
                      Close <kbd className="strand-kbd">esc</kbd>
                    </Btn>
                  </Dialog.Close>
                </div>
              </div>

              <div className="lab-num flex shrink-0 flex-wrap items-center gap-x-4 gap-y-1 border-b border-lab-border-subtle px-4 py-2 font-mono text-[11px] text-lab-text-dim">
                <span>{fmtInt(s.tokens ?? Math.round(s.chunks * tokensPerChunk))} tok</span>
                <Tick />
                <span>{fmtRate(s.tok_s)} tok/s{s.peak_tok_s ? ` · pk ${fmtRate(s.peak_tok_s)}` : ""}</span>
                <Tick />
                <span>ttft {fmtMs(s.ttft_ms)}</span>
                <Tick />
                <span>itl p50/p95 {itl.p50 === null ? "—" : `${Math.round(itl.p50)} / ${Math.round(itl.p95 ?? itl.p50)} ms`}</span>
                {itl.stalls > 0 && (
                  <>
                    <Tick />
                    <span className="text-lab-warn">{itl.stalls} stall{itl.stalls === 1 ? "" : "s"} ≥ 2 s</span>
                  </>
                )}
                {s.finish_reason && (
                  <>
                    <Tick />
                    <span>finish {s.finish_reason}</span>
                  </>
                )}
                {s.error && (
                  <>
                    <Tick />
                    <span className="text-lab-danger">{s.error}</span>
                  </>
                )}
                <ItlSparkline itl={s.itl_ms} width={160} height={22} className="ml-auto w-40" />
              </div>

              <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3 font-mono text-[12px] leading-relaxed">
                <Eyebrow>Prompt</Eyebrow>
                <pre className="mt-1 mb-4 border-l-2 border-l-lab-line-2 pl-3 whitespace-pre-wrap break-words text-lab-text-dim">{s.prompt}</pre>
                {s.reasoning && (
                  <details className="mb-4">
                    <summary className="strand-chip cursor-pointer">
                      thinking {fmtPct(share)} · {fmtInt(Math.round(s.reasoning_chunks * tokensPerChunk))} tok
                    </summary>
                    <pre className="mt-2 border-l-2 border-l-lab-border pl-3 whitespace-pre-wrap break-words text-lab-muted">{s.reasoning}</pre>
                  </details>
                )}
                <Eyebrow>Output</Eyebrow>
                <pre className="mt-1 whitespace-pre-wrap break-words text-lab-text">{s.text || <span className="text-lab-muted">(no output)</span>}</pre>
              </div>
            </>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
