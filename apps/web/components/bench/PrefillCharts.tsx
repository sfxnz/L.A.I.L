"use client";

import { useState } from "react";
import { fmtMs, fmtSize, fmtTokS } from "@/lib/bench/format";
import type { PrefillArm } from "@/lib/bench/result";
import { linear, log2Domain, niceMax, ticks } from "@/lib/bench/scale";
import type { LevelRow, StrandView } from "@/lib/use-stream-run";
import { Eyebrow } from "@/components/ui";
import { cn } from "@/lib/utils";
import { useEased, useWidth } from "./DecodeCharts";

/*
  Prefill on a log2 X (8k … 256k evenly spaced): prefill tok/s vs context on
  top, TTFT vs context beneath with a dashed LINEAR reference through the first
  point — the quadratic attention term is the gap that opens above it.
*/

const M = { top: 14, right: 14, bottom: 22, left: 46 };

type Pt = { size: number; x: number; y: number; v: number };

function pts(arms: PrefillArm[], pick: (a: PrefillArm) => number | null, x: (lx: number) => number, y: (v: number) => number): Pt[] {
  return arms
    .filter((a) => !a.skipped && a.ok > 0 && pick(a) !== null)
    .slice()
    .sort((a, b) => a.size - b.size)
    .map((a) => ({ size: a.size, v: pick(a) as number, x: x(Math.log2(a.size)), y: y(pick(a) as number) }));
}

function Segs({ p, className }: { p: Pt[]; className: string }) {
  return (
    <>
      {p.slice(1).map((b, i) => {
        const a = p[i];
        return (
          <path
            key={b.size}
            d={`M${a.x.toFixed(1)} ${a.y.toFixed(1)} L${b.x.toFixed(1)} ${b.y.toFixed(1)}`}
            pathLength={1}
            className={cn("bench-seg", className)}
            fill="none"
            strokeWidth="1.75"
            vectorEffect="non-scaling-stroke"
          />
        );
      })}
    </>
  );
}

function fmtT(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(ms % 1000 ? 1 : 0)}s` : `${Math.round(ms)}`;
}

export function PrefillCharts({ arms, sizes, className }: { arms: PrefillArm[]; sizes: number[]; className?: string }) {
  const [wrapRef, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const W = Math.max(280, width || 600);
  const H1 = 170;
  const H2 = 150;

  const keys = [...new Set([...sizes, ...arms.map((a) => a.size)])].sort((a, b) => a - b);
  const [d0, d1] = log2Domain(keys);
  const x = linear(d0, d1, M.left, W - M.right);

  const rateTarget = niceMax(Math.max(100, ...arms.map((a) => (a.ok > 0 ? a.prefillTokS ?? 0 : 0))));
  const rateMax = useEased(rateTarget);
  const y1 = linear(0, rateMax, H1 - M.bottom, M.top);
  const rate = pts(arms, (a) => a.prefillTokS, x, y1);

  const ttftDone = arms.filter((a) => !a.skipped && a.ok > 0 && a.ttftMs !== null).sort((a, b) => a.size - b.size);
  const first = ttftDone[0];
  const largestKey = keys[keys.length - 1] ?? 1;
  const linRef = first ? (size: number) => ((first.ttftMs as number) * size) / first.size : null;
  const ttftTarget = niceMax(Math.max(500, ...ttftDone.map((a) => a.ttftMs as number), linRef ? linRef(largestKey) : 0));
  const ttftMax = useEased(ttftTarget);
  const y2 = linear(0, ttftMax, H2 - M.bottom, M.top);
  const ttft = pts(arms, (a) => a.ttftMs, x, y2);

  const hovered = hover !== null ? arms.find((a) => a.size === hover) ?? null : null;
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let best: number | null = null;
    let bd = Infinity;
    for (const a of arms) {
      const d = Math.abs(x(Math.log2(a.size)) - px);
      if (d < bd) {
        bd = d;
        best = a.size;
      }
    }
    setHover(bd < 40 ? best : null);
  };
  const cursorX = hover !== null ? x(Math.log2(hover)) : null;

  const axis = (y0: number) =>
    keys.map((k) => (
      <text key={k} x={x(Math.log2(k))} y={y0 + 15} textAnchor="middle" className="fill-lab-muted font-mono" style={{ fontSize: 10 }}>
        {fmtSize(k)}
      </text>
    ));

  return (
    <div ref={wrapRef} className={cn("relative", className)}>
      <div className="mb-1 flex flex-wrap items-center gap-x-4 gap-y-1">
        <Eyebrow className="flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-[2px] w-4 bg-lab-line" /> prefill tok/s
        </Eyebrow>
        <Eyebrow className="flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-[2px] w-4 bg-lab-text-dim" /> TTFT
        </Eyebrow>
        <Eyebrow className="flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-[2px] w-4 border-t border-dashed border-lab-muted" /> linear reference
        </Eyebrow>
        {hovered && (
          <span className="lab-num ml-auto font-mono text-[11px] text-lab-text">
            {fmtSize(hovered.size)} ·{" "}
            {hovered.skipped ? (
              <span className="text-lab-muted">skipped — {hovered.skipped}</span>
            ) : (
              <>
                <span className="text-lab-line">{fmtTokS(hovered.prefillTokS)}</span> tok/s · TTFT {fmtMs(hovered.ttftMs) || "—"} ·{" "}
                {hovered.promptTokens != null ? `${hovered.promptTokens.toLocaleString("en-US")} tokens` : ""}
              </>
            )}
          </span>
        )}
      </div>

      <svg viewBox={`0 0 ${W} ${H1}`} width="100%" height={H1} className="block overflow-visible" role="img" aria-label="Prefill tok/s against context size" onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
        {ticks(rateMax, 3).map((t) => (
          <g key={t}>
            <line x1={M.left} x2={W - M.right} y1={y1(t)} y2={y1(t)} className="stroke-lab-border-subtle" strokeWidth="1" />
            <text x={M.left - 6} y={y1(t)} textAnchor="end" dominantBaseline="middle" className="fill-lab-muted font-mono" style={{ fontSize: 10 }}>
              {fmtTokS(t)}
            </text>
          </g>
        ))}
        <line x1={M.left} x2={W - M.right} y1={y1(0)} y2={y1(0)} className="stroke-lab-border" strokeWidth="1" />
        {axis(y1(0))}
        {arms
          .filter((a) => a.skipped)
          .map((a) => (
            <g key={a.size}>
              <rect x={x(Math.log2(a.size)) - 10} y={M.top} width={20} height={H1 - M.bottom - M.top} className="fill-lab-muted" opacity="0.08" />
              <text x={x(Math.log2(a.size))} y={M.top + 9} textAnchor="middle" className="fill-lab-muted font-mono" style={{ fontSize: 9 }}>
                skipped
              </text>
            </g>
          ))}
        <Segs p={rate} className="stroke-lab-line" />
        {rate.map((p) => (
          <circle key={p.size} cx={p.x} cy={p.y} r="3.5" className="bench-settle fill-lab-line" />
        ))}
        {cursorX !== null && <line x1={cursorX} x2={cursorX} y1={M.top} y2={H1 - M.bottom} className="stroke-lab-text-dim" strokeWidth="1" opacity="0.6" />}
      </svg>

      <svg viewBox={`0 0 ${W} ${H2}`} width="100%" height={H2} className="mt-1 block overflow-visible" role="img" aria-label="Time to first token against context size with a linear reference" onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
        {ticks(ttftMax, 3).map((t) => (
          <g key={t}>
            <line x1={M.left} x2={W - M.right} y1={y2(t)} y2={y2(t)} className="stroke-lab-border-subtle" strokeWidth="1" />
            <text x={M.left - 6} y={y2(t)} textAnchor="end" dominantBaseline="middle" className="fill-lab-muted font-mono" style={{ fontSize: 10 }}>
              {fmtT(t)}
            </text>
          </g>
        ))}
        <line x1={M.left} x2={W - M.right} y1={y2(0)} y2={y2(0)} className="stroke-lab-border" strokeWidth="1" />
        {axis(y2(0))}
        {linRef && keys.length > 1 && (
          <path
            d={keys.map((k, i) => `${i ? "L" : "M"}${x(Math.log2(k)).toFixed(1)} ${y2(Math.min(ttftMax, linRef(k))).toFixed(1)}`).join(" ")}
            fill="none"
            className="stroke-lab-muted"
            strokeDasharray="3 4"
            strokeWidth="1"
          />
        )}
        <Segs p={ttft} className="stroke-lab-text-dim" />
        {ttft.map((p) => (
          <path key={p.size} d={`M${p.x} ${p.y - 3.5} L${p.x + 3.5} ${p.y} L${p.x} ${p.y + 3.5} L${p.x - 3.5} ${p.y} Z`} className="bench-settle fill-lab-text" />
        ))}
        {cursorX !== null && <line x1={cursorX} x2={cursorX} y1={M.top} y2={H2 - M.bottom} className="stroke-lab-text-dim" strokeWidth="1" opacity="0.6" />}
      </svg>
    </div>
  );
}

/**
 * Run phase: one context-fill bar per size. Time-based fill against the
 * predicted TTFT (labelled "est."), a live stopwatch, and a flash on the first
 * token. Skipped sizes appear hatched with the engine's reason the moment the
 * `level` event lands.
 */
export function PrefillFillBars({
  sizes,
  rows,
  strands,
  current,
  predicted,
  now,
  startedAt,
  className,
}: {
  sizes: number[];
  rows: LevelRow[];
  strands: StrandView[];
  current: number;
  /** predicted TTFT (ms) for the current size, or null before any measurement */
  predicted: number | null;
  now: number;
  /** wall-clock ms when the current strand started (client-observed) */
  startedAt: number | null;
  className?: string;
}) {
  return (
    <ol className={cn("space-y-1.5", className)} aria-label="Context sizes">
      {sizes.map((size, i) => {
        const row = rows.find((r) => r.index === i);
        const strand = strands.find((s) => s.level === i);
        const isCur = i === current && !row;
        const skipped = row?.skipped;
        const failed = row && !skipped && row.ok === 0;
        const done = row && !skipped && row.ok > 0;
        const elapsed = isCur && startedAt !== null ? now - startedAt : 0;
        const firstToken = strand?.state === "decode" || strand?.state === "done";
        let frac = 0;
        if (done) frac = 1;
        else if (isCur) {
          if (firstToken) frac = 1;
          else if (predicted && predicted > 0) frac = Math.min(0.96, elapsed / predicted);
          else frac = Math.min(0.5, elapsed / 30_000);
        }
        return (
          <li
            key={size}
            className={cn("grid grid-cols-[3.25rem_minmax(0,1fr)_minmax(9rem,auto)] items-center gap-3 px-1 py-1", skipped && "bench-hatched opacity-75")}
            title={skipped ? `${fmtSize(size)} — skipped: ${skipped}` : undefined}
            aria-current={isCur ? "step" : undefined}
          >
            <span className={cn("lab-num font-[family-name:var(--font-display)] text-[13px] font-semibold", isCur ? "text-lab-text" : "text-lab-text-dim")}>
              {fmtSize(size)}
            </span>
            <div className="relative h-2.5 overflow-hidden bg-lab-hover">
              {!skipped && (
                <span
                  aria-hidden
                  className={cn("bench-lane-fill absolute inset-y-0 left-0 w-full", failed ? "bg-lab-danger" : done ? "bg-lab-line-2" : "bg-lab-line")}
                  style={{ transform: `scaleX(${frac})`, opacity: failed ? 0.6 : 1 }}
                />
              )}
              {isCur && firstToken && <span aria-hidden className="bench-flash absolute inset-0 bg-lab-text" />}
            </div>
            <span className="lab-num text-right font-mono text-[11px] text-lab-text-dim">
              {skipped ? (
                <span className="text-lab-muted">skipped: {skipped}</span>
              ) : failed ? (
                <span className="text-lab-danger">{row?.errors[0] ?? "failed"}</span>
              ) : done ? (
                <>
                  {fmtMs(row?.ttft_p50_ms)} · <span className="text-lab-text">{fmtTokS(row?.prefill_tok_s)}</span>
                  <span className="text-lab-muted"> tok/s</span>
                </>
              ) : isCur ? (
                <>
                  <span className="text-lab-text">{(elapsed / 1000).toFixed(1)} s</span>
                  {predicted ? <span className="text-lab-muted"> · est. {fmtMs(predicted)}</span> : <span className="text-lab-muted"> · est. …</span>}
                </>
              ) : (
                <span className="text-lab-muted">queued</span>
              )}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
