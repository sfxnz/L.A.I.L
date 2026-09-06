"use client";

import { useMemo } from "react";
import { Eyebrow } from "@/components/ui";
import type { StrandView } from "@/lib/use-stream-run";
import { fmtDuration } from "@/lib/streams/format";
import { sequenceLayout, type SeqClock } from "@/lib/streams/sequence";
import { strandColor } from "@/lib/streams/strand-color";
import { cn } from "@/lib/utils";

const W = 600;
const LABEL_W = 22;

/**
 * Sequence view — the prefill storm as a picture. One row per strand:
 * waiting (hairline) → TTFT (line-2) → decode (line) → end tick; stalls
 * hatched in warn. Only what this client observed (lib/streams/sequence.ts).
 */
export function SequenceView({
  strands,
  clock,
  helloAt,
  nowAt,
  hovered,
  focused,
  onHover,
  className,
}: {
  strands: readonly StrandView[];
  clock: SeqClock;
  helloAt: number | null;
  nowAt: number;
  hovered: number | null;
  focused: number | null;
  onHover: (i: number | null) => void;
  className?: string;
}) {
  const n = strands.length;
  const rowH = n > 16 ? 8 : n > 8 ? 11 : 14;
  const H = Math.max(120, n * rowH + 26);
  const PAD_T = 10;
  const PAD_B = 14;

  const layout = useMemo(() => sequenceLayout(strands, clock, helloAt, nowAt), [strands, clock, helloAt, nowAt]);
  const span = Math.max(1, layout.t_max - layout.t_min);
  const x = (t: number) => LABEL_W + ((t - layout.t_min) / span) * (W - LABEL_W);
  const ticks: number[] = [];
  const step = span > 60_000 ? 15_000 : span > 20_000 ? 5_000 : span > 5000 ? 1000 : 500;
  for (let t = layout.t_min; t <= layout.t_max; t += step) ticks.push(t);
  const active = hovered ?? focused;
  const noTimeline = layout.rows.every((r) => r.segments.length === 0);

  return (
    <div className={cn("relative", className)} style={{ height: H }}>
      {noTimeline && (
        <Eyebrow className="pointer-events-none absolute inset-x-0 top-1/2 -translate-y-1/2 text-center text-lab-muted">
          {n ? "attached after the run · no observed timeline" : "awaiting strands"}
        </Eyebrow>
      )}
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="block h-full w-full" role="img" aria-label="Sequence view: waiting, TTFT, decode per strand over run time">
        <defs>
          <pattern id="strand-stall-hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="4" stroke="var(--color-lab-warn)" strokeWidth="1.5" />
          </pattern>
        </defs>
        {ticks.map((t) => (
          <line key={t} x1={x(t)} x2={x(t)} y1={PAD_T} y2={H - PAD_B} stroke="var(--animus-hairline)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
        ))}
        {layout.rows.map((row, k) => {
          const y0 = PAD_T + k * ((H - PAD_T - PAD_B) / Math.max(1, n));
          const rh = Math.max(3, (H - PAD_T - PAD_B) / Math.max(1, n) - 2);
          const dim = active !== null && active !== row.i;
          return (
            <g
              key={row.i}
              opacity={dim ? 0.3 : 1}
              style={{ transition: "opacity var(--dur-tap) linear" }}
              onMouseEnter={() => onHover(row.i)}
              onMouseLeave={() => onHover(null)}
            >
              <rect x="0" y={y0 - 1} width={W} height={rh + 2} fill="transparent" />
              <text x="0" y={y0 + rh * 0.8} fontSize={Math.min(9, rh + 1)} fill="var(--color-lab-muted)" fontFamily="var(--font-mono)">
                {String(row.i + 1).padStart(2, "0")}
              </text>
              {row.segments.map((seg, j) => {
                const sx = x(seg.x0);
                const sw = Math.max(0.5, x(seg.x1) - sx);
                if (seg.kind === "wait") {
                  return <line key={j} x1={sx} x2={sx + sw} y1={y0 + rh / 2} y2={y0 + rh / 2} stroke="var(--animus-hairline)" strokeWidth="1" vectorEffect="non-scaling-stroke" />;
                }
                if (seg.kind === "stall") {
                  return <rect key={j} x={sx} y={y0} width={sw} height={rh} fill="url(#strand-stall-hatch)" />;
                }
                return (
                  <rect
                    key={j}
                    x={sx}
                    y={y0}
                    width={sw}
                    height={rh}
                    fill={seg.kind === "ttft" ? "var(--color-lab-line-2)" : active === row.i ? strandColor(row.i) : "var(--color-lab-line)"}
                    fillOpacity={seg.kind === "ttft" ? 0.9 : 0.7}
                  />
                );
              })}
              {row.end !== null && (
                <line
                  x1={x(row.end)}
                  x2={x(row.end)}
                  y1={y0 - 1}
                  y2={y0 + rh + 1}
                  stroke={row.state === "error" ? "var(--color-lab-danger)" : row.state === "cancelled" ? "var(--color-lab-warn)" : "var(--color-lab-text)"}
                  strokeWidth="1.5"
                  vectorEffect="non-scaling-stroke"
                />
              )}
            </g>
          );
        })}
        <line x1={LABEL_W} x2={W} y1={H - PAD_B} y2={H - PAD_B} stroke="var(--animus-hairline)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="pointer-events-none absolute inset-x-0 bottom-0 flex justify-between px-1">
        <span className="lab-num font-mono text-[9px] text-lab-muted">{fmtDuration(layout.t_min)}</span>
        <span className="lab-num font-mono text-[9px] text-lab-muted">{fmtDuration(layout.t_max)}</span>
      </div>
      <div className="pointer-events-none absolute top-0 right-1 flex items-center gap-2">
        <Eyebrow className="flex items-center gap-1"><span aria-hidden className="inline-block h-2 w-3 bg-lab-line-2" /> ttft</Eyebrow>
        <Eyebrow className="flex items-center gap-1"><span aria-hidden className="inline-block h-2 w-3 bg-lab-line opacity-70" /> decode</Eyebrow>
        <Eyebrow className="flex items-center gap-1"><span aria-hidden className="inline-block h-2 w-3 strand-hatch" /> stall</Eyebrow>
      </div>
    </div>
  );
}
