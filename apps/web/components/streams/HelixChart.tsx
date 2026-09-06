"use client";

import { useMemo, useRef } from "react";
import { Eyebrow } from "@/components/ui";
import type { AggPoint } from "@/lib/use-stream-run";
import { fmtRate } from "@/lib/streams/format";
import { strandColor, strandOpacity } from "@/lib/streams/strand-color";
import { cn } from "@/lib/utils";

const W = 600;

/**
 * The Helix — per-strand tok/s as stacked bands over the last 60 s. The top
 * edge is the engine's aggregate (`agg.tok_s`, silver line); the bands are the
 * client-observed split of that number (see lib/streams/live-rate.ts). Peak is
 * the one gold tick. Hover a band → its card lights, and vice versa.
 */
export function HelixChart({
  agg,
  n,
  windowMs,
  height = 120,
  hovered,
  focused,
  onHover,
  live,
  className,
}: {
  agg: readonly AggPoint[];
  n: number;
  windowMs: number;
  height?: number;
  hovered: number | null;
  focused: number | null;
  onHover: (i: number | null) => void;
  live: boolean;
  className?: string;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const H = height;
  const PAD_T = 14;
  const PAD_B = 12;

  const geo = useMemo(() => {
    if (!agg.length) return null;
    const tEnd = agg[agg.length - 1].t_ms;
    const tStart = Math.max(0, tEnd - windowMs);
    const span = Math.max(1, tEnd - tStart);
    const peak = agg.reduce((m, p) => Math.max(m, p.peak_tok_s), 0);
    const yMax = Math.max(1, agg.reduce((m, p) => Math.max(m, p.tok_s), 0), peak) * 1.06;
    const x = (t: number) => ((t - tStart) / span) * W;
    const y = (v: number) => PAD_T + (1 - Math.min(v, yMax) / yMax) * (H - PAD_T - PAD_B);
    const pts = agg.filter((p) => p.t_ms >= tStart);
    // cumulative stacks per point
    const lowers: number[][] = [];
    const uppers: number[][] = [];
    for (const p of pts) {
      const shares = p.strand_tok_s ?? [];
      const lo: number[] = [];
      const up: number[] = [];
      let acc = 0;
      for (let i = 0; i < n; i++) {
        lo.push(acc);
        acc += shares[i] ?? 0;
        up.push(acc);
      }
      lowers.push(lo);
      uppers.push(up);
    }
    const bands: string[] = [];
    for (let i = 0; i < n; i++) {
      let d = "";
      pts.forEach((p, k) => {
        d += `${k ? "L" : "M"}${x(p.t_ms).toFixed(1)} ${y(uppers[k][i]).toFixed(1)} `;
      });
      for (let k = pts.length - 1; k >= 0; k--) d += `L${x(pts[k].t_ms).toFixed(1)} ${y(lowers[k][i]).toFixed(1)} `;
      bands.push(d + "Z");
    }
    const top = pts.map((p, k) => `${k ? "L" : "M"}${x(p.t_ms).toFixed(1)} ${y(p.tok_s).toFixed(1)}`).join(" ");
    const ticks: Array<{ x: number; label: string }> = [];
    for (let back = 0; back <= windowMs; back += 15_000) {
      const t = tEnd - back;
      if (t < tStart) break;
      ticks.push({ x: x(t), label: back === 0 ? "now" : `−${back / 1000} s` });
    }
    return { pts, x, y, bands, top, peak, yMax, ticks, tStart, tEnd, lowers, uppers };
  }, [agg, n, windowMs, H]);

  const pick = (clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg || !geo) return null;
    const r = svg.getBoundingClientRect();
    const px = ((clientX - r.left) / r.width) * W;
    const py = ((clientY - r.top) / r.height) * H;
    let best = 0;
    let bestD = Infinity;
    geo.pts.forEach((p, k) => {
      const d = Math.abs(geo.x(p.t_ms) - px);
      if (d < bestD) {
        bestD = d;
        best = k;
      }
    });
    const v = (1 - (py - PAD_T) / (H - PAD_T - PAD_B)) * geo.yMax;
    for (let i = 0; i < n; i++) {
      if (v >= geo.lowers[best][i] && v < geo.uppers[best][i] && geo.uppers[best][i] > geo.lowers[best][i]) return i;
    }
    return null;
  };

  const active = hovered ?? focused;

  return (
    <div className={cn("relative", className)} style={{ height: H }}>
      {!geo ? (
        <div className="flex h-full items-end">
          <div className="animus-rule w-full" />
          <Eyebrow className="absolute top-2 left-0">{live ? "Awaiting first sample" : "Helix · last 60 s"}</Eyebrow>
        </div>
      ) : (
        <>
          <svg
            ref={svgRef}
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="none"
            className="block h-full w-full"
            role="img"
            aria-label={`Aggregate tok/s over the last ${windowMs / 1000} s, stacked per strand`}
            onMouseMove={(e) => onHover(pick(e.clientX, e.clientY))}
            onMouseLeave={() => onHover(null)}
          >
            {geo.ticks.map((t) => (
              <line key={t.x} x1={t.x} x2={t.x} y1={PAD_T} y2={H - PAD_B} stroke="var(--animus-hairline)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
            ))}
            <line x1="0" x2={W} y1={H - PAD_B} y2={H - PAD_B} stroke="var(--animus-hairline)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
            {geo.bands.map((d, i) => (
              <path
                key={i}
                d={d}
                fill={strandColor(i)}
                fillOpacity={active === null ? strandOpacity(i) * 0.75 : active === i ? 0.95 : 0.18}
                stroke={active === i ? strandColor(i) : "none"}
                strokeWidth="1"
                vectorEffect="non-scaling-stroke"
                style={{ transition: "fill-opacity var(--dur-tap) linear" }}
              />
            ))}
            <path d={geo.top} fill="none" stroke="var(--color-lab-text)" strokeWidth="1.25" vectorEffect="non-scaling-stroke" opacity="0.9" />
            {geo.peak > 0 && (
              <line
                x1="0"
                x2={W}
                y1={geo.y(geo.peak)}
                y2={geo.y(geo.peak)}
                stroke="var(--color-lab-target)"
                strokeWidth="1"
                strokeDasharray="3 4"
                vectorEffect="non-scaling-stroke"
                opacity="0.8"
              />
            )}
          </svg>
          {geo.pts.length < 2 && !live && (
            <Eyebrow className="pointer-events-none absolute inset-x-0 top-1/2 -translate-y-1/2 text-center text-lab-muted">
              attached after the run · no live history
            </Eyebrow>
          )}
          <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start justify-between px-1">
            <Eyebrow className="lab-num text-lab-text-dim">{fmtRate(geo.yMax / 1.06)} tok/s</Eyebrow>
            {geo.peak > 0 && (
              <Eyebrow className="lab-num text-lab-target">peak {fmtRate(geo.peak)}</Eyebrow>
            )}
          </div>
          <div className="pointer-events-none absolute inset-x-0 bottom-0 flex justify-between px-1">
            {geo.ticks
              .slice()
              .reverse()
              .map((t) => (
                <span key={t.label} className="lab-num font-mono text-[9px] text-lab-muted">
                  {t.label}
                </span>
              ))}
          </div>
        </>
      )}
    </div>
  );
}
