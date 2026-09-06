"use client";

import { useEffect, useRef } from "react";
import { fmtTokS } from "@/lib/bench/format";
import { cn } from "@/lib/utils";

/*
  Aggregate gauge — arc, needle and digits all track ONE smoothed value (the
  live 1 s-window aggregate from `agg`), eased with a ~1 s time constant in a
  single rAF loop that writes the DOM directly. A peak tick and a gold settled
  marker (the last completed level's aggregate) give the needle its reference.
*/

const CX = 100;
const CY = 100;
const R = 80;
const SWEEP = 100; // ± degrees from straight up
const TAU_S = 0.6; // smoothing time constant, s (on top of the server's 1 s window)
const NICE = [20, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000];

function niceCeil(v: number): number {
  for (const n of NICE) if (v <= n) return n;
  return Math.ceil(v / 10000) * 10000;
}

function pt(deg: number, r = R): [number, number] {
  const a = (deg * Math.PI) / 180;
  return [CX + r * Math.sin(a), CY - r * Math.cos(a)];
}

function arcPath(fromDeg: number, toDeg: number, r = R): string {
  if (toDeg <= fromDeg) return "";
  const [x0, y0] = pt(fromDeg, r);
  const [x1, y1] = pt(toDeg, r);
  const large = toDeg - fromDeg > 180 ? 1 : 0;
  return `M${x0.toFixed(2)} ${y0.toFixed(2)} A${r} ${r} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}

export function Gauge({
  value,
  peak,
  settled,
  label = "aggregate tok/s",
  className,
}: {
  /** live aggregate tok/s (server 1 s window) */
  value: number;
  peak: number;
  /** last completed level's aggregate — the gold reference */
  settled: number | null;
  label?: string;
  className?: string;
}) {
  const target = useRef(value);
  target.current = value;
  // The scale only ever grows during a run — a shrinking arc would read as a stall.
  const hwm = useRef(20);
  const max = Math.max(hwm.current, niceCeil(Math.max(20, value * 1.15, peak * 1.15, (settled ?? 0) * 1.15)));
  hwm.current = max;
  const maxRef = useRef(max);
  maxRef.current = max;

  const arcRef = useRef<SVGPathElement>(null);
  const needleRef = useRef<SVGGElement>(null);
  const digitsRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    let v = target.current;
    const write = (val: number) => {
      const frac = Math.max(0, Math.min(1, val / maxRef.current));
      const deg = -SWEEP + frac * 2 * SWEEP;
      if (arcRef.current) arcRef.current.setAttribute("d", arcPath(-SWEEP, deg));
      if (needleRef.current) needleRef.current.style.transform = `rotate(${deg.toFixed(2)}deg)`;
      if (digitsRef.current) digitsRef.current.textContent = fmtTokS(val) || "0";
    };
    const step = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      v += (target.current - v) * (1 - Math.exp(-dt / TAU_S));
      write(v);
      raf = requestAnimationFrame(step);
    };
    write(v);
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, []);

  const degOf = (val: number) => -SWEEP + Math.max(0, Math.min(1, val / max)) * 2 * SWEEP;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const [nx, ny] = pt(0, R - 8);
  const [sx, sy] = settled !== null ? pt(degOf(settled), R + 9) : [0, 0];
  const [px0, py0] = pt(degOf(peak), R - 5);
  const [px1, py1] = pt(degOf(peak), R + 6);

  return (
    <div className={cn("relative", className)} role="img" aria-label={`${label}: ${fmtTokS(value)} of ${max}`}>
      <svg viewBox="0 0 200 122" className="block h-auto w-full" aria-hidden>
        <path d={arcPath(-SWEEP, SWEEP)} fill="none" className="stroke-lab-border" strokeWidth="6" strokeLinecap="butt" />
        <path ref={arcRef} d="" fill="none" className="stroke-lab-line-2" strokeWidth="6" strokeLinecap="butt" />
        {ticks.map((t) => {
          const d = -SWEEP + t * 2 * SWEEP;
          const [ax, ay] = pt(d, R - 10);
          const [bx, by] = pt(d, R - 15);
          const [lx, ly] = pt(d, R - 25);
          return (
            <g key={t}>
              <line x1={ax} y1={ay} x2={bx} y2={by} className="stroke-lab-muted" strokeWidth="1" />
              <text
                x={lx}
                y={ly}
                textAnchor="middle"
                dominantBaseline="middle"
                className="fill-lab-muted font-mono"
                style={{ fontSize: 7 }}
              >
                {fmtTokS(t * max)}
              </text>
            </g>
          );
        })}
        {peak > 0 && <line x1={px0} y1={py0} x2={px1} y2={py1} className="stroke-lab-text" strokeWidth="1.5" />}
        {settled !== null && (
          <path
            d={`M${sx} ${sy - 4} L${sx + 4} ${sy} L${sx} ${sy + 4} L${sx - 4} ${sy} Z`}
            className="fill-lab-target bench-settle"
          />
        )}
        <g ref={needleRef} className="bench-needle">
          <line x1={CX} y1={CY} x2={nx} y2={ny} className="stroke-lab-accent-bright" strokeWidth="1.75" strokeLinecap="square" />
          <path d={`M${CX - 4} ${CY} L${CX} ${CY - 4} L${CX + 4} ${CY} L${CX} ${CY + 4} Z`} className="fill-lab-accent-bright" />
        </g>
      </svg>
      <div className="pointer-events-none absolute inset-x-0 bottom-0 flex flex-col items-center leading-none">
        <span
          ref={digitsRef}
          className="lab-num font-[family-name:var(--font-display)] text-[30px] font-bold text-lab-text"
        >
          {fmtTokS(value) || "0"}
        </span>
        <span className="animus-eyebrow mt-1 text-[9px]">{label}</span>
      </div>
    </div>
  );
}
