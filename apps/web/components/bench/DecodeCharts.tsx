"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { fmtMs, fmtTokS } from "@/lib/bench/format";
import type { DecodeArm } from "@/lib/bench/result";
import { linear, log2Domain, niceMax, ticks } from "@/lib/bench/scale";
import { Eyebrow } from "@/components/ui";
import { cn } from "@/lib/utils";

/*
  The curve that draws itself. x = log2(concurrency) so ×1…×32 sit evenly and a
  ghost with a different level set still lines up; y is zero-based tok/s shared
  by both series (aggregate rises in `line`, per-stream median decays in
  `line-2`). Each new level adds two points that settle in 250 ms and a segment
  that extends over 700 ms with --ease-chime; the y ceiling eases when a point
  needs more room. Beneath: TTFT p50 with a p95–p99 whisker on the same x.
  Hand-rolled SVG — no chart library owns these tokens or this timing.
*/

const M = { top: 12, right: 14, bottom: 22, left: 46 };

function reducedMotion() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Eases a number toward its target over `ms` (cubic out); snaps under reduced motion. */
export function useEased(target: number, ms = 600): number {
  const [v, setV] = useState(target);
  const cur = useRef(target);
  useEffect(() => {
    if (reducedMotion() || Math.abs(cur.current - target) < 1e-9) {
      cur.current = target;
      setV(target);
      return;
    }
    const v0 = cur.current;
    const t0 = performance.now();
    let raf = 0;
    const step = (now: number) => {
      const t = Math.min(1, (now - t0) / ms);
      const e = 1 - Math.pow(1 - t, 3);
      cur.current = v0 + (target - v0) * e;
      setV(cur.current);
      if (t < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

export function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T | null>(null);
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) setW(Math.round(e.contentRect.width));
    });
    ro.observe(el);
    setW(Math.round(el.getBoundingClientRect().width));
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

export type GhostArms = { arms: DecodeArm[]; label: string } | null;

type Pt = { c: number; x: number; y: number; v: number };

function series(arms: DecodeArm[], pick: (a: DecodeArm) => number | null, x: (c: number) => number, y: (v: number) => number): Pt[] {
  return arms
    .filter((a) => pick(a) !== null && a.ok > 0)
    .slice()
    .sort((a, b) => a.concurrency - b.concurrency)
    .map((a) => ({ c: a.concurrency, v: pick(a) as number, x: x(Math.log2(a.concurrency)), y: y(pick(a) as number) }));
}

function polyline(pts: Pt[]): string {
  return pts.map((p, i) => `${i ? "L" : "M"}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
}

/** Segments keyed by their end level so an already-drawn segment never re-animates. */
function Segments({ pts, className }: { pts: Pt[]; className: string }) {
  return (
    <>
      {pts.slice(1).map((p, i) => {
        const a = pts[i];
        return (
          <path
            key={p.c}
            d={`M${a.x.toFixed(1)} ${a.y.toFixed(1)} L${p.x.toFixed(1)} ${p.y.toFixed(1)}`}
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

function Diamond({ x, y, r = 3.5, className }: { x: number; y: number; r?: number; className: string }) {
  return <path d={`M${x} ${y - r} L${x + r} ${y} L${x} ${y + r} L${x - r} ${y} Z`} className={cn("bench-settle", className)} />;
}

function XAxis({ keys, x, y0, kind }: { keys: number[]; x: (lx: number) => number; y0: number; kind: "decode" }) {
  void kind;
  return (
    <>
      {keys.map((k) => (
        <text key={k} x={x(Math.log2(k))} y={y0 + 15} textAnchor="middle" className="fill-lab-muted font-mono" style={{ fontSize: 10 }}>
          ×{k}
        </text>
      ))}
    </>
  );
}

export function DecodeCharts({
  arms,
  levels,
  ghost,
  knee,
  floor,
  sloMs,
  showTtft = true,
  className,
}: {
  arms: DecodeArm[];
  /** configured levels — fixes the x domain before any point exists */
  levels: number[];
  ghost?: GhostArms;
  /** knee concurrency to shade from, gold */
  knee?: number | null;
  floor?: number;
  sloMs?: number;
  showTtft?: boolean;
  className?: string;
}) {
  const [wrapRef, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const H = 220;
  const HT = 110;
  const W = Math.max(280, width || 600);

  const keys = [...new Set([...levels, ...arms.map((a) => a.concurrency), ...(ghost?.arms.map((a) => a.concurrency) ?? [])])].sort((a, b) => a - b);
  const [d0, d1] = log2Domain(keys);
  const x = linear(d0, d1, M.left, W - M.right);

  const values = [
    ...arms.flatMap((a) => (a.ok > 0 ? [a.aggregate ?? 0, a.perStream ?? 0] : [])),
    ...(ghost?.arms.flatMap((a) => [a.aggregate ?? 0, a.perStream ?? 0]) ?? []),
    floor ?? 0,
  ];
  const yTarget = niceMax(Math.max(10, ...values));
  const yMax = useEased(yTarget);
  const y = linear(0, yMax, H - M.bottom, M.top);

  const agg = series(arms, (a) => a.aggregate, x, y);
  const per = series(arms, (a) => a.perStream, x, y);
  const gAgg = ghost ? series(ghost.arms, (a) => a.aggregate, x, y) : [];
  const gPer = ghost ? series(ghost.arms, (a) => a.perStream, x, y) : [];

  const ttftVals = arms.flatMap((a) => (a.ok > 0 ? [a.ttftP99 ?? a.ttftP95 ?? a.ttftP50 ?? 0] : []));
  const tTarget = niceMax(Math.max(100, ...ttftVals, sloMs ?? 0));
  const tMax = useEased(tTarget);
  const yt = linear(0, tMax, HT - M.bottom, M.top);
  const ttft = series(arms, (a) => a.ttftP50, x, yt);

  const hovered = hover !== null ? arms.find((a) => a.concurrency === hover) ?? null : null;
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!arms.length) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let best: number | null = null;
    let bd = Infinity;
    for (const a of arms) {
      const d = Math.abs(x(Math.log2(a.concurrency)) - px);
      if (d < bd) {
        bd = d;
        best = a.concurrency;
      }
    }
    setHover(bd < 40 ? best : null);
  };

  const cursorX = hover !== null ? x(Math.log2(hover)) : null;

  return (
    <div ref={wrapRef} className={cn("relative", className)}>
      <div className="mb-1 flex flex-wrap items-center gap-x-4 gap-y-1">
        <Eyebrow className="flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-[2px] w-4 bg-lab-line" /> aggregate tok/s
        </Eyebrow>
        <Eyebrow className="flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-[2px] w-4 bg-lab-line-2" /> per-stream median
        </Eyebrow>
        {ghost && (
          <Eyebrow className="flex items-center gap-1.5" title={ghost.label}>
            <span aria-hidden className="inline-block h-[2px] w-4 border-t border-dashed border-lab-muted" /> previous · {ghost.label}
          </Eyebrow>
        )}
        {knee != null && (
          <Eyebrow className="text-lab-target!">knee ×{knee}</Eyebrow>
        )}
        {hovered && (
          <span className="lab-num ml-auto font-mono text-[11px] text-lab-text">
            ×{hovered.concurrency} · <span className="text-lab-line">{fmtTokS(hovered.aggregate)}</span> ·{" "}
            <span className="text-lab-line-2">{fmtTokS(hovered.perStream)}</span> tok/s · TTFT p50 {fmtMs(hovered.ttftP50) || "—"}
            {hovered.ok < hovered.requests ? <span className="text-lab-danger"> · {hovered.ok}/{hovered.requests} ok</span> : null}
          </span>
        )}
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        height={H}
        className="block overflow-visible"
        role="img"
        aria-label="Aggregate and per-stream tok/s against concurrency"
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
      >
        {ticks(yMax, 4).map((t) => (
          <g key={t}>
            <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} className="stroke-lab-border-subtle" strokeWidth="1" />
            <text x={M.left - 6} y={y(t)} textAnchor="end" dominantBaseline="middle" className="fill-lab-muted font-mono" style={{ fontSize: 10 }}>
              {fmtTokS(t)}
            </text>
          </g>
        ))}
        <line x1={M.left} x2={W - M.right} y1={y(0)} y2={y(0)} className="stroke-lab-border" strokeWidth="1" />
        <XAxis keys={keys} x={x} y0={y(0)} kind="decode" />

        {knee != null && (
          <g>
            <rect x={x(Math.log2(knee))} y={M.top} width={Math.max(0, W - M.right - x(Math.log2(knee)))} height={H - M.bottom - M.top} className="fill-lab-target bench-fade" opacity="0.08" />
            <line x1={x(Math.log2(knee))} x2={x(Math.log2(knee))} y1={M.top} y2={H - M.bottom} className="stroke-lab-target" strokeDasharray="3 3" strokeWidth="1" />
          </g>
        )}
        {floor != null && floor > 0 && (
          <g>
            <line x1={M.left} x2={W - M.right} y1={y(floor)} y2={y(floor)} className="stroke-lab-muted" strokeDasharray="2 4" strokeWidth="1" />
            <text x={W - M.right} y={y(floor) - 3} textAnchor="end" className="fill-lab-muted font-mono" style={{ fontSize: 9 }}>
              floor {floor}
            </text>
          </g>
        )}

        {gAgg.length > 1 && <path d={polyline(gAgg)} fill="none" className="bench-ghost stroke-lab-line" strokeWidth="1.5" />}
        {gPer.length > 1 && <path d={polyline(gPer)} fill="none" className="bench-ghost stroke-lab-line-2" strokeWidth="1.5" />}

        <Segments pts={agg} className="stroke-lab-line" />
        <Segments pts={per} className="stroke-lab-line-2" />
        {agg.map((p) => (
          <circle key={p.c} cx={p.x} cy={p.y} r="3.5" className="bench-settle fill-lab-line" />
        ))}
        {per.map((p) => (
          <Diamond key={p.c} x={p.x} y={p.y} className="fill-lab-line-2" />
        ))}
        {arms
          .filter((a) => a.ok < a.requests)
          .map((a) => (
            <text key={a.concurrency} x={x(Math.log2(a.concurrency))} y={M.top + 8} textAnchor="middle" className="fill-lab-danger font-mono" style={{ fontSize: 9 }}>
              {a.ok}/{a.requests}
            </text>
          ))}
        {cursorX !== null && <line x1={cursorX} x2={cursorX} y1={M.top} y2={H - M.bottom} className="stroke-lab-text-dim" strokeWidth="1" opacity="0.6" />}
      </svg>

      {showTtft && (
        <svg
          viewBox={`0 0 ${W} ${HT}`}
          width="100%"
          height={HT}
          className="mt-1 block overflow-visible"
          role="img"
          aria-label="Time to first token p50 with p95–p99 whisker"
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
        >
          {ticks(tMax, 2).map((t) => (
            <g key={t}>
              <line x1={M.left} x2={W - M.right} y1={yt(t)} y2={yt(t)} className="stroke-lab-border-subtle" strokeWidth="1" />
              <text x={M.left - 6} y={yt(t)} textAnchor="end" dominantBaseline="middle" className="fill-lab-muted font-mono" style={{ fontSize: 10 }}>
                {t >= 1000 ? `${(t / 1000).toFixed(t % 1000 ? 1 : 0)}s` : `${Math.round(t)}`}
              </text>
            </g>
          ))}
          <text x={M.left} y={M.top - 3} className="fill-lab-muted font-[family-name:var(--font-display)] uppercase" style={{ fontSize: 9, letterSpacing: "0.14em" }}>
            TTFT p50 · p95–p99 (ms)
          </text>
          {sloMs != null && sloMs > 0 && (
            <g>
              <line x1={M.left} x2={W - M.right} y1={yt(sloMs)} y2={yt(sloMs)} className="stroke-lab-muted" strokeDasharray="2 4" strokeWidth="1" />
              <text x={W - M.right} y={yt(sloMs) - 3} textAnchor="end" className="fill-lab-muted font-mono" style={{ fontSize: 9 }}>
                SLO {fmtMs(sloMs)}
              </text>
            </g>
          )}
          {arms
            .filter((a) => a.ok > 0 && a.ttftP95 !== null)
            .map((a) => {
              const px = x(Math.log2(a.concurrency));
              const y95 = yt(a.ttftP95 as number);
              const y99 = yt(a.ttftP99 ?? (a.ttftP95 as number));
              return (
                <g key={a.concurrency} className="bench-settle">
                  <line x1={px} x2={px} y1={y95} y2={y99} className="stroke-lab-muted" strokeWidth="1.25" />
                  <line x1={px - 3} x2={px + 3} y1={y99} y2={y99} className="stroke-lab-muted" strokeWidth="1.25" />
                  <line x1={px - 3} x2={px + 3} y1={y95} y2={y95} className="stroke-lab-muted" strokeWidth="1" />
                </g>
              );
            })}
          <Segments pts={ttft} className="stroke-lab-text-dim" />
          {ttft.map((p) => (
            <Diamond key={p.c} x={p.x} y={p.y} r={3} className="fill-lab-text" />
          ))}
          {cursorX !== null && <line x1={cursorX} x2={cursorX} y1={M.top} y2={HT - M.bottom} className="stroke-lab-text-dim" strokeWidth="1" opacity="0.6" />}
          {hovered && hovered.ttftP50 !== null && (
            <text x={x(Math.log2(hovered.concurrency))} y={HT - M.bottom + 15} textAnchor="middle" className="fill-lab-text font-mono" style={{ fontSize: 10 }}>
              {fmtMs(hovered.ttftP50)}
              {hovered.ttftP99 !== null ? ` · p99 ${fmtMs(hovered.ttftP99)}` : ""}
            </text>
          )}
        </svg>
      )}
    </div>
  );
}

/** 240×80 mini curve for the Status card: aggregate only, optional ghost, no axes. */
export function MiniCurve({
  arms,
  ghost,
  width = 240,
  height = 80,
  className,
}: {
  arms: DecodeArm[];
  ghost?: DecodeArm[] | null;
  width?: number;
  height?: number;
  className?: string;
}) {
  const keys = [...new Set([...arms.map((a) => a.concurrency), ...(ghost?.map((a) => a.concurrency) ?? [])])];
  const [d0, d1] = log2Domain(keys);
  const x = linear(d0, d1, 6, width - 6);
  const yMax = niceMax(Math.max(10, ...arms.map((a) => a.aggregate ?? 0), ...(ghost?.map((a) => a.aggregate ?? 0) ?? [])));
  const y = linear(0, yMax, height - 6, 6);
  const agg = series(arms, (a) => a.aggregate, x, y);
  const g = ghost ? series(ghost, (a) => a.aggregate, x, y) : [];
  const peak = agg.reduce<Pt | null>((b, p) => (b === null || p.v > b.v ? p : b), null);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} className={cn("block max-w-full", className)} role="img" aria-label="Aggregate tok/s by concurrency, latest run">
      <line x1={6} x2={width - 6} y1={height - 6} y2={height - 6} className="stroke-lab-border" strokeWidth="1" />
      {g.length > 1 && <path d={polyline(g)} fill="none" className="bench-ghost stroke-lab-line" strokeWidth="1.25" />}
      {agg.length > 1 && <path d={`${polyline(agg)} L${agg[agg.length - 1].x} ${height - 6} L${agg[0].x} ${height - 6} Z`} className="fill-lab-line" opacity="0.1" />}
      {agg.length > 1 && <path d={polyline(agg)} fill="none" className="stroke-lab-line" strokeWidth="1.5" />}
      {agg.map((p) => (
        <circle key={p.c} cx={p.x} cy={p.y} r="2.5" className="fill-lab-line" />
      ))}
      {peak && <Diamond x={peak.x} y={peak.y} r={4} className="fill-lab-target" />}
    </svg>
  );
}
