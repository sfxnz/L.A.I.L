"use client";

import { STALL_MS, stallIndices } from "@/lib/streams/stalls";
import { cn } from "@/lib/utils";

/**
 * Inter-token latency, last ≤100 gaps, with stall markers (≥ 2 s) as warn
 * ticks. Y is clamped to 4× the median so one stall does not flatten the rest
 * of the trace; the stall itself is the marker.
 */
export function ItlSparkline({
  itl,
  width = 120,
  height = 22,
  className,
  label,
}: {
  itl: readonly number[] | undefined;
  width?: number;
  height?: number;
  className?: string;
  label?: string;
}) {
  const pts = (itl ?? []).slice(-100);
  const n = pts.length;
  const sorted = [...pts].sort((a, b) => a - b);
  const median = n ? sorted[n >> 1] : 0;
  const cap = Math.max(median * 4, 40);
  const x = (i: number) => (n > 1 ? (i / (n - 1)) * width : width / 2);
  const y = (v: number) => height - 1 - (Math.min(v, cap) / cap) * (height - 3);
  const stalls = stallIndices(pts);
  const line = pts.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      preserveAspectRatio="none"
      className={cn("block overflow-visible text-lab-line-2", className)}
      role="img"
      aria-label={label ?? (n ? `Inter-token latency, ${n} gaps, ${stalls.length} stalls` : "No inter-token latency yet")}
    >
      <line x1="0" x2={width} y1={height - 1} y2={height - 1} stroke="var(--animus-hairline)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
      {n > 1 && <path d={line} fill="none" stroke="currentColor" strokeWidth="1" vectorEffect="non-scaling-stroke" opacity="0.9" />}
      {n === 1 && <circle cx={x(0)} cy={y(pts[0])} r="1.5" fill="currentColor" />}
      {stalls.map((k) => (
        <g key={k} className="text-lab-warn">
          <line x1={x(k)} x2={x(k)} y1="0" y2={height} stroke="currentColor" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
          <title>{`stall ${(pts[k] / 1000).toFixed(1)} s (≥ ${STALL_MS / 1000} s)`}</title>
        </g>
      ))}
    </svg>
  );
}
