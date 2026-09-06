/** Scale helpers for the hand-rolled SVG charts. */

/** Zero-based axis ceiling with ~12 % headroom on a 1/2/2.5/5 grid. */
export function niceMax(maxValue: number): number {
  const v = Math.max(1e-9, maxValue) * 1.12;
  const mag = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) {
    if (v <= m * mag) return m * mag;
  }
  return 10 * mag;
}

/** Evenly spaced ticks from 0 to max (max itself included). */
export function ticks(max: number, count = 4): number[] {
  if (max <= 0) return [0];
  const step = max / count;
  return Array.from({ length: count + 1 }, (_, i) => Math.round(i * step * 1000) / 1000);
}

export type Scale = (v: number) => number;

export function linear(d0: number, d1: number, r0: number, r1: number): Scale {
  const span = d1 - d0 || 1;
  return (v) => r0 + ((v - d0) / span) * (r1 - r0);
}

/** log2 domain over the given keys, padded when a single key would collapse it. */
export function log2Domain(keys: number[]): [number, number] {
  const xs = keys.filter((k) => k > 0).map((k) => Math.log2(k));
  if (!xs.length) return [0, 1];
  let lo = Math.min(...xs);
  let hi = Math.max(...xs);
  if (hi - lo < 1e-9) {
    lo -= 0.5;
    hi += 0.5;
  }
  return [lo, hi];
}
