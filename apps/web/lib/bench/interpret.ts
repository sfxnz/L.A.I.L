/**
 * The one-sentence interpretation and the knee. Pure; hand-computable.
 *
 * Decode:
 *   efficiency      aggregate_N ÷ (N × aggregate_1) at the top level
 *   best interactive largest N whose per-stream median ≥ floor AND TTFT p50 ≤ SLO
 *   knee            first level whose marginal aggregate gain per added stream is
 *                   < 25 % of the ×1→×2 gain (the first two levels) by more than
 *                   the measured spread, or whose per-stream median falls below the
 *                   floor — whichever comes first
 *   saturated       the last level's marginal gain is < 25 % of the reference, beyond
 *                   the measured spread
 *   spread          half the min–max of a level's per-wave aggregates (0 for one wave)
 *   contended       a level that ran while the server held foreign requests is not
 *                   used for the knee or saturation (its numbers are not a clean
 *                   measurement)
 * Prefill:
 *   hold            1 − sustained ÷ peak prefill tok/s
 *   doubling ratios TTFT_i ÷ TTFT_{i−1}, normalised to one context doubling
 */
import { fmtMs, fmtSize, fmtTokS } from "./format";
import type { DecodeArm, PrefillArm } from "./result";

export const KNEE_FRACTION = 0.25;

export type DecodeInterpretation = {
  topLevel: number | null;
  /** levels measured while foreign requests ran on the server */
  contended: number[];
  efficiency: number | null;
  perStreamRatio: number | null;
  bestInteractive: DecodeArm | null;
  knee: number | null;
  kneeReason: "marginal" | "floor" | null;
  /** true once the last level no longer gains ≥ 25 % of the reference per added stream */
  saturated: boolean;
  /** levels above the top one worth running when not saturated */
  suggest: number[];
  sentence: string;
};

export type InterpretOpts = { floor: number; sloMs: number };

function measured(arms: DecodeArm[]): DecodeArm[] {
  return arms
    .filter((a) => a.aggregate !== null && a.ok > 0)
    .slice()
    .sort((a, b) => a.concurrency - b.concurrency);
}

function isContended(a: DecodeArm): boolean {
  return (a.foreignMax ?? 0) > 0;
}

/** Half the min–max of the level's per-wave aggregates; 0 when it ran one wave. */
export function spread(a: DecodeArm): number {
  return a.aggregateRange ? (a.aggregateRange[1] - a.aggregateRange[0]) / 2 : 0;
}

/** (agg_i − agg_{i−1}) ÷ (c_i − c_{i−1}) */
function marginalGain(prev: DecodeArm, cur: DecodeArm): number | null {
  if (prev.aggregate === null || cur.aggregate === null) return null;
  const dc = cur.concurrency - prev.concurrency;
  return dc > 0 ? (cur.aggregate - prev.aggregate) / dc : null;
}

/** The marginal gain is below the threshold even at the top of the measured spread. */
function clearlyBelow(prev: DecodeArm, cur: DecodeArm, threshold: number): boolean {
  const gain = marginalGain(prev, cur);
  if (gain === null) return false;
  const dc = cur.concurrency - prev.concurrency;
  return gain + (spread(prev) + spread(cur)) / dc < threshold;
}

/**
 * The gain every later step is judged against: ×1→×2 per added stream. When that
 * step lost throughput (a perturbed wave — the GPU was shared, a strand stalled)
 * it cannot serve as a yardstick, so the best positive step seen stands in.
 */
export function referenceGain(ms: DecodeArm[]): number | null {
  if (ms.length < 2) return null;
  const first = marginalGain(ms[0], ms[1]);
  if (first !== null && first > 0) return first;
  let best: number | null = first;
  for (let i = 2; i < ms.length; i++) {
    const g = marginalGain(ms[i - 1], ms[i]);
    if (g !== null && (best === null || g > best)) best = g;
  }
  return best;
}

export function findKnee(
  arms: DecodeArm[],
  floor: number,
): { knee: number | null; reason: "marginal" | "floor" | null; reference: number | null } {
  const ms = measured(arms).filter((a) => !isContended(a));
  const reference = referenceGain(ms);
  for (let i = 0; i < ms.length; i++) {
    const a = ms[i];
    if (a.perStream !== null && a.perStream < floor) return { knee: a.concurrency, reason: "floor", reference };
    if (i >= 2 && reference !== null && reference > 0 && clearlyBelow(ms[i - 1], a, KNEE_FRACTION * reference)) {
      return { knee: a.concurrency, reason: "marginal", reference };
    }
  }
  return { knee: null, reason: null, reference };
}

export function bestInteractive(arms: DecodeArm[], opts: InterpretOpts): DecodeArm | null {
  let best: DecodeArm | null = null;
  for (const a of measured(arms)) {
    const okStream = a.perStream !== null && a.perStream >= opts.floor;
    const okTtft = a.ttftP50 === null || a.ttftP50 <= opts.sloMs;
    if (okStream && okTtft && (best === null || a.concurrency > best.concurrency)) best = a;
  }
  return best;
}

export function scalingEfficiency(arms: DecodeArm[]): { top: DecodeArm; efficiency: number | null } | null {
  const ms = measured(arms);
  if (!ms.length) return null;
  const first = ms[0];
  const top = ms[ms.length - 1];
  if (ms.length < 2 || first.aggregate === null || top.aggregate === null || first.aggregate <= 0) {
    return { top, efficiency: null };
  }
  const ideal = (top.concurrency / first.concurrency) * first.aggregate;
  return { top, efficiency: ideal > 0 ? top.aggregate / ideal : null };
}

/** The next two preset levels above the top one measured: ×8 → [16, 32], ×4 → [8, 16]. */
export function suggestLevels(top: number): number[] {
  return [2, 4, 8, 16, 32].filter((n) => n > top).slice(0, 2);
}

export function interpretDecode(arms: DecodeArm[], opts: InterpretOpts): DecodeInterpretation {
  const ms = measured(arms);
  const empty: DecodeInterpretation = {
    topLevel: null,
    contended: [],
    efficiency: null,
    perStreamRatio: null,
    bestInteractive: null,
    knee: null,
    kneeReason: null,
    saturated: false,
    suggest: [],
    sentence: "No completed level — nothing to interpret yet.",
  };
  if (!ms.length) return empty;

  const first = ms[0];
  const eff = scalingEfficiency(ms);
  const top = eff?.top ?? ms[ms.length - 1];
  const best = bestInteractive(ms, opts);
  const { knee, reason, reference } = findKnee(ms, opts.floor);
  const perStreamRatio =
    first.perStream && top.perStream && top.perStream > 0 ? first.perStream / top.perStream : null;
  const contended = ms.filter(isContended).map((a) => a.concurrency);
  const clean = ms.filter((a) => !isContended(a));

  let saturated = false;
  if (clean.length >= 3 && reference !== null && reference > 0) {
    saturated = clearlyBelow(clean[clean.length - 2], clean[clean.length - 1], KNEE_FRACTION * reference);
  } else if (clean.length >= 2 && reference !== null) {
    saturated = reference + (spread(clean[0]) + spread(clean[1])) / (clean[1].concurrency - clean[0].concurrency) <= 0;
  }
  const suggest = saturated ? [] : suggestLevels(top.concurrency);

  const parts: string[] = [];
  if (ms.length === 1) {
    // ×1 is one stream: quote its decode rate (= 1 / TPOT), the number the hero's ×1 shows.
    parts.push(
      (top.concurrency === 1
        ? `Single stream: ${fmtTokS(top.perStream)} tok/s`
        : `Single level ×${top.concurrency}: ${fmtTokS(top.aggregate)} tok/s aggregate`) +
        (top.ttftP50 !== null ? `, TTFT p50 ${fmtMs(top.ttftP50)}` : "") +
        " — add levels to see scaling",
    );
  } else {
    if (eff?.efficiency != null) parts.push(`Scaling ${Math.round(eff.efficiency * 100)} % efficient at ×${top.concurrency}`);
    if (top.perStream !== null) {
      const ratio = perStreamRatio !== null ? ` (÷${perStreamRatio.toFixed(1)} vs ×${first.concurrency})` : "";
      parts.push(`per-stream ${fmtTokS(top.perStream)} tok/s${ratio}`);
    }
  }
  if (best) {
    const ttft = best.ttftP50 !== null ? `, TTFT p50 ${fmtMs(best.ttftP50)}` : "";
    parts.push(
      `best interactive point ×${best.concurrency} — ${fmtTokS(best.perStream)} tok/s per stream${ttft}`,
    );
  } else {
    parts.push(`no level holds ${opts.floor} tok/s per stream under ${fmtMs(opts.sloMs)} TTFT`);
  }
  if (knee !== null) {
    parts.push(reason === "floor" ? `knee at ×${knee} (per-stream below the ${opts.floor} tok/s floor)` : `knee at ×${knee}`);
  }
  if (contended.length) parts.push(`${contended.map((c) => `×${c}`).join("/")} contended (foreign requests on the server) — not used for knee or saturation`);
  if (ms.length >= 2) {
    if (saturated) parts.push(`saturated by ×${top.concurrency}`);
    else if (suggest.length) parts.push(`saturation not reached — run ${suggest.map((n) => `×${n}`).join("/")}`);
    else parts.push(`saturation not reached at ×${top.concurrency}`);
  }

  return {
    topLevel: top.concurrency,
    contended,
    efficiency: eff?.efficiency ?? null,
    perStreamRatio,
    bestInteractive: best,
    knee,
    kneeReason: reason,
    saturated,
    suggest,
    sentence: `${parts.join("; ")}.`,
  };
}

// ── Prefill ─────────────────────────────────────────────────────────

export type PrefillInterpretation = {
  peak: PrefillArm | null;
  sustained: PrefillArm | null;
  /** 1 − sustained ÷ peak */
  hold: number | null;
  /** [size, TTFT growth per context doubling] for every consecutive completed pair */
  doublings: Array<{ size: number; ratio: number }>;
  skipped: PrefillArm[];
  sentence: string;
};

export function interpretPrefill(arms: PrefillArm[]): PrefillInterpretation {
  const done = arms
    .filter((a) => !a.skipped && a.ok > 0 && a.prefillTokS !== null)
    .slice()
    .sort((a, b) => a.size - b.size);
  const skipped = arms.filter((a) => a.skipped);
  if (!done.length) {
    return {
      peak: null,
      sustained: null,
      hold: null,
      doublings: [],
      skipped,
      sentence: skipped.length
        ? `Nothing completed; ${skipped.map((a) => fmtSize(a.size)).join(", ")} skipped.`
        : "No completed size — nothing to interpret yet.",
    };
  }
  let peak = done[0];
  for (const a of done) if ((a.prefillTokS ?? 0) > (peak.prefillTokS ?? 0)) peak = a;
  const sustained = done[done.length - 1];
  const hold =
    peak.prefillTokS && sustained.prefillTokS !== null ? Math.max(0, 1 - sustained.prefillTokS / peak.prefillTokS) : null;

  const doublings: Array<{ size: number; ratio: number }> = [];
  for (let i = 1; i < done.length; i++) {
    const a = done[i - 1];
    const b = done[i];
    if (a.ttftMs && b.ttftMs && a.ttftMs > 0 && b.size > a.size) {
      const octaves = Math.log2(b.size / a.size);
      doublings.push({ size: b.size, ratio: Math.pow(b.ttftMs / a.ttftMs, 1 / octaves) });
    }
  }

  const parts: string[] = [];
  if (done.length === 1) {
    parts.push(`${fmtSize(sustained.size)} prefilled at ${fmtTokS(sustained.prefillTokS)} tok/s (TTFT ${fmtMs(sustained.ttftMs)})`);
  } else if (Math.round((hold ?? 0) * 100) === 0) {
    parts.push(`holds its peak (${fmtTokS(peak.prefillTokS)} tok/s at ${fmtSize(peak.size)}) to ${fmtSize(sustained.size)}`);
  } else {
    parts.push(
      `holds within ${Math.round((hold ?? 0) * 100)} % of peak (${fmtTokS(peak.prefillTokS)} tok/s at ${fmtSize(peak.size)}) to ${fmtSize(sustained.size)}`,
    );
  }
  if (doublings.length >= 2) {
    const firstD = doublings[0];
    const lastD = doublings[doublings.length - 1];
    parts.push(
      `TTFT grows ×${lastD.ratio.toFixed(2)} per doubling at ${fmtSize(lastD.size)} vs ×${firstD.ratio.toFixed(2)} at ${fmtSize(firstD.size)}`,
    );
  } else if (doublings.length === 1) {
    parts.push(`TTFT grows ×${doublings[0].ratio.toFixed(2)} per doubling at ${fmtSize(doublings[0].size)}`);
  }
  if (skipped.length) parts.push(`${skipped.map((a) => fmtSize(a.size)).join(", ")} skipped`);
  const sentence = `${parts.join("; ")}.`;
  return { peak, sustained, hold, doublings, skipped, sentence: sentence.charAt(0).toUpperCase() + sentence.slice(1) };
}
