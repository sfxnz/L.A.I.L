/**
 * Live-run helpers over the stream-run state: which level is running, the
 * gated per-strand tok/s pill, the per-level strand roll-up, and a sampler for
 * the hardware strip. Pure, so the instrument stays a thin renderer.
 */
import type { ClusterNode } from "../api";
import type { StrandView } from "../use-stream-run";

export const PILL_MIN_TOKENS = 4;
export const PILL_MIN_MS = 500;

/**
 * A rate is shown only once the strand has ≥ 4 tokens and has been decoding
 * ≥ 500 ms; before that the pill stays quiet instead of printing a bogus
 * 1000 t/s off two chunks. Finished strands show their final rate.
 */
export function gatedRate(s: StrandView, decodeStartedAt: number | undefined, now: number): number | null {
  if (s.state === "done") return s.tok_s ?? null;
  if (s.state !== "decode") return null;
  if ((s.tokens ?? 0) < PILL_MIN_TOKENS) return null;
  if (decodeStartedAt === undefined || now - decodeStartedAt < PILL_MIN_MS) return null;
  return s.tok_s ?? null;
}

export type LevelRollup = {
  total: number;
  done: number;
  errors: number;
  running: number;
  waiting: number;
  /** Σ tokens ÷ (total × maxTokens) */
  fill: number;
  /** median live rate of decoding strands (tok/s) */
  medianRate: number | null;
  waitingFirstToken: number;
  tokens: number[];
};

export function rollupLevel(strands: StrandView[], maxTokens: number): LevelRollup {
  let done = 0;
  let errors = 0;
  let running = 0;
  let waiting = 0;
  let waitingFirstToken = 0;
  let tokens = 0;
  const rates: number[] = [];
  const perStrand: number[] = [];
  for (const s of strands) {
    const t = s.tokens ?? 0;
    perStrand.push(t);
    tokens += t;
    switch (s.state) {
      case "done":
        done++;
        break;
      case "error":
      case "cancelled":
        errors++;
        break;
      case "waiting":
        waiting++;
        waitingFirstToken++;
        break;
      case "prefill":
        running++;
        waitingFirstToken++;
        break;
      case "decode":
        running++;
        if (s.tok_s != null && s.tok_s > 0) rates.push(s.tok_s);
        break;
    }
  }
  rates.sort((a, b) => a - b);
  const medianRate = rates.length ? rates[rates.length >> 1] : null;
  const cap = strands.length * Math.max(1, maxTokens);
  return {
    total: strands.length,
    done,
    errors,
    running,
    waiting,
    fill: cap > 0 ? Math.min(1, tokens / cap) : 0,
    medianRate,
    waitingFirstToken,
    tokens: perStrand,
  };
}

/** Index of the level being run: one past the last `level` event, clamped. */
export function currentLevelIndex(levelEvents: number, levelCount: number): number {
  return Math.min(levelEvents, Math.max(0, levelCount - 1));
}

// ── Hardware strip sampler ──────────────────────────────────────────

export type HardwareSample = { t: number; util: number | null; power: number | null; temp: number | null };
export type HardwareSeries = { id: string; label: string; samples: HardwareSample[] };

export const HARDWARE_KEEP = 120;

/** Append one sample per node; nodes come and go, so series are keyed by id. */
export function sampleHardware(prev: HardwareSeries[], nodes: ClusterNode[] | undefined, t: number): HardwareSeries[] {
  if (!nodes?.length) return prev;
  const byId = new Map(prev.map((s) => [s.id, s]));
  const next: HardwareSeries[] = [];
  for (const n of nodes) {
    const cur = byId.get(n.id) ?? { id: n.id, label: n.label || n.hostname || n.id, samples: [] };
    const sample: HardwareSample = {
      t,
      util: typeof n.gpu_util_pct === "number" ? n.gpu_util_pct : null,
      power: typeof n.power_w === "number" ? n.power_w : null,
      temp: typeof n.temperature_c === "number" ? n.temperature_c : null,
    };
    const last = cur.samples[cur.samples.length - 1];
    const samples = last && last.t === t ? cur.samples : [...cur.samples, sample].slice(-HARDWARE_KEEP);
    next.push({ ...cur, samples });
  }
  return next;
}
