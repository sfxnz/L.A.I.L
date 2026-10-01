/**
 * Live-run helpers over the stream-run state: which level is running, the
 * gated per-strand tok/s pill, the per-level strand roll-up, and a sampler for
 * the hardware strip. Pure, so the instrument stays a thin renderer.
 */
import { median } from "@lail/shared";
import type { ClusterNode } from "../api";
import type { NodeSample } from "../lab-status-store";
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
  // Live strand rates are re-emitted every tick, so a stalled strand reads its decayed rate, not its last one.
  const medianRate = median(rates);
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

/**
 * Fold the store's per-node series (the ONE sampler, lab-status-store) into the
 * run's series: append every sample newer than what we hold and not older than
 * `since` (the run start). Both are the serve-engine host's clock (sample `t` is a
 * node's server `sampled_at`). Nodes come and go, so series are keyed by id.
 */
export function appendHardware(
  prev: HardwareSeries[],
  samples: Record<string, NodeSample[]>,
  nodes: ClusterNode[] | undefined,
  since: number,
): HardwareSeries[] {
  if (!nodes?.length) return prev;
  const byId = new Map(prev.map((s) => [s.id, s]));
  let changed = false;
  const next: HardwareSeries[] = [];
  for (const n of nodes) {
    const cur = byId.get(n.id) ?? { id: n.id, label: n.label || n.hostname || n.id, samples: [] };
    const lastT = cur.samples[cur.samples.length - 1]?.t ?? since - 1;
    const fresh = (samples[n.id] ?? []).filter((s) => s.t > lastT && s.t >= since);
    if (!fresh.length) {
      next.push(cur);
      continue;
    }
    changed = true;
    const add: HardwareSample[] = fresh.map((s) => ({ t: s.t, util: s.util, power: s.power, temp: s.temp }));
    next.push({ ...cur, samples: [...cur.samples, ...add].slice(-HARDWARE_KEEP) });
  }
  return changed || next.length !== prev.length ? next : prev;
}
