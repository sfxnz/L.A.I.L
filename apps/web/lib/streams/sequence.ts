/**
 * Sequence view geometry — one row per strand: waiting → TTFT → decode → done,
 * with stalls hatched. Built ONLY from what this client observed (the reducer's
 * `at_*` stamps), mapped onto run time through the run clock (`agg.t_ms` paired
 * with its arrival). A strand first seen mid-flight (re-attach) starts where it
 * was first seen — the view never invents history.
 */
import type { StrandState } from "../stream-run-types";

export type SeqInput = {
  i: number;
  state: StrandState;
  at_prefill?: number;
  at_decode?: number;
  at_end?: number;
  at_last_delta?: number;
  stalls: ReadonlyArray<{ from: number; to: number }>;
};

export type SeqClock = { t_ms: number; at: number } | null;

export type SeqSegmentKind = "wait" | "ttft" | "decode" | "stall";
export type SeqSegment = { kind: SeqSegmentKind; x0: number; x1: number };
export type SeqRow = {
  i: number;
  state: StrandState;
  segments: SeqSegment[];
  /** run-ms of the terminal transition, if observed */
  end: number | null;
};
export type SeqLayout = { rows: SeqRow[]; t_min: number; t_max: number };

export const SEQ_STALL_MS = 2000;

/** Client stamp → run-relative ms. Falls back to hello-relative when no agg has arrived. */
export function toRunMs(stamp: number, clock: SeqClock, helloAt: number | null): number {
  if (clock) return clock.t_ms + (stamp - clock.at);
  if (helloAt !== null) return stamp - helloAt;
  return 0;
}

export function sequenceLayout(
  strands: readonly SeqInput[],
  clock: SeqClock,
  helloAt: number | null,
  nowAt: number,
  stallMs = SEQ_STALL_MS,
): SeqLayout {
  const run = (stamp: number) => Math.max(0, toRunMs(stamp, clock, helloAt));
  const now = run(nowAt);
  const t0 = helloAt !== null ? run(helloAt) : 0;
  let t_max = now;
  const rows: SeqRow[] = strands.map((s) => {
    const segments: SeqSegment[] = [];
    const end = s.at_end !== undefined ? run(s.at_end) : null;
    const tail = end ?? now;
    const prefill = s.at_prefill !== undefined ? run(s.at_prefill) : null;
    const decode = s.at_decode !== undefined ? run(s.at_decode) : null;

    if (prefill === null) {
      // never observed leaving the queue
      segments.push({ kind: "wait", x0: t0, x1: tail });
    } else {
      if (prefill > t0) segments.push({ kind: "wait", x0: t0, x1: prefill });
      if (decode === null) segments.push({ kind: "ttft", x0: prefill, x1: tail });
      else {
        segments.push({ kind: "ttft", x0: prefill, x1: Math.max(prefill, decode) });
        segments.push({ kind: "decode", x0: decode, x1: Math.max(decode, tail) });
        for (const st of s.stalls) segments.push({ kind: "stall", x0: run(st.from), x1: run(st.to) });
        if (s.state === "decode" && s.at_last_delta !== undefined && nowAt - s.at_last_delta >= stallMs) {
          segments.push({ kind: "stall", x0: run(s.at_last_delta), x1: now });
        }
      }
    }
    for (const seg of segments) if (seg.x1 > t_max) t_max = seg.x1;
    return { i: s.i, state: s.state, segments, end };
  });
  return { rows, t_min: t0, t_max: Math.max(t_max, t0 + 1) };
}
