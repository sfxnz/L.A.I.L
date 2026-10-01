"use client";

import { useEffect, useState } from "react";
import { api, streamRunEventsUrl } from "./api";
import { useLabStatusStore, type LiveRun } from "./lab-status-store";
import {
  STREAM_EVENT_TYPES,
  type StrandState,
  type StreamAggEvent,
  type StreamDoneEvent,
  type StreamHelloEvent,
  type StreamLevelEvent,
  type StreamRunEvent,
  type StreamRunSnapshot,
} from "./stream-run-types";
import { pushSample, stackShares, windowRate, type RateSample } from "./streams/live-rate";
import { STALL_MS } from "./streams/stalls";

/**
 * Client side of the A2 SSE protocol: one EventSource per run, a pure reducer
 * over the event union, DOM updates batched per animation frame (a timer while
 * the tab is hidden, where rAF does not run), and a snapshot re-hydrate when the
 * stream drops. Text per strand is a ring buffer
 * (8k chars); reasoning is kept in its own buffer so the card can dim it.
 *
 * Every action may carry `at` — the client `performance.now()` stamp of its
 * arrival. The reducer uses it (never a clock of its own) to record the
 * transitions it observed per strand (Sequence view), inter-delta stalls, the
 * per-strand rate samples behind the stacked sparkline, and the run clock
 * (`agg.t_ms` paired with its arrival) that maps client stamps onto run time.
 */

export const TEXT_RING_CHARS = 8_192;
export const AGG_WINDOW_MS = 60_000;
/** Decode steps a strand keeps (matches the engine). */
export const STEP_KEEP = 100;
/** Hidden tab: flush the queue on this timer (rAF is paused); also the cap before a forced flush. */
export const HIDDEN_FLUSH_MS = 500;
export const QUEUE_FLUSH_AT = 500;

export type StrandView = {
  i: number;
  title: string;
  pack: string;
  prompt: string;
  /** bench modes: index into hello.levels / hello.sizes */
  level?: number;
  /** bench modes: repeat (wave) of its level */
  wave?: number;
  state: StrandState;
  text: string;
  reasoning: string;
  /** Σ upstream chunks seen (content + reasoning). */
  chunks: number;
  /** Σ upstream chunks that were reasoning. */
  reasoning_chunks: number;
  /** Σ tokens that were reasoning (exact with per-chunk usage). */
  reasoning_tokens: number;
  ttft_ms?: number;
  tokens?: number;
  tok_s?: number;
  peak_tok_s?: number;
  finish_reason?: string;
  error?: string;
  /** Last ≤100 decode steps: gap (ms) and the tokens each carried (see StreamStrandEvent). */
  step_ms?: number[];
  step_tokens?: number[];
  /** Client stamps of the transitions this client observed. */
  at_prefill?: number;
  at_decode?: number;
  at_end?: number;
  at_last_delta?: number;
  /** Inter-delta gaps ≥ STALL_MS this client observed (client stamps). */
  stalls: Array<{ from: number; to: number }>;
  /** (at, cumulative tokens) behind the live per-strand split of the aggregate. */
  samples: RateSample[];
  /** Σ tokens seen in deltas (the samples' counter). */
  delta_tokens: number;
};

export type AggPoint = Omit<StreamAggEvent, "type"> & {
  /** Client-observed split of `tok_s` per strand (Σ = tok_s). Present once deltas were stamped. */
  strand_tok_s?: number[];
};
export type LevelRow = Omit<StreamLevelEvent, "type">;

export type StreamRunState = {
  hello: Omit<StreamHelloEvent, "type"> | null;
  strands: StrandView[];
  /** last 60s of aggregate samples, oldest first */
  agg: AggPoint[];
  latest: AggPoint | null;
  /** level table, ascending by index */
  levels: LevelRow[];
  done: Omit<StreamDoneEvent, "type"> | null;
  error: string | null;
  /** Run clock: the last agg's run-relative `t_ms` paired with its client arrival stamp. */
  clock: { t_ms: number; at: number } | null;
  /** Client stamp of the (last) hello. */
  hello_at: number | null;
};

type Stamped<T> = T & { at?: number };

export type StreamRunAction =
  | Stamped<StreamRunEvent>
  | Stamped<{ type: "snapshot"; snapshot: StreamRunSnapshot }>
  | { type: "reset" };

export function initialStreamRunState(): StreamRunState {
  return {
    hello: null,
    strands: [],
    agg: [],
    latest: null,
    levels: [],
    done: null,
    error: null,
    clock: null,
    hello_at: null,
  };
}

export function appendRing(buf: string, text: string, cap = TEXT_RING_CHARS): string {
  const next = buf + text;
  return next.length > cap ? next.slice(next.length - cap) : next;
}

function blankStrand(i: number): StrandView {
  return {
    i,
    title: `Strand ${i + 1}`,
    pack: "",
    prompt: "",
    state: "waiting",
    text: "",
    reasoning: "",
    chunks: 0,
    reasoning_chunks: 0,
    reasoning_tokens: 0,
    stalls: [],
    samples: [],
    delta_tokens: 0,
  };
}

function withStrand(
  strands: StrandView[],
  i: number,
  patch: (s: StrandView) => StrandView,
): StrandView[] {
  const next = strands.slice();
  while (next.length <= i) next.push(blankStrand(next.length));
  next[i] = patch(next[i]);
  return next;
}

function pushAgg(agg: AggPoint[], point: AggPoint): AggPoint[] {
  const floor = point.t_ms - AGG_WINDOW_MS;
  const kept = agg.length && agg[0].t_ms < floor ? agg.filter((p) => p.t_ms >= floor) : agg;
  return [...kept, point];
}

function upsertLevel(levels: LevelRow[], row: LevelRow): LevelRow[] {
  const next = levels.filter((l) => l.index !== row.index);
  next.push(row);
  next.sort((a, b) => a.index - b.index);
  return next;
}

const TERMINAL: ReadonlySet<StrandState> = new Set(["done", "error", "cancelled"]);
const ACTIVE: ReadonlySet<StrandState> = new Set(["prefill", "decode"]);

/** Stamp the transition this client just observed (first observation wins). */
function stampTransition(s: StrandView, state: StrandState, at: number | undefined): StrandView {
  if (at === undefined) return s;
  const out = { ...s };
  if (state === "prefill" && out.at_prefill === undefined) out.at_prefill = at;
  if (state === "decode") {
    if (out.at_prefill === undefined) out.at_prefill = at;
    if (out.at_decode === undefined) out.at_decode = at;
  }
  if (TERMINAL.has(state) && out.at_end === undefined) out.at_end = at;
  return out;
}

export function streamRunReducer(state: StreamRunState, action: StreamRunAction): StreamRunState {
  switch (action.type) {
    case "reset":
      return initialStreamRunState();
    case "hello": {
      const { type: _t, at, ...hello } = action;
      void _t;
      // The server always follows `hello` with a full replay of each strand's
      // retained text, so start every strand blank — appending onto what we
      // held would duplicate transcripts after an EventSource reconnect.
      const strands = hello.prompts.map((p) => ({
        ...blankStrand(p.i),
        title: p.title,
        pack: p.pack,
        prompt: p.text,
        level: p.level,
        wave: p.wave,
      }));
      return { ...state, hello, strands, hello_at: at ?? state.hello_at };
    }
    case "delta": {
      const at = action.at;
      return {
        ...state,
        strands: withStrand(state.strands, action.i, (s) => {
          const tokens = action.tokens ?? action.chunks;
          const deltaTokens = s.delta_tokens + tokens;
          const next: StrandView = {
            ...s,
            chunks: s.chunks + action.chunks,
            reasoning_chunks: action.reasoning ? s.reasoning_chunks + action.chunks : s.reasoning_chunks,
            reasoning_tokens: action.reasoning ? s.reasoning_tokens + tokens : s.reasoning_tokens,
            delta_tokens: deltaTokens,
            text: action.reasoning ? s.text : appendRing(s.text, action.text),
            reasoning: action.reasoning ? appendRing(s.reasoning, action.text) : s.reasoning,
          };
          if (at !== undefined) {
            next.samples = pushSample(s.samples, at, deltaTokens);
            if (s.at_last_delta !== undefined && at - s.at_last_delta >= STALL_MS) {
              next.stalls = [...s.stalls, { from: s.at_last_delta, to: at }];
            }
            next.at_last_delta = at;
          }
          return next;
        }),
      };
    }
    case "strand": {
      const { type: _t, at, i, steps_append, step_ms, step_tokens, ...rest } = action;
      void _t;
      return {
        ...state,
        strands: withStrand(state.strands, i, (s) => {
          const next: StrandView = { ...s, ...rest };
          if (step_ms) {
            // Live emits carry only the steps since the previous one; terminal emits all of them.
            next.step_ms = (steps_append ? [...(s.step_ms ?? []), ...step_ms] : step_ms).slice(-STEP_KEEP);
            next.step_tokens = (steps_append ? [...(s.step_tokens ?? []), ...(step_tokens ?? [])] : (step_tokens ?? [])).slice(-STEP_KEEP);
          }
          return stampTransition(next, rest.state, at);
        }),
      };
    }
    case "agg": {
      const { type: _t, at, ...rest } = action;
      void _t;
      const point: AggPoint = rest;
      let clock = state.clock;
      if (at !== undefined) {
        clock = { t_ms: point.t_ms, at };
        const rates = state.strands.map((s) => windowRate(s.samples, at));
        const active = state.strands.map((s) => ACTIVE.has(s.state));
        point.strand_tok_s = stackShares(rates, point.tok_s, active);
      }
      return { ...state, agg: pushAgg(state.agg, point), latest: point, clock };
    }
    case "level": {
      const { type: _t, at, ...row } = action;
      void _t;
      void at;
      return { ...state, levels: upsertLevel(state.levels, row) };
    }
    case "done": {
      const { type: _t, at, ...done } = action;
      void _t;
      return {
        ...state,
        done,
        strands: at === undefined ? state.strands : state.strands.map((s) => (TERMINAL.has(s.state) ? s : { ...s, at_end: s.at_end ?? at })),
      };
    }
    case "error":
      return { ...state, error: action.message };
    case "snapshot": {
      const snap = action.snapshot;
      const at = action.at;
      let strands: StrandView[] = snap.hello.prompts.map((p) => ({
        ...blankStrand(p.i),
        title: p.title,
        pack: p.pack,
        prompt: p.text,
        level: p.level,
        wave: p.wave,
      }));
      for (const s of snap.strands) {
        const { text, reasoning_text, steps_append: _a, ...rest } = s;
        void _a;
        strands = withStrand(strands, s.i, (cur) =>
          stampTransition(
            {
              ...cur,
              ...rest,
              text: appendRing("", text),
              reasoning: appendRing("", reasoning_text),
              // The snapshot carries the total only; approximate the split by characters.
              reasoning_chunks: text.length + reasoning_text.length
                ? Math.round((s.chunks * reasoning_text.length) / (text.length + reasoning_text.length))
                : 0,
              reasoning_tokens: s.reasoning_tokens ?? 0,
              delta_tokens: s.tokens ?? 0,
            },
            s.state,
            at,
          ),
        );
      }
      const agg = snap.agg.reduce<AggPoint[]>((acc, p) => pushAgg(acc, p), []);
      const latest = agg[agg.length - 1] ?? null;
      return {
        hello: snap.hello,
        strands,
        agg,
        latest,
        levels: snap.levels.reduce<LevelRow[]>((acc, l) => upsertLevel(acc, l), []),
        done: snap.done ?? null,
        error: snap.error ?? null,
        clock: latest && at !== undefined ? { t_ms: latest.t_ms, at } : null,
        hello_at: at ?? null,
      };
    }
    default:
      return state;
  }
}

const EVENT_TYPES = new Set<string>(STREAM_EVENT_TYPES);

/**
 * When to reduce the queued events: at the next animation frame normally; on a timer
 * while the tab is hidden (browsers pause rAF there, and the queue would grow for the
 * whole run); at once when a backlog has built up.
 */
export function flushPlan(queued: number, hidden: boolean): "now" | "timer" | "frame" {
  if (queued >= QUEUE_FLUSH_AT) return "now";
  return hidden ? "timer" : "frame";
}

/**
 * What the header's instrument strip shows while this run is live: the run's
 * own aggregate (the number Streams/Bench display), not the endpoint counter
 * rate. Null once the run is done, errored, or detached.
 */
export function liveRunOf(state: StreamRunState): LiveRun | null {
  if (!state.hello || state.done || state.error) return null;
  const l = state.latest;
  return {
    tok_s: l?.tok_s ?? 0,
    peak: l?.peak_tok_s ?? 0,
    running: l?.running ?? 0,
    waiting: l?.waiting ?? 0,
    source: state.hello.mode === "load" ? "streams" : "bench",
  };
}

/**
 * One EventSource for `runId`. Read-only: stopping a run is `api.stopStreamRun`,
 * never a side effect of detaching from it.
 */
export function useStreamRun(runId: string | null) {
  const [state, setState] = useState<StreamRunState>(initialStreamRunState);

  // Publish the run aggregate to the shared store (the header reads it).
  // setLiveRun is a no-op when nothing changed, so this cannot loop hydration.
  useEffect(() => {
    useLabStatusStore.getState().setLiveRun(runId ? liveRunOf(state) : null);
  }, [runId, state]);
  useEffect(() => () => useLabStatusStore.getState().setLiveRun(null), []);

  useEffect(() => {
    if (!runId) {
      setState(initialStreamRunState());
      return;
    }
    setState(initialStreamRunState());

    const queue: StreamRunAction[] = [];
    let raf = 0;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let terminal = false;
    const es = new EventSource(streamRunEventsUrl(runId));

    const flush = () => {
      if (raf) cancelAnimationFrame(raf);
      if (timer) clearTimeout(timer);
      raf = 0;
      timer = null;
      const batch = queue.splice(0);
      if (batch.length) setState((s) => batch.reduce(streamRunReducer, s));
    };
    const push = (a: StreamRunAction) => {
      queue.push(a);
      if (a.type === "done" || a.type === "error") {
        // The server ended the run: close now (no EventSource auto-reconnect), not at the next frame.
        terminal = true;
        es.close();
      }
      const plan = flushPlan(queue.length, document.hidden);
      if (plan === "now") flush();
      else if (plan === "timer") {
        if (!timer) timer = setTimeout(flush, HIDDEN_FLUSH_MS);
      } else if (!raf) raf = requestAnimationFrame(flush);
    };
    const parse = (raw: string, type?: string) => {
      try {
        const data = JSON.parse(raw) as Record<string, unknown>;
        const t = type ?? (typeof data.type === "string" ? data.type : "");
        if (EVENT_TYPES.has(t)) push({ ...data, type: t, at: performance.now() } as StreamRunAction);
      } catch {
        /* malformed frame — skip */
      }
    };

    for (const t of STREAM_EVENT_TYPES) {
      es.addEventListener(t, (e) => parse((e as MessageEvent).data, t));
    }
    es.onmessage = (e) => parse(e.data);

    let errors = 0;
    es.onerror = () => {
      if (terminal) return;
      errors += 1;
      // EventSource retries on its own; after a run of failures, resync from
      // the snapshot so the view doesn't miss what happened while we were out.
      if (errors < 3) return;
      errors = 0;
      void api
        .streamRunSnapshot(runId)
        .then((snapshot) => push({ type: "snapshot", snapshot, at: performance.now() }))
        .catch(() => {});
    };

    return () => {
      es.close();
      if (raf) cancelAnimationFrame(raf);
      if (timer) clearTimeout(timer);
    };
  }, [runId]);

  return { state };
}
