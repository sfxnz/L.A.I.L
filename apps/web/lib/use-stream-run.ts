"use client";

import { useCallback, useEffect, useState } from "react";
import { api, streamRunEventsUrl } from "./api";
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

/**
 * Client side of the A2 SSE protocol: one EventSource per run, a pure reducer
 * over the event union, DOM updates batched per animation frame, and a
 * snapshot re-hydrate when the stream drops. Text per strand is a ring buffer
 * (8k chars); reasoning is kept in its own buffer so the card can dim it.
 */

export const TEXT_RING_CHARS = 8_192;
export const AGG_WINDOW_MS = 60_000;

export type StrandView = {
  i: number;
  title: string;
  pack: string;
  prompt: string;
  /** bench modes: index into hello.levels / hello.sizes */
  level?: number;
  state: StrandState;
  text: string;
  reasoning: string;
  chunks: number;
  ttft_ms?: number;
  tokens?: number;
  tok_s?: number;
  peak_tok_s?: number;
  finish_reason?: string;
  error?: string;
  itl_ms?: number[];
};

export type AggPoint = Omit<StreamAggEvent, "type">;
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
};

export type StreamRunAction =
  | StreamRunEvent
  | { type: "snapshot"; snapshot: StreamRunSnapshot }
  | { type: "reset" };

export function initialStreamRunState(): StreamRunState {
  return { hello: null, strands: [], agg: [], latest: null, levels: [], done: null, error: null };
}

export function appendRing(buf: string, text: string, cap = TEXT_RING_CHARS): string {
  const next = buf + text;
  return next.length > cap ? next.slice(next.length - cap) : next;
}

function blankStrand(i: number): StrandView {
  return { i, title: `Strand ${i + 1}`, pack: "", prompt: "", state: "waiting", text: "", reasoning: "", chunks: 0 };
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

export function streamRunReducer(state: StreamRunState, action: StreamRunAction): StreamRunState {
  switch (action.type) {
    case "reset":
      return initialStreamRunState();
    case "hello": {
      const { type: _t, ...hello } = action;
      void _t;
      const strands = hello.prompts.map((p) => ({
        ...(state.strands[p.i] ?? blankStrand(p.i)),
        i: p.i,
        title: p.title,
        pack: p.pack,
        prompt: p.text,
        level: p.level,
      }));
      return { ...state, hello, strands };
    }
    case "delta":
      return {
        ...state,
        strands: withStrand(state.strands, action.i, (s) => ({
          ...s,
          chunks: action.chunks,
          text: action.reasoning ? s.text : appendRing(s.text, action.text),
          reasoning: action.reasoning ? appendRing(s.reasoning, action.text) : s.reasoning,
        })),
      };
    case "strand": {
      const { type: _t, i, ...rest } = action;
      void _t;
      return { ...state, strands: withStrand(state.strands, i, (s) => ({ ...s, ...rest })) };
    }
    case "agg": {
      const { type: _t, ...point } = action;
      void _t;
      return { ...state, agg: pushAgg(state.agg, point), latest: point };
    }
    case "level": {
      const { type: _t, ...row } = action;
      void _t;
      return { ...state, levels: upsertLevel(state.levels, row) };
    }
    case "done": {
      const { type: _t, ...done } = action;
      void _t;
      return { ...state, done };
    }
    case "error":
      return { ...state, error: action.message };
    case "snapshot": {
      const snap = action.snapshot;
      let strands: StrandView[] = snap.hello.prompts.map((p) => ({
        ...blankStrand(p.i),
        title: p.title,
        pack: p.pack,
        prompt: p.text,
        level: p.level,
      }));
      for (const s of snap.strands) {
        const { text, reasoning_text, ...rest } = s;
        strands = withStrand(strands, s.i, (cur) => ({
          ...cur,
          ...rest,
          text: appendRing("", text),
          reasoning: appendRing("", reasoning_text),
        }));
      }
      const agg = snap.agg.reduce<AggPoint[]>((acc, p) => pushAgg(acc, p), []);
      return {
        hello: snap.hello,
        strands,
        agg,
        latest: agg[agg.length - 1] ?? null,
        levels: snap.levels.reduce<LevelRow[]>((acc, l) => upsertLevel(acc, l), []),
        done: snap.done ?? null,
        error: snap.error ?? null,
      };
    }
    default:
      return state;
  }
}

const EVENT_TYPES = new Set<string>(STREAM_EVENT_TYPES);

export function useStreamRun(runId: string | null) {
  const [state, setState] = useState<StreamRunState>(initialStreamRunState);

  useEffect(() => {
    if (!runId) return;
    setState(initialStreamRunState());

    const queue: StreamRunAction[] = [];
    let raf = 0;
    let terminal = false;
    const es = new EventSource(streamRunEventsUrl(runId));

    const flush = () => {
      raf = 0;
      const batch = queue.splice(0);
      if (batch.length) setState((s) => batch.reduce(streamRunReducer, s));
      if (terminal) es.close(); // the server ended the run; don't let EventSource auto-reconnect
    };
    const push = (a: StreamRunAction) => {
      queue.push(a);
      if (a.type === "done" || a.type === "error") terminal = true;
      if (!raf) raf = requestAnimationFrame(flush);
    };
    const parse = (raw: string, type?: string) => {
      try {
        const data = JSON.parse(raw) as Record<string, unknown>;
        const t = type ?? (typeof data.type === "string" ? data.type : "");
        if (EVENT_TYPES.has(t)) push({ ...data, type: t } as StreamRunEvent);
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
        .then((snapshot) => push({ type: "snapshot", snapshot }))
        .catch(() => {});
    };

    return () => {
      es.close();
      if (raf) cancelAnimationFrame(raf);
    };
  }, [runId]);

  const stop = useCallback(() => (runId ? api.stopStreamRun(runId) : Promise.resolve({ ok: false })), [runId]);

  return { state, stop };
}
