"use client";

import { ApiError, lailTokenHeader } from "./api";
import { isUnauthorizedError } from "./auth-token";
import {
  LAB_STATUS_POLL_MS,
  ingestFailure,
  ingestHistory,
  ingestStatus,
  mergeTick,
  useLabStatusStore,
  type LiveMeta,
} from "./lab-status-store";

/**
 * The transport behind the lab-status store.
 *
 * Primary: GET /api/live, the controller's server-sent event stream (one 1 s
 * sample per event, pushed the moment serve-engine publishes it). It is read
 * with fetch(), not EventSource, so the operator token travels in the
 * X-Lail-Token header like every other call (never in a URL) and a 401 is
 * known by its HTTP status.
 *
 * - A stream that delivers nothing for STALL_MS (the controller pings every 5 s)
 *   is treated as broken and reconnected with backoff.
 * - Every failure asks /api/lab-status once: that answer decides what the user
 *   sees (401 → token banner, no answer → unreachable, an answer → keep going).
 * - When the controller answers but the stream keeps failing (a proxy that
 *   buffers event streams), the store is fed by polling every 2 s instead, and
 *   the stream is retried every RETRY_STREAM_MS. Polling has its own timer and
 *   keeps going through a retry; only a delivered tick stops it.
 * - A tab hidden for HIDDEN_CLOSE_MS closes everything; returning reconnects
 *   (the stream's history event refills the sparklines).
 */

export const STALL_MS = 8000;
const HIDDEN_CLOSE_MS = 60_000;
export const RETRY_STREAM_MS = 30_000;
const MAX_BACKOFF_MS = 10_000;
/** Stream failures in a row (with the controller answering polls) before polling takes over. */
const POLL_AFTER_FAILURES = 2;

/**
 * Minimal SSE reader: `event:` / `data:` fields, dispatched on a blank line.
 * `onBytes` fires for every chunk, comments (pings) included.
 */
export async function readSse(
  body: ReadableStream<Uint8Array>,
  onEvent: (event: string, data: string) => void,
  onBytes?: () => void,
): Promise<void> {
  const reader = body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let event = "message";
  let data: string[] = [];
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    onBytes?.();
    buf += dec.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).replace(/\r$/, "");
      buf = buf.slice(nl + 1);
      if (line === "") {
        if (data.length) onEvent(event, data.join("\n"));
        event = "message";
        data = [];
      } else if (line.startsWith("event:")) {
        event = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        data.push(line.slice(5).replace(/^ /, ""));
      }
      // ":" comments and retry/id fields need no handling here
    }
  }
}

/** Applies one stream event to the store. */
function handleLiveEvent(event: string, data: string): void {
  if (event === "meta") {
    useLabStatusStore.setState({ meta: JSON.parse(data) as LiveMeta });
  } else if (event === "history") {
    ingestHistory(JSON.parse(data));
  } else if (event === "tick") {
    ingestStatus(mergeTick(useLabStatusStore.getState().meta, JSON.parse(data)), "stream");
  }
}

async function errorOf(r: Response): Promise<ApiError> {
  const body = await r.text().catch(() => "");
  let json: ApiError["json"] = null;
  try {
    json = JSON.parse(body);
  } catch {
    /* not JSON */
  }
  return new ApiError(r.status, body || r.statusText, json);
}

/**
 * Start the live transport. Returns a stop function. Only AppShell calls this.
 * `fetchImpl` is injectable for tests.
 */
export function startLive(
  fetchImpl: (input: string, init?: RequestInit) => Promise<Response> = (input, init) => fetch(input, init),
): () => void {
  const doc = typeof document !== "undefined" ? document : null;
  let stopped = false;
  let ac: AbortController | null = null;
  /** the next stream (re)connect */
  let timer: ReturnType<typeof setTimeout> | null = null;
  /** the polling fallback's loop — its own timer, so a stream retry never pauses it */
  let pollTimer: ReturnType<typeof setTimeout> | null = null;
  let hiddenTimer: ReturnType<typeof setTimeout> | null = null;
  let failures = 0;
  let backoff = 1000;
  let sleeping = false; // closed because the tab stayed hidden

  const clearTimer = () => {
    if (timer) clearTimeout(timer);
    timer = null;
  };
  const clearPoll = () => {
    if (pollTimer) clearTimeout(pollTimer);
    pollTimer = null;
  };
  const later = (fn: () => void, ms: number) => {
    clearTimer();
    timer = setTimeout(fn, ms);
  };
  const polling = () => useLabStatusStore.getState().transport === "poll";

  const poll = async () => {
    pollTimer = null;
    if (stopped || sleeping || !polling()) return;
    await useLabStatusStore.getState().refresh();
    if (stopped || sleeping || !polling() || pollTimer) return;
    if (useLabStatusStore.getState().needToken) return; // the banner reloads the page
    pollTimer = setTimeout(poll, LAB_STATUS_POLL_MS);
  };

  const connect = async () => {
    if (stopped || sleeping) return;
    clearTimer();
    const ctrl = new AbortController();
    ac = ctrl;
    let stall: ReturnType<typeof setTimeout> | null = null;
    const arm = () => {
      if (stall) clearTimeout(stall);
      stall = setTimeout(() => ctrl.abort(), STALL_MS);
    };
    let delivered = false;
    let error: unknown = null;
    try {
      arm();
      const r = await fetchImpl("/api/live", {
        headers: { Accept: "text/event-stream", ...lailTokenHeader() },
        signal: ctrl.signal,
        cache: "no-store",
      });
      if (!r.ok || !r.body) throw await errorOf(r);
      await readSse(
        r.body,
        (event, data) => {
          if (event === "tick") {
            delivered = true;
            failures = 0;
            backoff = 1000;
            clearPoll(); // the stream is back: this tick takes over from polling
          }
          handleLiveEvent(event, data);
        },
        arm,
      );
    } catch (e) {
      error = e;
    } finally {
      if (stall) clearTimeout(stall);
      if (ac === ctrl) ac = null;
    }
    if (stopped || sleeping) return;
    if (isUnauthorizedError(error)) {
      clearPoll();
      ingestFailure(error); // the banner reloads the page once a token is pasted
      return;
    }
    if (!delivered && polling()) {
      // A retry from poll mode that failed: the poll loop never stopped; try again later.
      later(() => void connect(), RETRY_STREAM_MS);
      return;
    }
    if (!delivered) failures += 1;
    // What does the controller say? A poll decides between "unreachable" and "keep going".
    const answered = await useLabStatusStore.getState().refresh();
    if (stopped || sleeping || useLabStatusStore.getState().needToken) return;
    if (answered && failures >= POLL_AFTER_FAILURES) {
      useLabStatusStore.setState({ transport: "poll" });
      clearPoll();
      pollTimer = setTimeout(poll, LAB_STATUS_POLL_MS);
      // keep trying the stream now and then; the first tick takes over from polling
      later(() => void connect(), RETRY_STREAM_MS);
      return;
    }
    const wait = delivered ? 1000 : backoff;
    backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
    later(() => void connect(), wait);
  };

  const onVisibility = () => {
    if (!doc) return;
    if (doc.hidden) {
      if (hiddenTimer) return;
      hiddenTimer = setTimeout(() => {
        hiddenTimer = null;
        sleeping = true;
        clearTimer();
        clearPoll();
        ac?.abort();
      }, HIDDEN_CLOSE_MS);
    } else {
      if (hiddenTimer) clearTimeout(hiddenTimer);
      hiddenTimer = null;
      if (sleeping) {
        sleeping = false;
        failures = 0;
        backoff = 1000;
        if (polling()) void poll(); // the fallback resumes at once; the stream is retried too
        void connect();
      }
    }
  };

  void connect();
  doc?.addEventListener("visibilitychange", onVisibility);
  return () => {
    stopped = true;
    clearTimer();
    clearPoll();
    if (hiddenTimer) clearTimeout(hiddenTimer);
    ac?.abort();
    doc?.removeEventListener("visibilitychange", onVisibility);
  };
}
