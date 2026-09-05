/**
 * Streams engine wire contract — the types live in @lail/shared (PLAN.md A2);
 * this module re-exports them for the web and adds the one runtime constant the
 * SSE consumer needs (the named-event list to subscribe to).
 */
import type { StreamRunEvent } from "@lail/shared";

export type {
  StrandState,
  StreamAggEvent,
  StreamArrival,
  StreamDeltaEvent,
  StreamDoneEvent,
  StreamErrorEvent,
  StreamHelloEvent,
  StreamLevelEvent,
  StreamPack,
  StreamRunEvent,
  StreamRunMode,
  StreamRunRequest,
  StreamRunRow,
  StreamRunSnapshot,
  StreamRunSummary,
  StreamStrandEvent,
  StreamStrandSnapshot,
  StreamThinking,
} from "@lail/shared";

export const STREAM_EVENT_TYPES = [
  "hello",
  "delta",
  "strand",
  "agg",
  "level",
  "done",
  "error",
] as const satisfies ReadonlyArray<StreamRunEvent["type"]>;
