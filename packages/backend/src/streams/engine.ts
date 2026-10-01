import { z } from "zod";
import type {
  BenchEnvelope,
  StreamAggEvent,
  StreamArrival,
  StreamDeltaEvent,
  StreamHelloEvent,
  StreamLevelEvent,
  StreamPromptRef,
  StreamRunEvent,
  StreamRunMode,
  StreamRunRow,
  StreamRunSnapshot,
  StreamRunStatus,
  StreamRunSummary,
  StreamStrandEvent,
  StreamStrandSnapshot,
  StreamThinking,
  StrandState,
} from "@lail/shared";
import { LIVE_RATE_WINDOW_MS, TAIL_MIN_SAMPLES, type BenchHardware } from "@lail/shared";
import { config } from "../config";
import { getSettings, openAiBase } from "../controller/settings";
import { assignPrompts, getPack, strandSystemPrompt } from "./packs";
import { SseParser } from "./sse-parser";
import {
  buildEnvelope,
  decodeTokPerS,
  headline as benchHeadline,
  peakArm,
  percentile,
  skippedLevelEvent,
  summarizeLevel,
  summarizeWave,
  toArm,
  toLevelEvent,
  type BenchArmInput,
  type LevelContext,
  type StrandResult,
} from "./metrics";
import { buildPrefillPrompt, vllmTokenCounter, type PrefillSizer } from "./prefill";
import {
  acquireBenchLease,
  readServeStatus,
  releaseBenchLease,
  scrapeMetrics,
  serverDelta,
  serverLoad,
  summarizeHardware,
  type MetricSample,
  type StatusReading,
} from "./probes";

export class StreamsError extends Error {
  constructor(
    public status: 400 | 404 | 409 | 502,
    public code: string,
    message: string,
    public extra: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "StreamsError";
  }
}

const requestSchema = z.object({
  mode: z.enum(["load", "bench-decode", "bench-prefill"]),
  base_url: z.string().min(1).optional(),
  model: z.string().min(1).optional(),
  pack: z.string().min(1),
  n: z.number().int().min(1).max(32).optional(),
  levels: z.array(z.number().int().min(1).max(32)).min(1).max(16).optional(),
  sizes: z.array(z.number().int().min(16).max(1_048_576)).min(1).max(16).optional(),
  samples: z.number().int().min(1).max(16).optional(),
  max_tokens: z.number().int().min(1).max(65_536).default(512),
  fill_to_max: z.boolean().default(false),
  thinking: z.enum(["auto", "on", "off"]).default("off"),
  temperature: z.number().min(0).max(2).default(0.2),
  arrival: z.enum(["burst", "staggered", "poisson"]).default("burst"),
  arrival_ms: z.number().int().min(0).max(60_000).default(250),
});

export type NormalizedRequest = {
  mode: StreamRunMode;
  base_url: string;
  model?: string;
  pack: string;
  n: number;
  levels: number[];
  sizes: number[];
  samples: number;
  max_tokens: number;
  fill_to_max: boolean;
  thinking: StreamThinking;
  temperature: number;
  arrival: StreamArrival;
  arrival_ms: number;
};

const DEFAULT_N = 4;
const DEFAULT_LEVELS = [1, 2, 4, 8];
const DEFAULT_SIZES = [8192, 16384, 32768, 65536, 131072];
/** bench-decode: minimum strands per level; bench-prefill: requests per size. */
const DEFAULT_SAMPLES = { "bench-decode": 3, "bench-prefill": 2, load: 1 } as const;
const TEXT_CAP = 8000;
const STEP_KEEP = 100;
/** agg samples kept for reconnect: 60 s at 4 Hz. */
const AGG_KEEP = 240;
/** Queued events beyond which a subscriber is considered dead and closed. */
const QUEUE_HARD_CAP = 2048;
const RECENT_KEEP = 50;
/** See `LIVE_RATE_WINDOW_MS`. */
export const RATE_WINDOW_MS = LIVE_RATE_WINDOW_MS;
/** The first second of a run divides by 1 s, not less: it ramps instead of spiking. */
const RATE_MIN_SPAN_MS = 1000;
/** Bench: one discarded request first (CUDA graphs, caches, scheduler), same pack. */
const WARMUP_TOKENS = 16;
/** Bench: before each level, wait this long for foreign requests on the server to drain. */
const IDLE_WAIT_MS = 10_000;
const FOREIGN_POLL_MS = 1000;
/** Foreign-load sampling: a strand that ended this long before a scrape began still counts as ours (gauge lag). */
const OWN_GRACE_MS = 250;
const HARDWARE_POLL_MS = 2000;
const LEASE_TTL_S = 90;
const LEASE_RENEW_MS = 30_000;

const TERMINAL: ReadonlySet<StrandState> = new Set(["done", "error", "cancelled"]);
const now = () => performance.now();
const r1 = (v: number) => Math.round(v * 10) / 10;
const r2 = (v: number | null) => (v === null ? null : Math.round(v * 100) / 100);
const sToMs = (v: number | null) => (v === null ? null : r1(v * 1000));

/**
 * Bounded per-subscriber event queue. A lagging subscriber never loses text: a `delta`
 * for a strand whose previous delta (same text/reasoning stream) is still queued is
 * merged into it, so text queues at most one event per stream however far it lags.
 */
export class Subscriber {
  private queue: StreamRunEvent[] = [];
  /** The queued (not yet consumed) delta per `i:reasoning` stream. */
  private queuedDelta = new Map<string, StreamDeltaEvent>();
  private wake: (() => void) | null = null;
  closed = false;

  push(ev: StreamRunEvent) {
    if (this.closed) return;
    if (ev.type === "delta") {
      const key = `${ev.i}:${ev.reasoning ? 1 : 0}`;
      const held = this.queuedDelta.get(key);
      if (held) {
        held.text += ev.text;
        held.chunks += ev.chunks;
        held.tokens += ev.tokens;
        return;
      }
      const own = { ...ev };
      this.queuedDelta.set(key, own);
      this.queue.push(own);
    } else {
      this.queue.push(ev);
    }
    if (this.queue.length > QUEUE_HARD_CAP) {
      this.close();
      return;
    }
    const w = this.wake;
    this.wake = null;
    w?.();
  }

  /** Next event, or null once closed and drained. */
  next(): Promise<StreamRunEvent | null> {
    if (this.queue.length) {
      const ev = this.queue.shift()!;
      if (ev.type === "delta") {
        const key = `${ev.i}:${ev.reasoning ? 1 : 0}`;
        if (this.queuedDelta.get(key) === ev) this.queuedDelta.delete(key);
      }
      return Promise.resolve(ev);
    }
    if (this.closed) return Promise.resolve(null);
    return new Promise<void>((res) => {
      this.wake = res;
    }).then(() => this.next());
  }

  close() {
    this.closed = true;
    this.queue = [];
    this.queuedDelta.clear();
    const w = this.wake;
    this.wake = null;
    w?.();
  }
}

type Strand = {
  ref: StreamPromptRef;
  /** Prefill: the built prompt (the ref only carries a description). */
  promptText?: string;
  state: StrandState;
  ctrl: AbortController;
  t_start: number | null;
  t_first: number | null;
  t_last: number | null;
  t_end: number | null;
  chunks: number;
  textChunks: number;
  reasoningChunks: number;
  /** Cumulative completion tokens: the latest per-chunk `usage.completion_tokens`, else 1 per chunk. */
  tokens: number;
  /** Every token-bearing chunk carried cumulative usage, so `tokens` at t_last is exact. */
  exact: boolean;
  textTokens: number;
  reasoningTokens: number;
  /** Tokens delivered with the first token-bearing chunk. */
  firstTokens: number | null;
  usage: { prompt_tokens?: number; completion_tokens?: number } | null;
  finish_reason: string | null;
  error: string | null;
  /** Per token-bearing chunk: ms since t_start, and cumulative tokens at it. */
  token_times: number[];
  token_counts: number[];
  /** Last ≤STEP_KEEP decode steps: gap before the chunk (ms) and the tokens it carried. */
  steps: number[];
  stepTokens: number[];
  stepsTotal: number;
  /** stepsTotal at the last live strand emit (live emits carry only newer steps). */
  stepsSent: number;
  text: string;
  reasoning: string;
  pending: { text: string; reasoning: string; textChunks: number; reasoningChunks: number; textTokens: number; reasoningTokens: number };
  peak_tok_s: number;
};

type Run = {
  id: string;
  req: NormalizedRequest;
  model: string;
  max_model_len: number | null;
  max_tokens: number;
  fill_to_max: boolean;
  status: StreamRunStatus;
  started_at: string;
  ended_at: string | null;
  t0: number;
  strands: Strand[];
  levels: StreamLevelEvent[];
  arms: BenchArmInput[];
  ctrl: AbortController;
  subscribers: Set<Subscriber>;
  /** Token arrivals inside the live rate window. */
  window: Array<{ t: number; n: number }>;
  firstTokenAt: number | null;
  peak_tok_s: number;
  /** False once a chunk arrived without `usage` (counted as 1 token). */
  tokensExact: boolean;
  aggSeries: StreamAggEvent[];
  /** Bench: serve fingerprint at start; status-snapshot node readings over the run. */
  fingerprint: string | null;
  hardware: BenchHardware["series"];
  /** Bench: each node's last recorded `sampled_at` — a reading is recorded once. */
  hardwareAt: Map<string, number>;
  /** Bench: each level's measured span [from, to] (ms since t0) — what energy is integrated over. */
  levelWindows: Array<[number, number]>;
  /** Bench: holds the serve-engine bench lease (null when the serve-engine was unreachable). */
  lease: boolean;
  summary: StreamRunSummary | null;
  saved_run_id: string | null;
  error: string | null;
  flushTimer: ReturnType<typeof setInterval> | null;
  aggTimer: ReturnType<typeof setInterval> | null;
  idleTimer: ReturnType<typeof setTimeout> | null;
  hardwareTimer: ReturnType<typeof setInterval> | null;
  leaseTimer: ReturnType<typeof setInterval> | null;
};

/** One bench level's server probes: /metrics before, foreign load sampled while it runs. */
type LevelProbe = {
  before: MetricSample | null;
  foreign: number | null;
  timer: ReturnType<typeof setInterval> | null;
  /** When the level's measured span began (after the idle-drain wait), performance.now() ms. */
  t_begin: number;
};

function canonBase(raw: string): string | null {
  let u: URL;
  try {
    u = new URL(raw);
  } catch {
    return null;
  }
  if (u.protocol !== "http:" && u.protocol !== "https:") return null;
  if (u.search || u.hash || u.username || u.password) return null;
  const path = u.pathname.replace(/\/+$/, "").replace(/\/v1$/, "");
  return `${u.origin}${path}`;
}

function capTail(s: string, cap: number): string {
  return s.length > cap ? s.slice(s.length - cap) : s;
}

function stripType<T extends { type: string }>(ev: T): Omit<T, "type"> {
  const { type, ...rest } = ev;
  void type;
  return rest;
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((res) => {
    if (signal.aborted) return res();
    const onAbort = () => {
      clearTimeout(t);
      res();
    };
    const t = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      res();
    }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export function arrivalDelays(count: number, arrival: StreamArrival, arrivalMs: number): number[] {
  const out: number[] = [];
  let t = 0;
  for (let k = 0; k < count; k++) {
    if (k > 0) {
      if (arrival === "staggered") t += arrivalMs;
      else if (arrival === "poisson") t += -Math.log(1 - Math.random()) * arrivalMs;
    }
    out.push(Math.round(t));
  }
  return out;
}

/** Strands of ours in flight at any moment since `since` (started, and not ended before it). */
export function ownSince(strands: Array<Pick<Strand, "t_start" | "t_end">>, since: number): number {
  return strands.filter((s) => s.t_start !== null && (s.t_end === null || s.t_end >= since)).length;
}

function toResult(s: Strand): StrandResult {
  const usageN = s.usage?.completion_tokens;
  const fromUsage = typeof usageN === "number";
  return {
    i: s.ref.i,
    ok: s.state === "done",
    t_start: s.t_start ?? 0,
    t_first: s.t_first,
    t_last: s.t_last,
    t_end: s.t_end ?? s.t_last ?? s.t_start ?? 0,
    completion_tokens: fromUsage ? usageN : s.tokens || null,
    last_tokens: s.exact && s.t_last !== null ? s.tokens : null,
    first_tokens: s.firstTokens ?? 1,
    prompt_tokens: typeof s.usage?.prompt_tokens === "number" ? s.usage.prompt_tokens : null,
    estimated: !fromUsage,
    finish_reason: s.finish_reason,
    error: s.error ?? (s.state === "cancelled" ? "cancelled" : null),
    token_times_ms: s.token_times,
    token_counts: s.token_counts,
    ...(s.ref.wave !== undefined ? { wave: s.ref.wave } : {}),
  };
}

/** Span a live window rate divides by: the window, or the time since the first token (≥ 1 s) before it fills. */
function rateSpanMs(t: number, firstAt: number): number {
  return Math.min(RATE_WINDOW_MS, Math.max(RATE_MIN_SPAN_MS, t - firstAt));
}

/** The strand's tokens that arrived in the last RATE_WINDOW_MS ÷ that span (live per-strand tok/s). */
export function strandWindowRate(
  s: Pick<Strand, "t_start" | "t_first" | "token_times" | "token_counts">,
  t: number,
): number {
  if (s.t_start === null || s.t_first === null || !s.token_counts.length) return 0;
  const cutoff = t - s.t_start - RATE_WINDOW_MS;
  let k = s.token_times.length - 1;
  while (k >= 0 && s.token_times[k] > cutoff) k--;
  const inWindow = s.token_counts[s.token_counts.length - 1] - (k >= 0 ? s.token_counts[k] : 0);
  return inWindow / (rateSpanMs(t, s.t_first) / 1000);
}

export class StreamsEngine {
  private runs = new Map<string, Run>();
  private order: string[] = [];
  private idleAbortMs: number;
  private flushMs: number;
  private aggMs: number;
  private idleWaitMs: number;

  constructor(opts: { idleAbortMs?: number; flushMs?: number; aggMs?: number; idleWaitMs?: number } = {}) {
    this.idleAbortMs = opts.idleAbortMs ?? 5000;
    this.flushMs = opts.flushMs ?? 80;
    this.aggMs = opts.aggMs ?? 250;
    this.idleWaitMs = opts.idleWaitMs ?? IDLE_WAIT_MS;
  }

  // ── Request validation ──────────────────────────────────────────

  parseRequest(body: unknown): NormalizedRequest {
    const parsed = requestSchema.safeParse(body);
    if (!parsed.success) {
      const msg = parsed.error.issues.map((i) => `${i.path.join(".") || "body"}: ${i.message}`).join("; ");
      throw new StreamsError(400, "invalid_request", msg);
    }
    const r = parsed.data;
    if (!getPack(r.pack)) throw new StreamsError(400, "unknown_pack", `unknown pack: ${r.pack}`);
    const base_url = this.validateBaseUrl(r.base_url);
    const uniqAsc = (xs: number[]) => [...new Set(xs)].sort((a, b) => a - b);
    return {
      mode: r.mode,
      base_url,
      model: r.model,
      pack: r.pack,
      n: r.n ?? DEFAULT_N,
      levels: uniqAsc(r.levels ?? DEFAULT_LEVELS),
      sizes: uniqAsc(r.sizes ?? DEFAULT_SIZES),
      samples: r.samples ?? DEFAULT_SAMPLES[r.mode],
      max_tokens: r.max_tokens,
      fill_to_max: r.fill_to_max,
      thinking: r.thinking,
      temperature: r.temperature,
      arrival: r.arrival,
      arrival_ms: r.arrival_ms,
    };
  }

  /** A configured (enabled) backend URL or `http://127.0.0.1:<port>`; anything else is rejected. */
  private validateBaseUrl(raw: string | undefined): string {
    const base = canonBase(raw ?? openAiBase());
    if (!base) throw new StreamsError(400, "invalid_base_url", `base_url is not a valid http(s) URL: ${raw}`);
    const u = new URL(base);
    const loopback = u.protocol === "http:" && u.hostname === "127.0.0.1" && u.port !== "" && u.pathname === "/";
    const configured = Object.values(getSettings().backends)
      .filter((b) => b.enabled)
      .map((b) => canonBase(b.url));
    if (!loopback && !configured.includes(base)) {
      throw new StreamsError(
        400,
        "base_url_not_allowed",
        `base_url must be a configured backend or http://127.0.0.1:<port>, got ${base}`,
      );
    }
    return base;
  }

  private activeFor(base_url: string): Run | undefined {
    for (const run of this.runs.values()) {
      if (run.status === "running" && run.req.base_url === base_url) return run;
    }
    return undefined;
  }

  private async resolveModel(base_url: string, requested?: string): Promise<{ model: string; max_model_len: number | null }> {
    let data: Array<{ id: string; max_model_len?: number }>;
    try {
      const r = await fetch(`${base_url}/v1/models`, { signal: AbortSignal.timeout(5000) });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = (await r.json()) as { data?: Array<{ id: string; max_model_len?: number }> };
      data = j.data ?? [];
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      throw new StreamsError(502, "models_unreachable", `${base_url}/v1/models: ${msg}`);
    }
    const pick = (requested ? data.find((m) => m.id === requested) : undefined) ?? data[0];
    if (!pick && !requested) throw new StreamsError(502, "no_model", `Nothing is served at ${base_url}/v1/models`);
    const len = pick?.max_model_len;
    return { model: requested ?? pick!.id, max_model_len: typeof len === "number" ? len : null };
  }

  // ── Lifecycle ───────────────────────────────────────────────────

  async createRun(body: unknown): Promise<string> {
    const req = this.parseRequest(body);
    const conflict = () => {
      const active = this.activeFor(req.base_url);
      if (active) {
        throw new StreamsError(409, "run_active", `a run is already active on ${req.base_url}`, { run_id: active.id });
      }
    };
    conflict();
    const { model, max_model_len } = await this.resolveModel(req.base_url, req.model);
    conflict();

    const id = `${Date.now().toString(36)}-${crypto.randomUUID().slice(0, 6)}`;
    const isBench = req.mode !== "load";
    // Bench: one GPU, one measurement — take the serve-engine bench lease (its agentic
    // evals hold the other side). An unreachable serve-engine does not block the run.
    let status: StatusReading | null = null;
    let lease = false;
    if (isBench) {
      const got = await acquireBenchLease(id, LEASE_TTL_S);
      if (got.ok === false) {
        const what = typeof got.holder.job_id === "string" ? `serve-engine job ${got.holder.job_id} (${String(got.holder.kind ?? "bench")})` : "another bench";
        throw new StreamsError(409, "bench_busy", `${what} is measuring this GPU — wait for it to finish`, got.holder);
      }
      lease = got.ok === true;
      status = await readServeStatus();
      if (this.activeFor(req.base_url)) {
        if (lease) void releaseBenchLease(id);
        conflict();
      }
    }

    const pack = getPack(req.pack)!;
    const refs: StreamPromptRef[] = [];
    if (req.mode === "load") {
      refs.push(...assignPrompts(pack, req.n));
    } else if (req.mode === "bench-decode") {
      for (const [level, c] of req.levels.entries()) {
        const waves = Math.ceil(req.samples / c);
        for (let w = 0; w < waves; w++) {
          refs.push(...assignPrompts(pack, c, refs.length, level).map((r) => ({ ...r, wave: w })));
        }
      }
    } else {
      for (const [level, size] of req.sizes.entries()) {
        for (let w = 0; w < req.samples; w++) {
          refs.push({ i: refs.length, title: `prefill_${size}`, text: `unique-prefix filler, target ${size} tokens`, pack: req.pack, level, wave: w });
        }
      }
    }
    const run: Run = {
      id,
      req,
      model,
      max_model_len,
      max_tokens: req.mode === "bench-prefill" ? 1 : req.max_tokens,
      fill_to_max: req.mode === "load" ? req.fill_to_max : true,
      status: "running",
      started_at: new Date().toISOString(),
      ended_at: null,
      t0: now(),
      strands: refs.map((ref) => ({
        ref,
        state: "waiting",
        ctrl: new AbortController(),
        t_start: null,
        t_first: null,
        t_last: null,
        t_end: null,
        chunks: 0,
        textChunks: 0,
        reasoningChunks: 0,
        tokens: 0,
        textTokens: 0,
        reasoningTokens: 0,
        firstTokens: null,
        exact: true,
        usage: null,
        finish_reason: null,
        error: null,
        token_times: [],
        token_counts: [],
        steps: [],
        stepTokens: [],
        stepsTotal: 0,
        stepsSent: 0,
        text: "",
        reasoning: "",
        pending: { text: "", reasoning: "", textChunks: 0, reasoningChunks: 0, textTokens: 0, reasoningTokens: 0 },
        peak_tok_s: 0,
      })),
      levels: [],
      arms: [],
      ctrl: new AbortController(),
      subscribers: new Set(),
      window: [],
      firstTokenAt: null,
      peak_tok_s: 0,
      tokensExact: true,
      aggSeries: [],
      fingerprint: status?.fingerprint ?? null,
      hardware: [],
      hardwareAt: new Map(),
      levelWindows: [],
      lease,
      summary: null,
      saved_run_id: null,
      error: null,
      flushTimer: null,
      aggTimer: null,
      idleTimer: null,
      hardwareTimer: null,
      leaseTimer: null,
    };
    this.runs.set(id, run);
    this.order.push(id);
    this.evict();
    if (isBench) {
      this.recordHardware(run, status);
      run.hardwareTimer = setInterval(() => void readServeStatus().then((st) => this.recordHardware(run, st)), HARDWARE_POLL_MS);
      if (lease) run.leaseTimer = setInterval(() => void acquireBenchLease(id, LEASE_TTL_S), LEASE_RENEW_MS);
    }
    void this.execute(run);
    // Nobody may ever attach (palette start, tab closed first): a load run with no
    // subscriber is aborted like one whose last subscriber left.
    if (req.mode === "load") this.armIdle(run);
    return id;
  }

  private recordHardware(run: Run, st: StatusReading | null) {
    if (!st || run.status !== "running") return;
    const t = Math.round(now() - run.t0);
    for (const n of st.nodes) {
      if (n.at === null || n.at === run.hardwareAt.get(n.id)) continue;
      run.hardwareAt.set(n.id, n.at);
      run.hardware.push([t, n.id, n.temp, n.power, n.avail]);
    }
  }

  private evict() {
    while (this.order.length > RECENT_KEEP) {
      const victim = this.order.find((id) => this.runs.get(id)?.status !== "running");
      if (!victim) return;
      this.order.splice(this.order.indexOf(victim), 1);
      this.runs.delete(victim);
    }
  }

  /** Abort every stream; the run finishes as `cancelled`. */
  stop(id: string): boolean {
    const run = this.runs.get(id);
    if (!run) return false;
    if (run.status !== "running") return true;
    run.ctrl.abort();
    for (const s of run.strands) s.ctrl.abort();
    return true;
  }

  private async execute(run: Run) {
    run.flushTimer = setInterval(() => this.flush(run), this.flushMs);
    run.aggTimer = setInterval(() => this.tickAgg(run), this.aggMs);
    try {
      if (run.req.mode === "load") await this.runWave(run, run.strands, run.req.arrival, run.req.arrival_ms);
      else {
        await this.warmup(run);
        if (run.req.mode === "bench-decode") await this.runBenchDecode(run);
        else await this.runBenchPrefill(run);
      }
      await this.finish(run);
    } catch (e) {
      if (run.ctrl.signal.aborted && run.status === "running") await this.finish(run);
      else this.fail(run, e instanceof Error ? e.message : String(e));
    }
  }

  /** One discarded request before measuring: same pack, own nonce, a few tokens. */
  private async warmup(run: Run) {
    const pack = getPack(run.req.pack)!;
    const body: Record<string, unknown> = {
      model: run.model,
      messages: [
        { role: "system", content: `Warmup ${run.id}. Answer the user directly.` },
        { role: "user", content: pack.prompts[0].text },
      ],
      max_tokens: WARMUP_TOKENS,
      temperature: run.req.temperature,
      stream: false,
    };
    if (run.req.thinking !== "auto") body.chat_template_kwargs = { enable_thinking: run.req.thinking === "on" };
    const res = await fetch(`${run.req.base_url}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: run.ctrl.signal,
    });
    const txt = await res.text();
    if (!res.ok) throw new Error(`warmup request failed: HTTP ${res.status}${txt ? `: ${txt.slice(0, 300)}` : ""}`);
  }

  /**
   * Before a level: wait (≤ IDLE_WAIT_MS) for requests that are not ours to drain, scrape
   * /metrics, then sample foreign load at 1 Hz while the level runs. Foreign = server
   * running+waiting − ours. The server's count is taken at some moment during the scrape
   * (and its gauge can trail a step), so ours = every strand in flight at any moment of
   * [scrape start − OWN_GRACE_MS, scrape end]: a strand finishing inside that window is
   * still ours, and foreign never over-reports.
   */
  private async beginLevel(run: Run): Promise<LevelProbe> {
    const signal = run.ctrl.signal;
    const t0 = now();
    let before = await scrapeMetrics(run.req.base_url, signal);
    let load = serverLoad(before);
    while (load !== null && load > 0 && now() - t0 < this.idleWaitMs && !signal.aborted) {
      await sleep(500, signal);
      before = await scrapeMetrics(run.req.base_url, signal);
      load = serverLoad(before);
    }
    const probe: LevelProbe = { before, foreign: load, timer: null, t_begin: now() };
    if (load === null) return probe;
    let busy = false;
    probe.timer = setInterval(async () => {
      if (busy) return;
      busy = true;
      try {
        const since = now() - OWN_GRACE_MS;
        const l = serverLoad(await scrapeMetrics(run.req.base_url, signal));
        if (l === null) return;
        probe.foreign = Math.max(probe.foreign ?? 0, l - ownSince(run.strands, since));
      } finally {
        busy = false;
      }
    }, FOREIGN_POLL_MS);
    return probe;
  }

  private async endLevel(run: Run, probe: LevelProbe): Promise<LevelContext> {
    if (probe.timer) clearInterval(probe.timer);
    if (!run.ctrl.signal.aborted) run.levelWindows.push([Math.round(probe.t_begin - run.t0), Math.round(now() - run.t0)]);
    const after = run.ctrl.signal.aborted ? null : await scrapeMetrics(run.req.base_url, run.ctrl.signal);
    return { foreign_max: probe.foreign, server: serverDelta(probe.before, after) };
  }

  private async runWave(run: Run, strands: Strand[], arrival: StreamArrival, arrivalMs: number) {
    const delays = arrivalDelays(strands.length, arrival, arrivalMs);
    await Promise.all(
      strands.map(async (s, k) => {
        if (delays[k] > 0) await sleep(delays[k], run.ctrl.signal);
        if (run.ctrl.signal.aborted) {
          this.setState(run, s, "cancelled");
          return;
        }
        await this.runStrand(run, s);
      }),
    );
  }

  /** Each level: ⌈samples ÷ c⌉ sequential burst waves of c strands. */
  private async runBenchDecode(run: Run) {
    for (const [index, c] of run.req.levels.entries()) {
      if (run.ctrl.signal.aborted) break;
      const strands = run.strands.filter((s) => s.ref.level === index);
      const waveCount = Math.max(...strands.map((s) => s.ref.wave ?? 0)) + 1;
      const probe = await this.beginLevel(run);
      const waves: StrandResult[][] = [];
      for (let w = 0; w < waveCount && !run.ctrl.signal.aborted; w++) {
        const wave = strands.filter((s) => (s.ref.wave ?? 0) === w);
        await this.runWave(run, wave, "burst", 0);
        if (!run.ctrl.signal.aborted) waves.push(wave.map(toResult));
      }
      const ctx = await this.endLevel(run, probe);
      if (run.ctrl.signal.aborted) break;
      const ws = summarizeLevel(waves);
      run.arms.push({ key: { concurrency: c }, ws, results: waves.flat(), ctx });
      this.pushLevel(run, toLevelEvent(index, { concurrency: c }, ws, ctx));
    }
  }

  /**
   * Each size: `samples` sequential requests (max_tokens 1), each with its own nonce'd
   * prompt so prefix caching cannot serve a repeat. Prefill tok/s = prompt_tokens / TTFT
   * on prompts of 8k+ tokens; the level also carries vLLM's own prefill rate.
   */
  private async runBenchPrefill(run: Run) {
    const sizer: PrefillSizer = { charsPerToken: 3.5 };
    const count = vllmTokenCounter(run.req.base_url, run.model, run.ctrl.signal);
    for (const [index, size] of run.req.sizes.entries()) {
      if (run.ctrl.signal.aborted) break;
      const strands = run.strands.filter((s) => s.ref.level === index);
      if (run.max_model_len !== null && size >= run.max_model_len) {
        const reason = `size ${size} ≥ max_model_len ${run.max_model_len}`;
        for (const s of strands) {
          s.error = reason;
          this.setState(run, s, "cancelled");
        }
        run.arms.push({ key: { size }, ws: null, results: [], skipped: reason });
        this.pushLevel(run, skippedLevelEvent(index, { size }, reason));
        continue;
      }
      const probe = await this.beginLevel(run);
      const waves: StrandResult[][] = [];
      for (const s of strands) {
        if (run.ctrl.signal.aborted) break;
        const built = await buildPrefillPrompt(sizer, count, size, `${run.id}-${size}-${s.ref.wave ?? 0}`);
        s.promptText = built.text;
        await this.runWave(run, [s], "burst", 0);
        if (run.ctrl.signal.aborted) break;
        const result = toResult(s);
        if (result.prompt_tokens === null) {
          result.prompt_tokens = built.measured ?? Math.round(built.text.length / sizer.charsPerToken);
          result.estimated = true;
        } else if (sizer.tokenizeAvailable === false && result.prompt_tokens > 0) {
          sizer.charsPerToken = built.text.length / result.prompt_tokens;
        }
        waves.push([result]);
      }
      const ctx = await this.endLevel(run, probe);
      if (run.ctrl.signal.aborted) break;
      const ws = summarizeLevel(waves);
      run.arms.push({ key: { size }, ws, results: waves.flat(), ctx });
      this.pushLevel(run, toLevelEvent(index, { size }, ws, ctx));
    }
  }

  private pushLevel(run: Run, lvl: StreamLevelEvent) {
    run.levels.push(lvl);
    this.emit(run, lvl);
  }

  private bodyFor(run: Run, s: Strand): Record<string, unknown> {
    // Prefill prompts are already unique (nonce'd filler) and sized via /tokenize on the
    // user turn alone, so they get no system message.
    const messages = s.promptText
      ? [{ role: "user", content: s.promptText }]
      : [
          { role: "system", content: strandSystemPrompt(run.id, s.ref.i) },
          { role: "user", content: s.ref.text },
        ];
    const body: Record<string, unknown> = {
      model: run.model,
      messages,
      max_tokens: run.max_tokens,
      temperature: run.req.temperature,
      stream: true,
      // continuous_usage_stats: vLLM puts the cumulative `usage` on every chunk, so every
      // chunk's exact token count is known (MTP chunks carry 1–4 tokens).
      stream_options: { include_usage: true, continuous_usage_stats: true },
    };
    if (run.fill_to_max) {
      body.min_tokens = run.max_tokens;
      body.ignore_eos = true;
    }
    if (run.req.thinking !== "auto") body.chat_template_kwargs = { enable_thinking: run.req.thinking === "on" };
    return body;
  }

  private async runStrand(run: Run, s: Strand) {
    s.t_start = now();
    this.setState(run, s, "prefill");
    try {
      const res = await fetch(`${run.req.base_url}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json", accept: "text/event-stream" },
        body: JSON.stringify(this.bodyFor(run, s)),
        signal: s.ctrl.signal,
      });
      if (!res.ok) {
        const txt = (await res.text().catch(() => "")).slice(0, 300);
        throw new Error(`HTTP ${res.status}${txt ? `: ${txt}` : ""}`);
      }
      if (!res.body) throw new Error("empty response body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      const parser = new SseParser();
      let finished = false;
      while (!finished) {
        const { done, value } = await reader.read();
        const payloads = done ? parser.end() : parser.feed(decoder.decode(value, { stream: true }));
        for (const p of payloads) {
          if (p === "[DONE]") {
            finished = true;
            break;
          }
          this.handleChunk(run, s, p);
        }
        if (done) break;
      }
      if (finished) reader.cancel().catch(() => {});
      s.t_end = now();
      if (!s.error && s.chunks === 0) s.error = "no_output";
      this.setState(run, s, s.error ? "error" : "done");
    } catch (e) {
      s.t_end = now();
      if (s.ctrl.signal.aborted) {
        this.setState(run, s, "cancelled");
      } else {
        s.error = e instanceof Error ? e.message : String(e);
        this.setState(run, s, "error");
      }
    }
  }

  private handleChunk(run: Run, s: Strand, payload: string) {
    let obj: {
      error?: unknown;
      usage?: { prompt_tokens?: number; completion_tokens?: number } | null;
      choices?: Array<{ finish_reason?: string | null; delta?: Record<string, unknown> }>;
    };
    try {
      obj = JSON.parse(payload);
    } catch {
      return;
    }
    if (obj.error) {
      const err = obj.error as { message?: string };
      s.error = typeof obj.error === "string" ? obj.error : err.message || JSON.stringify(obj.error).slice(0, 300);
      return;
    }
    if (obj.usage && typeof obj.usage === "object") s.usage = obj.usage;
    const choice = obj.choices?.[0];
    if (!choice) return;
    if (choice.finish_reason) s.finish_reason = choice.finish_reason;
    const delta = choice.delta ?? {};
    const str = (v: unknown) => (typeof v === "string" ? v : "");
    // Content, else reasoning — either counts as output.
    const content = str(delta.content);
    const reasoning = content ? "" : str(delta.reasoning_content) || str(delta.reasoning);
    if (!content && !reasoning) return;

    const t = now();
    // Exact tokens in this chunk from the cumulative usage it carries; 1 when it has none.
    const cum = obj.usage?.completion_tokens;
    let n: number;
    if (typeof cum === "number") {
      n = Math.max(0, cum - s.tokens);
      s.tokens = Math.max(s.tokens, cum);
    } else {
      n = 1;
      s.tokens += 1;
      s.exact = false;
      run.tokensExact = false;
    }
    s.chunks++;
    if (n > 0) {
      const first = s.t_first === null;
      if (first) {
        s.t_first = t;
        s.firstTokens = n;
      } else {
        s.steps.push(r1(t - (s.t_last ?? t)));
        s.stepTokens.push(n);
        s.stepsTotal++;
        if (s.steps.length > STEP_KEEP) {
          s.steps.shift();
          s.stepTokens.shift();
        }
      }
      s.t_last = t;
      s.token_times.push(r1(t - (s.t_start ?? t)));
      s.token_counts.push(s.tokens);
      run.window.push({ t, n });
      if (run.firstTokenAt === null) run.firstTokenAt = t;
    }
    if (content) {
      s.textChunks++;
      s.textTokens += n;
      s.pending.textChunks++;
      s.pending.textTokens += n;
      s.pending.text += content;
      s.text = capTail(s.text + content, TEXT_CAP);
    } else {
      s.reasoningChunks++;
      s.reasoningTokens += n;
      s.pending.reasoningChunks++;
      s.pending.reasoningTokens += n;
      s.pending.reasoning += reasoning;
      s.reasoning = capTail(s.reasoning + reasoning, TEXT_CAP);
    }
    if (n > 0 && s.state === "prefill") this.setState(run, s, "decode");
  }

  private setState(run: Run, s: Strand, state: StrandState) {
    s.state = state;
    this.emit(run, this.strandEvent(s));
  }

  /**
   * `full`: every retained step (terminal emits, snapshots, a new subscriber's replay).
   * Otherwise a live decode emit carries only the steps since the previous one.
   */
  private strandEvent(s: Strand, full = false): StreamStrandEvent {
    const ev: StreamStrandEvent = { type: "strand", i: s.ref.i, state: s.state };
    if (s.t_first !== null && s.t_start !== null) ev.ttft_ms = r1(s.t_first - s.t_start);
    if (TERMINAL.has(s.state)) {
      const r = toResult(s);
      ev.tokens = r.completion_tokens ?? 0;
      const rate = r2(decodeTokPerS(r));
      if (rate !== null) ev.tok_s = rate;
      ev.peak_tok_s = s.peak_tok_s;
      if (s.finish_reason) ev.finish_reason = s.finish_reason;
      if (s.error) ev.error = s.error;
      ev.step_ms = s.steps.slice();
      ev.step_tokens = s.stepTokens.slice();
    } else if (s.state === "decode") {
      ev.tokens = s.tokens;
      ev.peak_tok_s = s.peak_tok_s;
      if (full) {
        ev.step_ms = s.steps.slice();
        ev.step_tokens = s.stepTokens.slice();
      } else {
        const fresh = Math.min(s.stepsTotal - s.stepsSent, s.steps.length);
        s.stepsSent = s.stepsTotal;
        if (fresh > 0) {
          ev.step_ms = s.steps.slice(-fresh);
          ev.step_tokens = s.stepTokens.slice(-fresh);
          ev.steps_append = true;
        }
      }
    }
    return ev;
  }

  // ── Ticks ───────────────────────────────────────────────────────

  private flush(run: Run) {
    for (const s of run.strands) {
      const p = s.pending;
      if (p.reasoningChunks) {
        this.emit(run, { type: "delta", i: s.ref.i, text: p.reasoning, reasoning: true, chunks: p.reasoningChunks, tokens: p.reasoningTokens });
        p.reasoning = "";
        p.reasoningChunks = 0;
        p.reasoningTokens = 0;
      }
      if (p.textChunks) {
        this.emit(run, { type: "delta", i: s.ref.i, text: p.text, reasoning: false, chunks: p.textChunks, tokens: p.textTokens });
        p.text = "";
        p.textChunks = 0;
        p.textTokens = 0;
      }
    }
  }

  private tickAgg(run: Run) {
    const t = now();
    const cutoff = t - RATE_WINDOW_MS;
    let drop = 0;
    while (drop < run.window.length && run.window[drop].t <= cutoff) drop++;
    if (drop) run.window.splice(0, drop);
    let inWindow = 0;
    for (const w of run.window) inWindow += w.n;
    const sinceFirst = run.firstTokenAt === null ? 0 : t - run.firstTokenAt;
    const tok_s = run.firstTokenAt === null ? 0 : r1(inWindow / (rateSpanMs(t, run.firstTokenAt) / 1000));
    if (sinceFirst >= RATE_WINDOW_MS && tok_s > run.peak_tok_s) run.peak_tok_s = tok_s;

    let tokens = 0;
    let running = 0;
    let waiting = 0;
    let done = 0;
    const ttfts: number[] = [];
    for (const s of run.strands) {
      if (TERMINAL.has(s.state)) {
        done++;
        tokens += toResult(s).completion_tokens ?? 0;
      } else {
        tokens += s.tokens;
        if (s.state === "waiting") waiting++;
        else running++;
      }
      if (s.t_first !== null && s.t_start !== null) ttfts.push(s.t_first - s.t_start);
    }
    ttfts.sort((a, b) => a - b);
    const agg: StreamAggEvent = {
      type: "agg",
      t_ms: Math.round(t - run.t0),
      tok_s,
      peak_tok_s: run.peak_tok_s,
      tokens,
      running,
      waiting,
      done,
      tokens_exact: run.tokensExact,
    };
    const p50 = percentile(ttfts, 50);
    // Tail percentiles follow the same rule as the levels and the summary.
    const p95 = ttfts.length >= TAIL_MIN_SAMPLES ? percentile(ttfts, 95) : null;
    if (p50 !== null) agg.ttft_p50_ms = r1(p50);
    if (p95 !== null) agg.ttft_p95_ms = r1(p95);
    run.aggSeries.push(agg);
    if (run.aggSeries.length > AGG_KEEP) run.aggSeries.shift();
    this.emit(run, agg);

    // Every decoding strand, every tick: a stalled strand's rate decays to 0 instead of
    // holding its last value.
    for (const s of run.strands) {
      if (s.state !== "decode") continue;
      const rate = r1(strandWindowRate(s, t));
      if (s.t_first !== null && t - s.t_first >= RATE_WINDOW_MS && rate > s.peak_tok_s) s.peak_tok_s = rate;
      this.emit(run, { ...this.strandEvent(s), tok_s: rate });
    }
  }

  private stopTimers(run: Run) {
    if (run.flushTimer) clearInterval(run.flushTimer);
    if (run.aggTimer) clearInterval(run.aggTimer);
    if (run.idleTimer) clearTimeout(run.idleTimer);
    if (run.hardwareTimer) clearInterval(run.hardwareTimer);
    if (run.leaseTimer) clearInterval(run.leaseTimer);
    run.flushTimer = run.aggTimer = run.idleTimer = run.hardwareTimer = run.leaseTimer = null;
    if (run.lease) {
      run.lease = false;
      void releaseBenchLease(run.id);
    }
  }

  // ── Completion ──────────────────────────────────────────────────

  private async finish(run: Run) {
    const isBench = run.req.mode !== "load";
    const cancelled = run.ctrl.signal.aborted;
    // A bench where no level produced a single ok strand measured nothing: it is a
    // failure, never a stored result.
    if (isBench && !cancelled && !run.arms.some((a) => (a.ws?.ok ?? 0) > 0)) {
      const errs = run.arms.flatMap((a) => a.ws?.errors ?? (a.skipped ? [a.skipped] : []));
      this.fail(run, `no level completed — nothing saved${errs.length ? ` (${errs.slice(0, 3).join("; ")})` : ""}`);
      return;
    }
    if (isBench) this.recordHardware(run, await readServeStatus());
    this.stopTimers(run);
    if (cancelled) for (const s of run.strands) if (s.state === "waiting") this.setState(run, s, "cancelled");
    this.flush(run);
    this.tickAgg(run);
    const results = run.strands.map(toResult);
    const kind = run.req.mode === "bench-decode" ? "decode" : "prefill";
    const arms = run.arms.map((a) => toArm(a.key, a.ws, a.skipped, a.ctx));
    const head = isBench ? benchHeadline(kind, arms) : undefined;
    let saved: string | null = null;
    const tokens = results.reduce((a, r) => a + (r.completion_tokens ?? 0), 0);
    // Prefill answers are 1 token each: energy per *output* token would be meaningless there.
    const hardware =
      isBench && run.hardware.length ? summarizeHardware(run.hardware, run.levelWindows, kind === "decode" ? tokens : null) : null;
    if (isBench && !cancelled) {
      const envelope = buildEnvelope({
        kind,
        model: run.model,
        workload: {
          pack: run.req.pack,
          ...(run.req.mode === "bench-decode" ? { levels: run.req.levels } : { sizes: run.req.sizes }),
          samples: run.req.samples,
          max_tokens: run.max_tokens,
          thinking: run.req.thinking,
          temperature: run.req.temperature,
          fill_to_max: run.fill_to_max,
          base_url: run.req.base_url,
          serve_fingerprint: run.fingerprint,
        },
        arms: run.arms,
        hardware,
      });
      saved = await this.importEnvelope(envelope);
    }
    // Cancelled: keep the numbers that were observed. A strand that produced output
    // counts as ok, so aggregate spans the window actually run and TTFT percentiles
    // cover every strand that reached first token.
    if (cancelled) for (const r of results) if (r.t_first !== null && r.error === "cancelled") r.ok = true;
    const ws = summarizeWave(results);
    run.status = cancelled ? "cancelled" : "done";
    run.ended_at = new Date().toISOString();
    run.saved_run_id = saved;
    // Bench: the run-level numbers are the headline's (never one aggregate across levels
    // run one after the other); load: one wave over every strand.
    const c1 = arms.find((a) => a.concurrency === 1) ?? null;
    const peak = peakArm(arms);
    run.summary = {
      status: run.status,
      mode: run.req.mode,
      model: run.model,
      duration_ms: Math.round(now() - run.t0),
      // Every strand's output counts here (cancelled ones too); ws.* below use ok strands only.
      tokens,
      peak_tok_s: run.peak_tok_s,
      aggregate_tok_s: head ? head.aggregate_peak_tok_per_s : ws.aggregate_tok_s,
      aggregate_steady_tok_s: head ? (peak?.aggregate_steady_tok_per_s ?? null) : ws.aggregate_steady_tok_s,
      per_stream_median_tok_s: head ? head.decode_tok_per_s_median_c1 : ws.per_stream_median_tok_s,
      ttft_p50_ms: head ? sToMs(c1?.ttft_s.p50 ?? null) : sToMs(ws.ttft_s.p50),
      ttft_p95_ms: head ? sToMs(c1?.ttft_s.p95 ?? null) : sToMs(ws.ttft_s.p95),
      ok: ws.ok,
      requests: ws.requests,
      errors: ws.errors,
      ...(head ? { headline: head } : {}),
      ...(hardware ? { hardware: { nodes: hardware.nodes, energy_j: hardware.energy_j, energy_j_per_token: hardware.energy_j_per_token } } : {}),
    };
    this.emit(run, { type: "done", run_id: run.id, summary: run.summary, saved_run_id: saved });
  }

  private fail(run: Run, message: string) {
    this.stopTimers(run);
    run.ctrl.abort();
    for (const s of run.strands) s.ctrl.abort();
    run.status = "error";
    run.error = message;
    run.ended_at = new Date().toISOString();
    run.summary = {
      status: "error",
      mode: run.req.mode,
      model: run.model,
      duration_ms: Math.round(now() - run.t0),
      tokens: 0,
      peak_tok_s: run.peak_tok_s,
      aggregate_tok_s: null,
      aggregate_steady_tok_s: null,
      per_stream_median_tok_s: null,
      ttft_p50_ms: null,
      ttft_p95_ms: null,
      ok: 0,
      requests: run.strands.length,
      errors: [message],
      error: message,
    };
    console.error(`[streams] run ${run.id} failed: ${message}`);
    this.emit(run, { type: "error", message });
    this.emit(run, { type: "done", run_id: run.id, summary: run.summary, saved_run_id: null });
  }

  private async importEnvelope(envelope: BenchEnvelope): Promise<string | null> {
    const url = `${config.serveEngineUrl}/api/runs/import`;
    try {
      const headers: Record<string, string> = { "content-type": "application/json" };
      if (config.token) headers["x-lail-token"] = config.token;
      const r = await fetch(url, { method: "POST", headers, body: JSON.stringify(envelope), signal: AbortSignal.timeout(15_000) });
      if (!r.ok) {
        console.warn(`[streams] run import failed: HTTP ${r.status} from ${url}`);
        return null;
      }
      const j = (await r.json()) as { run_id?: unknown };
      return typeof j.run_id === "string" ? j.run_id : null;
    } catch (e) {
      console.warn(`[streams] run import failed: ${e instanceof Error ? e.message : String(e)}`);
      return null;
    }
  }

  // ── Fan-out ─────────────────────────────────────────────────────

  private emit(run: Run, ev: StreamRunEvent) {
    for (const sub of run.subscribers) sub.push(ev);
  }

  private hello(run: Run): StreamHelloEvent {
    return {
      type: "hello",
      run_id: run.id,
      mode: run.req.mode,
      model: run.model,
      base_url: run.req.base_url,
      n: run.strands.length,
      ...(run.req.mode === "bench-decode" ? { levels: run.req.levels } : {}),
      ...(run.req.mode === "bench-prefill" ? { sizes: run.req.sizes } : {}),
      ...(run.req.mode !== "load" ? { samples: run.req.samples, serve_fingerprint: run.fingerprint } : {}),
      max_tokens: run.max_tokens,
      max_model_len: run.max_model_len,
      started_at: run.started_at,
      prompts: run.strands.map((s) => s.ref),
    };
  }

  /** `hello` + snapshot (strand states, retained text, levels, retained agg history) then live events. */
  subscribe(id: string): Subscriber | null {
    const run = this.runs.get(id);
    if (!run) return null;
    const sub = new Subscriber();
    // `s.text` already contains what is still sitting in `s.pending`; drain that to
    // the current subscribers first, or the next flush re-sends it to this one and
    // the transcript shows the join-point text twice ("YouYou have to…").
    this.flush(run);
    sub.push(this.hello(run));
    for (const s of run.strands) {
      if (s.state === "waiting") continue;
      sub.push(this.strandEvent(s, true));
      if (s.reasoning) sub.push({ type: "delta", i: s.ref.i, text: s.reasoning, reasoning: true, chunks: s.reasoningChunks, tokens: s.reasoningTokens });
      if (s.text) sub.push({ type: "delta", i: s.ref.i, text: s.text, reasoning: false, chunks: s.textChunks, tokens: s.textTokens });
    }
    for (const lvl of run.levels) sub.push(lvl);
    // The whole retained series (≤ AGG_KEEP), not just the last point, so a re-attached
    // or finished run can draw its Helix / Sequence history.
    for (const agg of run.aggSeries) sub.push(agg);
    if (run.status !== "running") {
      if (run.status === "error" && run.error) sub.push({ type: "error", message: run.error });
      sub.push({ type: "done", run_id: run.id, summary: run.summary!, saved_run_id: run.saved_run_id });
      return sub;
    }
    run.subscribers.add(sub);
    if (run.idleTimer) {
      clearTimeout(run.idleTimer);
      run.idleTimer = null;
    }
    return sub;
  }

  /** Load runs are aborted when the last subscriber leaves and nobody reattaches in time. */
  unsubscribe(id: string, sub: Subscriber) {
    sub.close();
    const run = this.runs.get(id);
    if (!run) return;
    run.subscribers.delete(sub);
    this.armIdle(run);
  }

  /** Abort a load run that nobody watches after idleAbortMs (subscribe() disarms it). */
  private armIdle(run: Run) {
    if (run.req.mode !== "load" || run.status !== "running" || run.subscribers.size > 0 || run.idleTimer) return;
    run.idleTimer = setTimeout(() => {
      run.idleTimer = null;
      if (run.status === "running" && run.subscribers.size === 0) this.stop(run.id);
    }, this.idleAbortMs);
  }

  // ── Read models ─────────────────────────────────────────────────

  snapshot(id: string): StreamRunSnapshot | null {
    const run = this.runs.get(id);
    if (!run) return null;
    const strands: StreamStrandSnapshot[] = run.strands.map((s) => ({
      ...stripType(this.strandEvent(s, true)),
      text: s.text,
      reasoning_text: s.reasoning,
      chunks: s.chunks,
      reasoning_tokens: s.reasoningTokens,
    }));
    return {
      hello: stripType(this.hello(run)),
      status: run.status,
      strands,
      agg: run.aggSeries.map(stripType),
      levels: run.levels.map(stripType),
      done: run.summary ? { run_id: run.id, summary: run.summary, saved_run_id: run.saved_run_id } : null,
      error: run.error,
    };
  }

  list(): StreamRunRow[] {
    return [...this.order]
      .reverse()
      .map((id) => this.runs.get(id)!)
      .map((run) => ({
        run_id: run.id,
        mode: run.req.mode,
        model: run.model,
        base_url: run.req.base_url,
        pack: run.req.pack,
        n: run.strands.length,
        status: run.status,
        started_at: run.started_at,
        finished_at: run.ended_at,
        summary: run.summary,
        saved_run_id: run.saved_run_id,
      }));
  }
}

export const streamsEngine = new StreamsEngine();
