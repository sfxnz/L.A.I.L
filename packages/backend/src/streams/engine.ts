import { z } from "zod";
import type {
  BenchEnvelope,
  BenchHeadline,
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
import { config } from "../config";
import { getSettings, openAiBase } from "../controller/settings";
import { assignPrompts, getPack } from "./packs";
import { SseParser } from "./sse-parser";
import {
  buildEnvelope,
  decodeTokPerS,
  percentile,
  skippedLevelEvent,
  summarizeWave,
  toLevelEvent,
  type BenchArmInput,
  type StrandResult,
} from "./metrics";
import { buildPrefillPrompt, vllmTokenCounter, type PrefillSizer } from "./prefill";

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
const TEXT_CAP = 8000;
const ITL_KEEP = 100;
/** agg samples kept for reconnect: 60 s at 4 Hz. */
const AGG_KEEP = 240;
/** Queued events beyond which `delta` text is dropped for a lagging subscriber. */
const DELTA_DROP_AT = 64;
/** Queued events beyond which a subscriber is considered dead and closed. */
const QUEUE_HARD_CAP = 2048;
const RECENT_KEEP = 50;

const TERMINAL: ReadonlySet<StrandState> = new Set(["done", "error", "cancelled"]);
const now = () => performance.now();
const r1 = (v: number) => Math.round(v * 10) / 10;
const r2 = (v: number | null) => (v === null ? null : Math.round(v * 100) / 100);

/** Bounded per-subscriber event queue. Snapshot pushes bypass the delta-drop rule. */
export class Subscriber {
  private queue: StreamRunEvent[] = [];
  private wake: (() => void) | null = null;
  closed = false;

  push(ev: StreamRunEvent, force = false) {
    if (this.closed) return;
    if (!force && ev.type === "delta" && this.queue.length >= DELTA_DROP_AT) return;
    this.queue.push(ev);
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
    if (this.queue.length) return Promise.resolve(this.queue.shift()!);
    if (this.closed) return Promise.resolve(null);
    return new Promise<void>((res) => {
      this.wake = res;
    }).then(() => this.next());
  }

  close() {
    this.closed = true;
    this.queue = [];
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
  usage: { prompt_tokens?: number; completion_tokens?: number } | null;
  finish_reason: string | null;
  error: string | null;
  token_times: number[];
  itl: number[];
  text: string;
  reasoning: string;
  pending: { text: string; reasoning: string; textChunks: number; reasoningChunks: number };
  peak_tok_s: number;
  dirty: boolean;
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
  chunkTimes: number[];
  peak_tok_s: number;
  aggSeries: StreamAggEvent[];
  /** Live token estimate: tokens per upstream chunk, calibrated from finished strands' `usage`. */
  tpc: number;
  calibTokens: number;
  calibChunks: number;
  summary: StreamRunSummary | null;
  saved_run_id: string | null;
  error: string | null;
  flushTimer: ReturnType<typeof setInterval> | null;
  aggTimer: ReturnType<typeof setInterval> | null;
  idleTimer: ReturnType<typeof setTimeout> | null;
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
    completion_tokens: fromUsage ? usageN : s.chunks || null,
    prompt_tokens: typeof s.usage?.prompt_tokens === "number" ? s.usage.prompt_tokens : null,
    estimated: !fromUsage,
    finish_reason: s.finish_reason,
    error: s.error ?? (s.state === "cancelled" ? "cancelled" : null),
    token_times_ms: s.token_times,
  };
}

/** Chunks in the strand's last 1 s (live per-strand tok/s estimate). */
function strandWindowRate(s: Strand, t: number): number {
  if (s.t_start === null) return 0;
  const cutoff = t - s.t_start - 1000;
  let n = 0;
  for (let k = s.token_times.length - 1; k >= 0 && s.token_times[k] > cutoff; k--) n++;
  return n;
}

export class StreamsEngine {
  private runs = new Map<string, Run>();
  private order: string[] = [];
  /** Last calibrated tokens-per-chunk per `base_url model`, so the next run starts calibrated. */
  private tokensPerChunk = new Map<string, number>();
  private idleAbortMs: number;
  private flushMs: number;
  private aggMs: number;

  constructor(opts: { idleAbortMs?: number; flushMs?: number; aggMs?: number } = {}) {
    this.idleAbortMs = opts.idleAbortMs ?? 5000;
    this.flushMs = opts.flushMs ?? 80;
    this.aggMs = opts.aggMs ?? 250;
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
    const pack = getPack(req.pack)!;
    const refs: StreamPromptRef[] = [];
    if (req.mode === "load") {
      refs.push(...assignPrompts(pack, req.n, id));
    } else if (req.mode === "bench-decode") {
      for (const [level, c] of req.levels.entries()) refs.push(...assignPrompts(pack, c, `${id}-L${level}`, refs.length, level));
    } else {
      for (const [level, size] of req.sizes.entries()) {
        refs.push({ i: level, title: `prefill_${size}`, text: `unique-prefix filler, target ${size} tokens`, pack: req.pack, level });
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
        usage: null,
        finish_reason: null,
        error: null,
        token_times: [],
        itl: [],
        text: "",
        reasoning: "",
        pending: { text: "", reasoning: "", textChunks: 0, reasoningChunks: 0 },
        peak_tok_s: 0,
        dirty: false,
      })),
      levels: [],
      arms: [],
      ctrl: new AbortController(),
      subscribers: new Set(),
      chunkTimes: [],
      peak_tok_s: 0,
      aggSeries: [],
      tpc: this.tokensPerChunk.get(`${req.base_url} ${model}`) ?? 1,
      calibTokens: 0,
      calibChunks: 0,
      summary: null,
      saved_run_id: null,
      error: null,
      flushTimer: null,
      aggTimer: null,
      idleTimer: null,
    };
    this.runs.set(id, run);
    this.order.push(id);
    this.evict();
    void this.execute(run);
    return id;
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
      else if (run.req.mode === "bench-decode") await this.runBenchDecode(run);
      else await this.runBenchPrefill(run);
      await this.finish(run);
    } catch (e) {
      this.fail(run, e instanceof Error ? e.message : String(e));
    }
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

  private async runBenchDecode(run: Run) {
    for (const [index, c] of run.req.levels.entries()) {
      if (run.ctrl.signal.aborted) break;
      const strands = run.strands.filter((s) => s.ref.level === index);
      await this.runWave(run, strands, "burst", 0);
      if (run.ctrl.signal.aborted) break;
      const results = strands.map(toResult);
      const ws = summarizeWave(results);
      run.arms.push({ key: { concurrency: c }, ws, results });
      this.pushLevel(run, toLevelEvent(index, { concurrency: c }, ws));
    }
  }

  private async runBenchPrefill(run: Run) {
    const sizer: PrefillSizer = { charsPerToken: 3.5 };
    const count = vllmTokenCounter(run.req.base_url, run.model, run.ctrl.signal);
    for (const [index, size] of run.req.sizes.entries()) {
      if (run.ctrl.signal.aborted) break;
      const s = run.strands[index];
      if (run.max_model_len !== null && size >= run.max_model_len) {
        const reason = `size ${size} ≥ max_model_len ${run.max_model_len}`;
        s.error = reason;
        this.setState(run, s, "cancelled");
        run.arms.push({ key: { size }, ws: null, results: [], skipped: reason });
        this.pushLevel(run, skippedLevelEvent(index, { size }, reason));
        continue;
      }
      const built = await buildPrefillPrompt(sizer, count, size, `${run.id}-${size}`);
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
      const ws = summarizeWave([result]);
      run.arms.push({ key: { size }, ws, results: [result] });
      this.pushLevel(run, toLevelEvent(index, { size }, ws));
    }
  }

  private pushLevel(run: Run, lvl: StreamLevelEvent) {
    run.levels.push(lvl);
    this.emit(run, lvl);
  }

  private bodyFor(run: Run, s: Strand): Record<string, unknown> {
    const body: Record<string, unknown> = {
      model: run.model,
      messages: [{ role: "user", content: s.promptText ?? s.ref.text }],
      max_tokens: run.max_tokens,
      temperature: run.req.temperature,
      stream: true,
      stream_options: { include_usage: true },
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
      if (!s.error && typeof s.usage?.completion_tokens === "number") this.calibrate(run, s.usage.completion_tokens, s.chunks);
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
    // Mirror perf.output_piece: content, else reasoning — either counts as output.
    const content = str(delta.content);
    const reasoning = content ? "" : str(delta.reasoning_content) || str(delta.reasoning);
    if (!content && !reasoning) return;

    const t = now();
    const first = s.t_first === null;
    if (first) s.t_first = t;
    else {
      s.itl.push(r1(t - (s.t_last ?? t)));
      if (s.itl.length > ITL_KEEP) s.itl.shift();
    }
    s.t_last = t;
    s.chunks++;
    s.token_times.push(r1(t - (s.t_start ?? t)));
    run.chunkTimes.push(t);
    s.dirty = true;
    if (first) this.setState(run, s, "decode");
    if (content) {
      s.textChunks++;
      s.pending.textChunks++;
      s.pending.text += content;
      s.text = capTail(s.text + content, TEXT_CAP);
    } else {
      s.reasoningChunks++;
      s.pending.reasoningChunks++;
      s.pending.reasoning += reasoning;
      s.reasoning = capTail(s.reasoning + reasoning, TEXT_CAP);
    }
  }

  /** Σ completion_tokens / Σ chunks over finished strands; remembered per base_url+model. */
  private calibrate(run: Run, completionTokens: number, chunks: number) {
    if (chunks <= 0 || completionTokens <= 0) return;
    run.calibTokens += completionTokens;
    run.calibChunks += chunks;
    run.tpc = run.calibTokens / run.calibChunks;
    this.tokensPerChunk.set(`${run.req.base_url} ${run.model}`, run.tpc);
  }

  private setState(run: Run, s: Strand, state: StrandState) {
    s.state = state;
    this.emit(run, this.strandEvent(run, s));
  }

  private strandEvent(run: Run, s: Strand): StreamStrandEvent {
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
      ev.itl_ms = s.itl;
    } else if (s.state === "decode") {
      ev.tokens = Math.round(s.chunks * run.tpc);
      ev.peak_tok_s = s.peak_tok_s;
    }
    return ev;
  }

  // ── Ticks ───────────────────────────────────────────────────────

  private flush(run: Run) {
    for (const s of run.strands) {
      const p = s.pending;
      if (p.reasoningChunks) {
        this.emit(run, { type: "delta", i: s.ref.i, text: p.reasoning, reasoning: true, chunks: p.reasoningChunks });
        p.reasoning = "";
        p.reasoningChunks = 0;
      }
      if (p.textChunks) {
        this.emit(run, { type: "delta", i: s.ref.i, text: p.text, reasoning: false, chunks: p.textChunks });
        p.text = "";
        p.textChunks = 0;
      }
    }
  }

  private tickAgg(run: Run) {
    const t = now();
    const cutoff = t - 1000;
    let drop = 0;
    while (drop < run.chunkTimes.length && run.chunkTimes[drop] < cutoff) drop++;
    if (drop) run.chunkTimes.splice(0, drop);
    const tok_s = r1(run.chunkTimes.length * run.tpc);
    if (tok_s > run.peak_tok_s) run.peak_tok_s = tok_s;

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
        tokens += Math.round(s.chunks * run.tpc);
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
      tokens_per_chunk: Math.round(run.tpc * 100) / 100,
    };
    const p50 = percentile(ttfts, 50);
    const p95 = percentile(ttfts, 95);
    if (p50 !== null) agg.ttft_p50_ms = r1(p50);
    if (p95 !== null) agg.ttft_p95_ms = r1(p95);
    run.aggSeries.push(agg);
    if (run.aggSeries.length > AGG_KEEP) run.aggSeries.shift();
    this.emit(run, agg);

    for (const s of run.strands) {
      if (!s.dirty || s.state !== "decode") continue;
      s.dirty = false;
      const rate = r1(strandWindowRate(s, t) * run.tpc);
      if (rate > s.peak_tok_s) s.peak_tok_s = rate;
      this.emit(run, { ...this.strandEvent(run, s), tok_s: rate });
    }
  }

  private stopTimers(run: Run) {
    if (run.flushTimer) clearInterval(run.flushTimer);
    if (run.aggTimer) clearInterval(run.aggTimer);
    if (run.idleTimer) clearTimeout(run.idleTimer);
    run.flushTimer = run.aggTimer = run.idleTimer = null;
  }

  // ── Completion ──────────────────────────────────────────────────

  private async finish(run: Run) {
    this.stopTimers(run);
    const cancelled = run.ctrl.signal.aborted;
    if (cancelled) for (const s of run.strands) if (s.state === "waiting") this.setState(run, s, "cancelled");
    this.flush(run);
    this.tickAgg(run);
    const isBench = run.req.mode !== "load";
    let headline: BenchHeadline | undefined;
    let saved: string | null = null;
    if (isBench && !cancelled) {
      const envelope = buildEnvelope({
        kind: run.req.mode === "bench-decode" ? "decode" : "prefill",
        model: run.model,
        workload: {
          pack: run.req.pack,
          ...(run.req.mode === "bench-decode" ? { levels: run.req.levels } : { sizes: run.req.sizes }),
          max_tokens: run.max_tokens,
          thinking: run.req.thinking,
          temperature: run.req.temperature,
          fill_to_max: run.fill_to_max,
          base_url: run.req.base_url,
        },
        arms: run.arms,
      });
      headline = envelope.metrics.headline;
      saved = await this.importEnvelope(envelope);
    }
    const results = run.strands.map(toResult);
    const ws = summarizeWave(results);
    run.status = cancelled ? "cancelled" : "done";
    run.ended_at = new Date().toISOString();
    run.saved_run_id = saved;
    run.summary = {
      status: run.status,
      mode: run.req.mode,
      model: run.model,
      duration_ms: Math.round(now() - run.t0),
      // Every strand's output counts here (cancelled ones too); ws.* below follow perf.py and use ok strands only.
      tokens: results.reduce((a, r) => a + (r.completion_tokens ?? 0), 0),
      peak_tok_s: run.peak_tok_s,
      aggregate_tok_s: ws.aggregate_tok_s,
      per_stream_median_tok_s: ws.per_stream_median_tok_s,
      ttft_p50_ms: ws.ttft_s.p50 === null ? null : r1(ws.ttft_s.p50 * 1000),
      ttft_p95_ms: ws.ttft_s.p95 === null ? null : r1(ws.ttft_s.p95 * 1000),
      ok: ws.ok,
      requests: ws.requests,
      errors: ws.errors,
      ...(headline ? { headline } : {}),
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
      max_tokens: run.max_tokens,
      started_at: run.started_at,
      prompts: run.strands.map((s) => s.ref),
    };
  }

  /** `hello` + snapshot (strand states, retained text, levels, last agg) then live events. */
  subscribe(id: string): Subscriber | null {
    const run = this.runs.get(id);
    if (!run) return null;
    const sub = new Subscriber();
    sub.push(this.hello(run), true);
    for (const s of run.strands) {
      if (s.state === "waiting") continue;
      sub.push(this.strandEvent(run, s), true);
      if (s.reasoning) sub.push({ type: "delta", i: s.ref.i, text: s.reasoning, reasoning: true, chunks: s.reasoningChunks }, true);
      if (s.text) sub.push({ type: "delta", i: s.ref.i, text: s.text, reasoning: false, chunks: s.textChunks }, true);
    }
    for (const lvl of run.levels) sub.push(lvl, true);
    const lastAgg = run.aggSeries[run.aggSeries.length - 1];
    if (lastAgg) sub.push(lastAgg, true);
    if (run.status !== "running") {
      if (run.status === "error" && run.error) sub.push({ type: "error", message: run.error }, true);
      sub.push({ type: "done", run_id: run.id, summary: run.summary!, saved_run_id: run.saved_run_id }, true);
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
      ...stripType(this.strandEvent(run, s)),
      text: s.text,
      reasoning_text: s.reasoning,
      chunks: s.chunks,
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
