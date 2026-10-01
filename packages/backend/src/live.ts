import { config } from "./config";
import { getSettings, openAiBase } from "./controller/settings";

/**
 * The ONE live data path for every open dashboard tab.
 *
 * GET /api/live is a server-sent event stream. While anyone is subscribed (and
 * for LINGER_MS after the last one leaves), a single loop long-polls serve-engine (`/api/status?after=<sampled_at_ms>&wait=`),
 * which answers the moment the sampler publishes its next 1 s snapshot, and fans
 * each one out to every subscriber. N tabs cost one upstream request per sample,
 * not N polls on a beat against the sampler.
 *
 * Events (data is one line of JSON):
 *   meta     static and slow data — controller settings, backend reachability,
 *            served model list, engine version and flags. Sent on connect and
 *            only when it changes.
 *   history  on connect: compact snapshots of the last ~60 s, so a freshly
 *            opened page draws full sparklines at once.
 *   tick     every published snapshot, minus what `meta` carries.
 * A comment line every PING_MS keeps proxies and idle timers from closing it.
 *
 * GET /api/lab-status stays for curl and the browser's polling fallback: the same
 * object, assembled from one upstream read.
 */

export type Backends = Record<string, { ok: boolean; url: string; error?: string }>;

type Json = Record<string, unknown>;

export type LabStatusBody = {
  controller: "ok";
  defaultBackend: string;
  defaultModel: string;
  openAiBase: string;
  backends: Backends;
  serve: Json;
};

/** How long a backend reachability probe is reused (every tab, every route). */
export const BACKENDS_TTL_MS = 5000;
/** Long-poll bound handed to serve-engine (its own cap is 10 s). */
export const WAIT_S = 5;
/** Upstream budget for one call: the long poll plus a normal request's slack. */
const FETCH_SLACK_MS = 3000;
export const PING_MS = 5000;
/** History kept for new subscribers: a little over the 60 s every sparkline draws. */
export const HISTORY_MS = 65_000;
/** The cached last tick is handed to a new subscriber only if it is this fresh. */
const REPLAY_MS = 3000;
/**
 * After the last tab closes the loop keeps sampling this long, so a reload or the
 * next tab opened gets full sparklines from the history at once.
 */
export const LINGER_MS = 10 * 60_000;
/** A subscriber this far behind (bytes queued) is dropped rather than buffered forever. */
const MAX_QUEUED_BYTES = 1 << 20;

let backendsCache: { at: number; value: Backends } | null = null;
let backendsInflight: Promise<Backends> | null = null;

/** Enabled backends' /v1/models reachability, probed at most once per BACKENDS_TTL_MS. */
export function probeBackends(now = Date.now()): Promise<Backends> {
  if (backendsCache && now - backendsCache.at < BACKENDS_TTL_MS) return Promise.resolve(backendsCache.value);
  if (backendsInflight) return backendsInflight;
  const settings = getSettings();
  const out: Backends = {};
  backendsInflight = Promise.all(
    Object.entries(settings.backends)
      .filter(([, v]) => v.enabled)
      .map(async ([k, v]) => {
        try {
          const base = v.url.replace(/\/$/, "").replace(/\/v1$/, "");
          const r = await fetch(`${base}/v1/models`, { signal: AbortSignal.timeout(2000) });
          out[k] = { ok: r.ok, url: v.url };
        } catch (e) {
          out[k] = { ok: false, url: v.url, error: e instanceof Error ? e.message : String(e) };
        }
      }),
  ).then(() => {
    backendsCache = { at: Date.now(), value: out };
    backendsInflight = null;
    return out;
  });
  return backendsInflight;
}

export function resetBackendsCache(): void {
  backendsCache = null;
  backendsInflight = null;
}

/** serve-engine /api/status; `after` + `waitS` make it a long poll. Never throws. */
export async function fetchServe(after?: number | null, waitS = 0, signal?: AbortSignal): Promise<Json> {
  const q = after != null && waitS > 0 ? `?after=${after}&wait=${waitS}` : "";
  const timeout = AbortSignal.timeout(waitS * 1000 + FETCH_SLACK_MS);
  try {
    const r = await fetch(`${config.serveEngineUrl}/api/status${q}`, {
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
      headers: config.token ? { "x-lail-token": config.token } : undefined,
    });
    if (r.ok) return (await r.json()) as Json;
    return { error: `serve-engine ${r.status}` };
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e), unreachable: true };
  }
}

export function labStatusOf(serve: Json, backends: Backends): LabStatusBody {
  const settings = getSettings();
  return {
    controller: "ok",
    defaultBackend: settings.defaultBackend,
    defaultModel: settings.defaultModel,
    openAiBase: openAiBase(),
    backends,
    serve,
  };
}

/** The static/slow part of a status (the `meta` event) and the per-sample rest (`tick`). */
export function splitStatus(s: LabStatusBody): { meta: Json; tick: Json } {
  const { models, version, engine, ...serve } = s.serve as Json & { engine?: Json | null };
  const { flags, ...liveEngine } = engine ?? {};
  return {
    meta: {
      controller: s.controller,
      defaultBackend: s.defaultBackend,
      defaultModel: s.defaultModel,
      openAiBase: s.openAiBase,
      backends: s.backends,
      models: models ?? null,
      version: version ?? null,
      flags: flags ?? null,
    },
    tick: { serve: engine ? { ...serve, engine: liveEngine } : serve },
  };
}

type Node = Json & { id?: string };

/** What a sparkline needs from one snapshot, in the same shape (a partial status). */
export function compactStatus(serve: Json): Json {
  const metrics = (serve.metrics ?? {}) as Json;
  const nodes = (((serve.cluster ?? {}) as Json).nodes ?? []) as Node[];
  return {
    serve: {
      sampled_at_ms: serve.sampled_at_ms,
      metrics: {
        sampled_at: metrics.sampled_at,
        decode_tok_per_s: metrics.decode_tok_per_s,
        throughput_tok_per_s: metrics.throughput_tok_per_s,
      },
      cluster: {
        nodes: nodes.map((n) => ({
          id: n.id,
          local: n.local,
          online: n.online,
          sampled_at: n.sampled_at,
          temperature_c: n.temperature_c,
          gpu_util_pct: n.gpu_util_pct,
          power_w: n.power_w,
          cpu_util_pct: n.cpu_util_pct,
          ram_gib: n.ram_gib,
          available_gib: n.available_gib,
          swap_used_gib: n.swap_used_gib,
          rail_rates: n.rail_rates,
        })),
      },
    },
  };
}

export type Subscriber = (event: string, data: string) => void;

export class LiveHub {
  private subs = new Set<Subscriber>();
  private meta: string | null = null;
  private tick: { data: string; recv: number } | null = null;
  /** `recv` is this process's clock: pruning never compares two hosts' clocks. */
  private history: Array<{ recv: number; data: Json }> = [];
  private abort: AbortController | null = null;
  private running = false;
  private lingerUntil = 0;

  constructor(
    private readonly fetchStatus: typeof fetchServe = fetchServe,
    private readonly backends: () => Promise<Backends> = () => probeBackends(),
    private readonly lingerMs = LINGER_MS,
  ) {}

  private get wanted(): boolean {
    return this.subs.size > 0 || Date.now() < this.lingerUntil;
  }

  get size(): number {
    return this.subs.size;
  }

  /** Adds a subscriber (it gets meta, history and the latest tick at once). Returns unsubscribe. */
  subscribe(send: Subscriber): () => void {
    this.subs.add(send);
    // Replayed only while current: after an idle spell (no subscribers, no loop) the
    // cached sample is old, and a new tab must not paint it as live.
    const now = Date.now();
    this.history = this.history.filter((h) => h.recv >= now - HISTORY_MS);
    if (this.meta) send("meta", this.meta);
    if (this.history.length) send("history", JSON.stringify(this.history.map((h) => h.data)));
    if (this.tick && now - this.tick.recv < REPLAY_MS) send("tick", this.tick.data);
    if (!this.running) void this.loop();
    return () => {
      this.subs.delete(send);
      if (this.subs.size) return;
      this.lingerUntil = Date.now() + this.lingerMs;
      if (!this.lingerMs) this.abort?.abort();
    };
  }

  private broadcast(event: string, data: string): void {
    for (const send of [...this.subs]) {
      try {
        send(event, data);
      } catch {
        this.subs.delete(send);
      }
    }
  }

  /** One upstream long poll at a time, for as long as anyone listens. */
  private async loop(): Promise<void> {
    this.running = true;
    let after: number | null = null;
    let backoff = 1000;
    try {
      while (this.wanted) {
        this.abort = new AbortController();
        const t0 = Date.now();
        const serve = await this.fetchStatus(after, WAIT_S, this.abort.signal);
        if (!this.wanted) break;
        const at = typeof serve.sampled_at_ms === "number" ? serve.sampled_at_ms : null;
        if (at != null && at === after && !serve.error) {
          // The long poll timed out with the snapshot we already sent: nothing new.
          // An engine that answers at once instead of waiting is polled at 4 Hz, not spun on.
          if (Date.now() - t0 < 200) await new Promise((r) => setTimeout(r, 250));
          continue;
        }
        const status = labStatusOf(serve, await this.backends());
        const { meta, tick } = splitStatus(status);
        const metaJson = JSON.stringify(meta);
        if (metaJson !== this.meta) {
          this.meta = metaJson;
          this.broadcast("meta", metaJson);
        }
        const recv = Date.now();
        this.tick = { data: JSON.stringify(tick), recv };
        this.broadcast("tick", this.tick.data);
        if (at != null) {
          this.history.push({ recv, data: compactStatus(serve) });
          while (this.history.length && this.history[0].recv < recv - HISTORY_MS) this.history.shift();
        }
        if (serve.error && at == null) {
          // serve-engine down (or warming up with no snapshot): retry with backoff.
          await new Promise((r) => setTimeout(r, backoff));
          backoff = Math.min(backoff * 2, 5000);
        } else {
          backoff = 1000;
          after = at;
        }
      }
    } finally {
      this.running = false;
      this.abort = null;
      // A subscriber that arrived while the loop was winding down needs a loop.
      if (this.wanted) void this.loop();
    }
  }
}

export const liveHub = new LiveHub();

/** The SSE response for one subscriber of `hub`. Closes when the browser goes away. */
export function liveResponse(hub: LiveHub, signal: AbortSignal): Response {
  const enc = new TextEncoder();
  let cleanup = () => {};
  const body = new ReadableStream<Uint8Array>({
    start(ctrl) {
      let closed = false;
      const close = () => {
        if (closed) return;
        closed = true;
        cleanup();
        try {
          ctrl.close();
        } catch {
          /* already closed */
        }
      };
      const write = (chunk: string) => {
        if (closed) return;
        if ((ctrl.desiredSize ?? 0) < -MAX_QUEUED_BYTES) {
          close(); // a reader this far behind is gone or stuck; it will reconnect
          return;
        }
        ctrl.enqueue(enc.encode(chunk));
      };
      write(`retry: 2000\n\n`);
      const unsubscribe = hub.subscribe((event, data) => write(`event: ${event}\ndata: ${data}\n\n`));
      const ping = setInterval(() => write(`: ping\n\n`), PING_MS);
      cleanup = () => {
        clearInterval(ping);
        unsubscribe();
      };
      if (signal.aborted) close();
      else signal.addEventListener("abort", close, { once: true });
    },
    cancel() {
      cleanup();
    },
  });
  return new Response(body, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      // Never compressed or buffered by a proxy in front (see routes/serve-proxy.ts).
      "content-encoding": "identity",
      "x-accel-buffering": "no",
    },
  });
}
