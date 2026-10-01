/**
 * What a bench reads besides its own streams: the engine's `/metrics` (foreign load and
 * the server's per-level view — vLLM, SGLang, llama.cpp or TensorFold names, mapped onto
 * one set of keys), the serve-engine status snapshot (serve fingerprint, node temperature
 * / power / memory), and the serve-engine bench lease. Every probe is best-effort: a
 * server that does not expose it yields null, never a failed run.
 */
import type { BenchHardware, BenchHardwareNode, ServerLevelMetrics } from "@lail/shared";
import { config } from "../config";

/** Engine series → bench key. Each engine exposes at most one name per key. */
const SERIES: Record<string, keyof MetricSample> = {
  "vllm:num_requests_running": "running",
  "sglang:num_running_reqs": "running",
  "llamacpp:requests_processing": "running",
  "tensorfold:requests_running": "running",
  "vllm:num_requests_waiting": "waiting",
  "sglang:num_queue_reqs": "waiting",
  "llamacpp:requests_deferred": "waiting",
  "tensorfold:requests_waiting": "waiting",
  "vllm:spec_decode_num_accepted_tokens_total": "spec_accepted",
  "llamacpp:spec_decode_num_accepted_tokens_total": "spec_accepted",
  "tensorfold:mtp_accepted_total": "spec_accepted",
  "vllm:spec_decode_num_draft_tokens_total": "spec_drafted",
  "llamacpp:spec_decode_num_draft_tokens_total": "spec_drafted",
  "tensorfold:mtp_drafted_total": "spec_drafted",
  "vllm:spec_decode_num_drafts_total": "spec_drafts",
  "llamacpp:spec_decode_num_drafts_total": "spec_drafts",
  "vllm:time_to_first_token_seconds_sum": "ttft_sum",
  "sglang:time_to_first_token_seconds_sum": "ttft_sum",
  "tensorfold:time_to_first_token_seconds_sum": "ttft_sum",
  "vllm:time_to_first_token_seconds_count": "ttft_count",
  "sglang:time_to_first_token_seconds_count": "ttft_count",
  "tensorfold:time_to_first_token_seconds_count": "ttft_count",
  // Computed (uncached) prompt tokens and the time spent on them.
  "vllm:request_prefill_kv_computed_tokens_sum": "prefill_tokens",
  "llamacpp:prompt_tokens_total": "prefill_tokens",
  "vllm:request_prefill_time_seconds_sum": "prefill_seconds",
  "llamacpp:prompt_seconds_total": "prefill_seconds",
  "vllm:num_preemptions_total": "preemptions",
};

export type MetricSample = Partial<
  Record<
    | "running"
    | "waiting"
    | "spec_accepted"
    | "spec_drafted"
    | "spec_drafts"
    | "ttft_sum"
    | "ttft_count"
    | "prefill_tokens"
    | "prefill_seconds"
    | "preemptions",
    number
  >
>;

/** Prometheus text → the bench keys, summed over label sets (one per engine / model / is_streaming). */
export function parseMetrics(text: string): MetricSample {
  const out: MetricSample = {};
  for (const line of text.split("\n")) {
    if (!line || line.startsWith("#")) continue;
    const brace = line.indexOf("{");
    const space = line.lastIndexOf(" ");
    if (space < 0) continue;
    const name = brace >= 0 && brace < space ? line.slice(0, brace) : line.slice(0, space);
    const key = SERIES[name];
    if (!key) continue;
    const v = Number(line.slice(space + 1));
    if (Number.isFinite(v)) out[key] = (out[key] ?? 0) + v;
  }
  return out;
}

export async function scrapeMetrics(baseUrl: string, signal?: AbortSignal): Promise<MetricSample | null> {
  try {
    const r = await fetch(`${baseUrl}/metrics`, { signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(3000)]) : AbortSignal.timeout(3000) });
    if (!r.ok) return null;
    const m = parseMetrics(await r.text());
    return Object.keys(m).length ? m : null;
  } catch {
    return null;
  }
}

/** Requests the server holds (running + waiting); null when it does not report them. */
export function serverLoad(m: MetricSample | null): number | null {
  if (!m) return null;
  const run = m.running;
  const wait = m.waiting;
  return run === undefined && wait === undefined ? null : (run ?? 0) + (wait ?? 0);
}

const delta = (a: MetricSample, b: MetricSample, k: keyof MetricSample): number | null => {
  const x = a[k];
  const y = b[k];
  return x === undefined || y === undefined ? null : y - x;
};
const ratio = (n: number | null, d: number | null): number | null => (n === null || d === null || d <= 0 ? null : n / d);
const round = (v: number | null, digits: number) => (v === null ? null : Math.round(v * 10 ** digits) / 10 ** digits);

/** Server-side deltas between two scrapes bracketing a level. */
export function serverDelta(before: MetricSample | null, after: MetricSample | null): ServerLevelMetrics | null {
  if (!before || !after) return null;
  const accepted = delta(before, after, "spec_accepted");
  const drafted = delta(before, after, "spec_drafted");
  const drafts = delta(before, after, "spec_drafts");
  const perStep = ratio(accepted, drafts);
  const ttft = ratio(delta(before, after, "ttft_sum"), delta(before, after, "ttft_count"));
  return {
    spec_acceptance: round(ratio(accepted, drafted), 3),
    spec_tokens_per_step: perStep === null ? null : round(1 + perStep, 2),
    ttft_mean_ms: ttft === null ? null : round(ttft * 1000, 1),
    prefill_tok_s: round(
      ratio(delta(before, after, "prefill_tokens"), delta(before, after, "prefill_seconds")),
      1,
    ),
    preemptions: delta(before, after, "preemptions"),
  };
}

// ── Serve-engine ───────────────────────────────────────────────────

function seHeaders(): Record<string, string> {
  const h: Record<string, string> = { "content-type": "application/json" };
  if (config.token) h["x-lail-token"] = config.token;
  return h;
}

/** One node's live reading; `at` is its own `sampled_at` (epoch ms), null when the node has no current reading. */
export type NodeReading = { id: string; at: number | null; temp: number | null; power: number | null; avail: number | null };
export type StatusReading = { sampled_at: string | null; fingerprint: string | null; nodes: NodeReading[] };

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** The serve-engine's cached status snapshot (the one sampler; never a fresh probe). */
export async function readServeStatus(signal?: AbortSignal): Promise<StatusReading | null> {
  try {
    const r = await fetch(`${config.serveEngineUrl}/api/status`, {
      headers: seHeaders(),
      signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(3000)]) : AbortSignal.timeout(3000),
    });
    if (!r.ok) return null;
    const j = (await r.json()) as {
      sampled_at?: string;
      engine?: { flags_fingerprint?: string | null };
      cluster?: { nodes?: Array<Record<string, unknown>> };
    };
    // Every cluster node carries its freshest telemetry (the head from the fast tick, peers
    // from their streams) with its own `sampled_at`; a down node has none.
    const nodes = (j.cluster?.nodes ?? []).map((n) => ({
      id: String(n.id ?? n.hostname ?? "node"),
      at: num(n.sampled_at),
      temp: num(n.temperature_c),
      power: num(n.power_w),
      avail: num(n.available_gib),
    }));
    return { sampled_at: j.sampled_at ?? null, fingerprint: j.engine?.flags_fingerprint ?? null, nodes };
  } catch {
    return null;
  }
}

/** Power at `t`: readings joined linearly, held flat before the first / after the last. */
function powerAt(pw: Array<[number, number]>, t: number): number {
  if (t <= pw[0][0]) return pw[0][1];
  for (let k = 1; k < pw.length; k++) {
    const [t1, p1] = pw[k];
    if (t <= t1) {
      const [t0, p0] = pw[k - 1];
      return t1 === t0 ? p1 : p0 + ((p1 - p0) * (t - t0)) / (t1 - t0);
    }
  }
  return pw[pw.length - 1][1];
}

/** ∫ power dt (J) over `windows` (ms since run start) — exact trapezoids on the joined readings. */
function energyOver(pw: Array<[number, number]>, windows: Array<[number, number]>): number {
  let j = 0;
  for (const [a, b] of windows) {
    if (b <= a) continue;
    const ts = [a, ...pw.map((r) => r[0]).filter((t) => t > a && t < b), b];
    for (let k = 1; k < ts.length; k++) j += ((powerAt(pw, ts[k - 1]) + powerAt(pw, ts[k])) / 2) * ((ts[k] - ts[k - 1]) / 1000);
  }
  return j;
}

/**
 * Per node over the run, plus energy: Σ over nodes of ∫ power dt over the measured
 * `windows` only (the level spans — not the warmup or the idle-drain waits), so energy and
 * `tokens` cover the same time; per token over `tokens` (null = not a per-token quantity,
 * e.g. the prefill bench's 1-token answers). `series` holds only new readings (a node's
 * `sampled_at` moved). Energy is reported only when every node's power was re-read during
 * the run (≥ 2 readings): integrating one reading held flat would report the power from
 * before the run as the run's.
 */
export function summarizeHardware(
  series: BenchHardware["series"],
  windows: Array<[number, number]>,
  tokens: number | null,
): BenchHardware {
  const byNode = new Map<string, BenchHardware["series"]>();
  for (const s of series) {
    const list = byNode.get(s[1]) ?? [];
    list.push(s);
    byNode.set(s[1], list);
  }
  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
  const r1 = (v: number | null) => (v === null ? null : Math.round(v * 10) / 10);
  const nodes: BenchHardwareNode[] = [];
  let energy = 0;
  let stale = false;
  for (const [id, rows] of byNode) {
    const temps = rows.map((r) => r[2]).filter((v): v is number => v !== null);
    const powers = rows.map((r) => r[3]).filter((v): v is number => v !== null);
    const avails = rows.map((r) => r[4]).filter((v): v is number => v !== null);
    nodes.push({
      id,
      samples: rows.length,
      temp_max_c: temps.length ? Math.max(...temps) : null,
      temp_mean_c: r1(mean(temps)),
      power_mean_w: r1(mean(powers)),
      available_min_gib: avails.length ? Math.min(...avails) : null,
    });
    const pw = rows.filter((r) => r[3] !== null).map((r): [number, number] => [r[0], r[3] as number]);
    if (pw.length < 2) stale = true;
    else energy += energyOver(pw, windows);
  }
  const measured = windows.some(([a, b]) => b > a);
  const energy_j = byNode.size && measured && !stale ? Math.round(energy) : null;
  return {
    series,
    nodes,
    energy_j,
    energy_j_per_token: energy_j !== null && tokens !== null && tokens > 0 ? Math.round((energy_j / tokens) * 1000) / 1000 : null,
  };
}

export type LeaseResult = { ok: true } | { ok: false; holder: Record<string, unknown> } | { ok: null };

/** Take / renew the serve-engine bench lease. `ok: null` = serve-engine unreachable (run without it). */
export async function acquireBenchLease(leaseId: string, ttlS: number): Promise<LeaseResult> {
  try {
    const r = await fetch(`${config.serveEngineUrl}/api/bench/lease`, {
      method: "POST",
      headers: seHeaders(),
      body: JSON.stringify({ lease_id: leaseId, owner: "controller", ttl_s: ttlS }),
      signal: AbortSignal.timeout(3000),
    });
    if (r.status === 409) {
      const j = (await r.json().catch(() => ({}))) as { detail?: Record<string, unknown> };
      return { ok: false, holder: j.detail ?? {} };
    }
    return r.ok ? { ok: true } : { ok: null };
  } catch {
    return { ok: null };
  }
}

export async function releaseBenchLease(leaseId: string): Promise<void> {
  try {
    await fetch(`${config.serveEngineUrl}/api/bench/lease/${encodeURIComponent(leaseId)}`, {
      method: "DELETE",
      headers: seHeaders(),
      signal: AbortSignal.timeout(3000),
    });
  } catch {
    /* the lease expires on its own */
  }
}
