/**
 * What a bench reads besides its own streams: vLLM `/metrics` (foreign load and the
 * server's per-level view), the serve-engine status snapshot (serve fingerprint, node
 * temperature / power / memory), and the serve-engine bench lease. Every probe is
 * best-effort: a server that does not expose it yields null, never a failed run.
 */
import type { BenchHardware, BenchHardwareNode, ServerLevelMetrics } from "@lail/shared";
import { config } from "../config";

const COUNTERS = [
  "vllm:num_requests_running",
  "vllm:num_requests_waiting",
  "vllm:spec_decode_num_accepted_tokens_total",
  "vllm:spec_decode_num_draft_tokens_total",
  "vllm:spec_decode_num_drafts_total",
  "vllm:time_to_first_token_seconds_sum",
  "vllm:time_to_first_token_seconds_count",
  "vllm:request_prefill_kv_computed_tokens_sum",
  "vllm:request_prefill_time_seconds_sum",
  "vllm:num_preemptions_total",
] as const;

export type MetricSample = Partial<Record<(typeof COUNTERS)[number], number>>;

/** Prometheus text → the counters above, summed over label sets (one per engine / model). */
export function parseMetrics(text: string): MetricSample {
  const want = new Set<string>(COUNTERS);
  const out: Record<string, number> = {};
  for (const line of text.split("\n")) {
    if (!line || line.startsWith("#")) continue;
    const brace = line.indexOf("{");
    const space = line.lastIndexOf(" ");
    if (space < 0) continue;
    const name = brace >= 0 && brace < space ? line.slice(0, brace) : line.slice(0, space);
    if (!want.has(name)) continue;
    const v = Number(line.slice(space + 1));
    if (Number.isFinite(v)) out[name] = (out[name] ?? 0) + v;
  }
  return out as MetricSample;
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
  const run = m["vllm:num_requests_running"];
  const wait = m["vllm:num_requests_waiting"];
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
  const accepted = delta(before, after, "vllm:spec_decode_num_accepted_tokens_total");
  const drafted = delta(before, after, "vllm:spec_decode_num_draft_tokens_total");
  const drafts = delta(before, after, "vllm:spec_decode_num_drafts_total");
  const perStep = ratio(accepted, drafts);
  const ttft = ratio(delta(before, after, "vllm:time_to_first_token_seconds_sum"), delta(before, after, "vllm:time_to_first_token_seconds_count"));
  return {
    spec_acceptance: round(ratio(accepted, drafted), 3),
    spec_tokens_per_step: perStep === null ? null : round(1 + perStep, 2),
    ttft_mean_ms: ttft === null ? null : round(ttft * 1000, 1),
    prefill_tok_s: round(
      ratio(delta(before, after, "vllm:request_prefill_kv_computed_tokens_sum"), delta(before, after, "vllm:request_prefill_time_seconds_sum")),
      1,
    ),
    preemptions: delta(before, after, "vllm:num_preemptions_total"),
  };
}

// ── Serve-engine ───────────────────────────────────────────────────

function seHeaders(): Record<string, string> {
  const h: Record<string, string> = { "content-type": "application/json" };
  if (config.token) h["x-lail-token"] = config.token;
  return h;
}

export type NodeReading = { id: string; temp: number | null; power: number | null; avail: number | null };
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
      hardware?: Record<string, unknown>;
      cluster?: { nodes?: Array<Record<string, unknown>> };
    };
    // Cluster node telemetry refreshes on the sampler's slow (10 s) tick; the head's own
    // `hardware` block on every fast (2 s) tick — prefer it for the node it describes.
    const local = j.hardware && typeof j.hardware.hostname === "string" ? j.hardware : null;
    const nodes = (j.cluster?.nodes ?? []).map((n) => {
      const src = local && (n.hostname === local.hostname || n.id === local.hostname) ? local : n;
      return {
        id: String(n.id ?? n.hostname ?? "node"),
        temp: num(src.temperature_c),
        power: num(src.power_w),
        avail: num(src.available_gib),
      };
    });
    return { sampled_at: j.sampled_at ?? null, fingerprint: j.engine?.flags_fingerprint ?? null, nodes };
  } catch {
    return null;
  }
}

/**
 * Per node over the run, plus energy: Σ over nodes of the trapezoid ∫ power dt between
 * that node's consecutive power samples; per token over `tokens`. Energy is reported only
 * when every node's power was actually re-read during the run (≥ 2 distinct readings): a
 * run shorter than the telemetry cadence sees one stale reading repeated, and integrating
 * that would report idle power as the run's.
 */
export function summarizeHardware(series: BenchHardware["series"], tokens: number): BenchHardware {
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
  let integrated = false;
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
    const pw = rows.filter((r) => r[3] !== null);
    if (pw.length && new Set(pw.map((r) => r[3])).size < 2) stale = true;
    for (let k = 1; k < pw.length; k++) {
      energy += (((pw[k - 1][3] as number) + (pw[k][3] as number)) / 2) * ((pw[k][0] - pw[k - 1][0]) / 1000);
      integrated = true;
    }
  }
  const energy_j = integrated && !stale ? Math.round(energy) : null;
  return {
    series,
    nodes,
    energy_j,
    energy_j_per_token: energy_j !== null && tokens > 0 ? Math.round((energy_j / tokens) * 1000) / 1000 : null,
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
