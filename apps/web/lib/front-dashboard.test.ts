/**
 * Front page: Spark instruments and the "Last synchronization" card (the retired
 * DecodeBench panel's replacement — the bench itself lives on /bench).
 */
import { describe, expect, test } from "bun:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { readFileSync } from "fs";
import { join } from "path";
import { ClusterPanel } from "../components/ClusterPanel";
import { LastSyncCard } from "../components/bench/LastSyncCard";
import { Nil } from "../components/ui";
import type { ClusterNode, RunRow } from "./api";
import { lastSync, latestDecodeRuns } from "./bench/last-sync";
import { CONCURRENCY_LEVELS, PACK_LABELS, sortConcurrencies } from "./bench/levels";
import type { DecodeResult } from "./bench/result";

const webRoot = join(import.meta.dir, "..");

const MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4";
const row = (run_id: string, created_at: string, kind: string, summary: Record<string, unknown> = {}, model_id = MODEL): RunRow => ({
  run_id,
  created_at,
  kind,
  intent: null,
  model_id,
  summary,
  path: `${run_id}.json`,
});
/** Index summary as the controller writes it: headline + comparability keys. */
const HEADLINE = (c1: number, peak: number, at: number, over: Record<string, unknown> = {}) => ({
  decode_tok_per_s_median_c1: c1,
  aggregate_peak_tok_per_s: peak,
  aggregate_peak_concurrency: at,
  ttft_p50_s_c1: 0.21,
  pack: "prose",
  max_tokens: 512,
  serve_fingerprint: "8488b677",
  ...over,
});

describe("bench domain kept from the retired decode bench", () => {
  test("concurrency levels 1..32 and packs structured/prose/code/json", () => {
    expect(CONCURRENCY_LEVELS).toEqual(Array.from({ length: 32 }, (_, i) => i + 1));
    expect(Object.keys(PACK_LABELS)).toEqual(expect.arrayContaining(["structured", "prose", "code", "json"]));
    expect(PACK_LABELS.json).toBe("JSON");
  });

  test("selected concurrencies run in ascending order", () => {
    expect(sortConcurrencies(new Set([16, 1, 4]))).toEqual([1, 4, 16]);
  });

});

describe("Last synchronization (Status card)", () => {
  const rows = [
    row("old-prefill", "2026-09-06T10:00:00Z", "prefill", { prefill_tok_per_s_sustained: 2874.9 }),
    row("d1", "2026-09-06T11:00:00Z", "decode", HEADLINE(34.2, 71.8, 4)),
    row("d1b", "2026-09-06T11:10:00Z", "decode", HEADLINE(30.0, 60.0, 4, { max_tokens: 256 })),
    row("d1c", "2026-09-06T11:20:00Z", "decode", HEADLINE(33.0, 99.0, 4, { serve_fingerprint: "other-flags" })),
    row("tool", "2026-09-06T11:30:00Z", "tool_eval"),
    row("other", "2026-09-06T11:40:00Z", "decode", HEADLINE(20.0, 40.0, 2), "deepseek-ai/DeepSeek-V4.1-Flash"),
    row("d2", "2026-09-06T12:00:00Z", "decode", HEADLINE(34.0, 80.5, 4)),
  ];

  test("newest decode run of the served model; previous = newest earlier comparable run", () => {
    const { cur, prev } = latestDecodeRuns(rows, MODEL);
    expect(cur?.run_id).toBe("d2");
    // d1c (other serve flags) and d1b (256 tokens) are not like for like; the other model never is
    expect(prev?.run_id).toBe("d1");
    expect(latestDecodeRuns(rows, "deepseek-ai/DeepSeek-V4.1-Flash")).toEqual({ cur: rows[5], prev: null });
    expect(latestDecodeRuns(rows, "nobody/serves-this")).toEqual({ cur: null, prev: null });
    // nothing served: the newest run of any model
    expect(latestDecodeRuns(rows, null).cur?.run_id).toBe("d2");
  });

  test("a run without comparability keys (before they were recorded) has no delta", () => {
    const legacy = [row("a", "2026-09-01T00:00:00Z", "decode", { aggregate_peak_tok_per_s: 50 }), row("b", "2026-09-02T00:00:00Z", "decode", { aggregate_peak_tok_per_s: 60 })];
    expect(lastSync(legacy, MODEL, { current: null, previous: null })!.delta).toBeNull();
  });

  test("hero, delta vs previous and both hrefs come straight from the index rows", () => {
    const s = lastSync(rows, MODEL, { current: null, previous: null });
    expect(s).not.toBeNull();
    expect(s!.id).toBe("d2");
    expect(s!.peak).toBe(80.5);
    expect(s!.peakAt).toBe(4);
    expect(s!.c1).toBe(34.0);
    // (80.5 − 71.8) / 71.8 = +12.1 %
    expect(s!.delta).toBeCloseTo(0.1212, 3);
    expect(s!.openHref).toBe("/bench?run=d2");
    expect(s!.comparedTo).toBe("2026-09-06T11:00:00Z");
    expect(s!.runAgainHref).toBe("/bench?tab=decode&pack=prose&levels=1%2C2%2C4%2C8&tokens=512");
  });

  test("the envelope, when loaded, supplies pack, tokens and arms for the mini curve", () => {
    const arm = (concurrency: number, aggregate: number, perStream: number) => ({
      samples: 1,
      foreignMax: 0,
      server: null,
      concurrency,
      aggregate,
      steady: aggregate,
      aggregateRange: null,
      perStream,
      perStreamRange: null,
      ttftP50: 200,
      ttftP95: 300,
      ttftP99: 350,
      tpotMs: 1000 / perStream,
      ok: concurrency,
      requests: concurrency,
      errors: [],
    });
    const current: DecodeResult = {
      kind: "decode",
      id: "d2",
      savedRunId: "d2",
      model: "nvidia/Qwen3.8-Flash-Next-NVFP4",
      pack: "code",
      maxTokens: 128,
      createdAt: "2026-09-06T12:00:00Z",
      durationMs: 16000,
      engine: "vllm",
      fingerprint: "8488b677",
      hardware: null,
      source: "history",
      levels: [1, 2, 4],
      arms: [arm(1, 34, 34), arm(2, 52, 26), arm(4, 80.5, 20.1)],
    };
    const s = lastSync(rows, MODEL, { current, previous: null })!;
    expect(s.pack).toBe("code");
    expect(s.arms).toHaveLength(3);
    expect(s.previousArms).toBeNull();
    expect(s.delta).toBeCloseTo(0.1212, 3); // previous still from its index row
    expect(s.runAgainHref).toBe("/bench?tab=decode&pack=code&levels=1%2C2%2C4&tokens=128");
  });

  test("no decode runs → null; a lone run has no delta", () => {
    expect(lastSync([rows[0], rows[4]], MODEL, { current: null, previous: null })).toBeNull();
    const lone = lastSync([rows[6]], MODEL, { current: null, previous: null })!;
    expect(lone.delta).toBeNull();
  });

  test("renders the empty state with a Bench CTA, and the hero when a run exists", () => {
    const empty = renderToStaticMarkup(createElement(LastSyncCard, { runs: [], loading: false, servingModel: null }));
    expect(empty).toContain("Run a decode sync to draw the first slice.");
    expect(empty).toContain('href="/bench"');
    expect(empty).not.toContain("—");
    const unbenched = renderToStaticMarkup(createElement(LastSyncCard, { runs: rows, loading: false, servingModel: "org/new-model" }));
    expect(unbenched).toContain("No bench for this model yet");

    const html = renderToStaticMarkup(createElement(LastSyncCard, { runs: rows, loading: false, servingModel: MODEL }));
    expect(html).toContain("Last synchronization");
    expect(html).toContain("80.5");
    expect(html).toContain("peak aggregate @ ×4");
    expect(html).toContain("+12 % vs previous");
    expect(html).toContain('href="/bench?run=d2"');
    expect(html).toContain("Run again");
  });
});

describe("shipped Spark card metric slots", () => {
  const panel = readFileSync(join(webRoot, "components/ClusterPanel.tsx"), "utf8");

  test("shows temperature, usage, tok/s, and prefill slots", () => {
    expect(panel).toContain('label="Temperature"');
    expect(panel).toContain('label="Usage"');
    expect(panel).toContain('label="tok/s"');
    expect(panel).toContain('label="Prefill"');
  });

  test("keeps AWAITING / NONE nil treatment for missing readings", () => {
    // The shared <Nil/> renders the state word behind a hollow diamond.
    expect(renderToStaticMarkup(createElement(Nil))).toContain("Awaiting");
    expect(renderToStaticMarkup(createElement(Nil, { word: "None" }))).toContain("None");
    expect(panel).toContain("<Nil");
    expect(panel).toContain("TrafficValue");
    expect(panel).not.toContain('{"—"}');
  });
});

describe("shipped ClusterPanel render", () => {
  const serving: ClusterNode = {
    id: "spark1",
    label: "spark1",
    state: "serving",
    local: true,
    hostname: "spark1",
    temperature_c: 47,
    gpu_util_pct: 83,
    power_w: 32.1,
    available_gib: 13.3,
    ram_gib: 121.7,
    engine_reserved_gib: 91.9,
    swap_total_gib: 16,
    swap_used_gib: 8.1,
    mem_pressure: "ok",
    tp_rank: 0,
  };
  const worker: ClusterNode = {
    id: "spark2",
    label: "spark2",
    state: "serving_worker",
    hostname: "spark2",
    tp_rank: 1,
    temperature_c: 42,
  };
  const idle: ClusterNode = {
    id: "spark2",
    label: "spark2",
    state: "idle",
    hostname: "spark2",
  };

  test("serving Spark shows temperature, usage, the endpoint tok/s, prefill, memory and swap", () => {
    const html = renderToStaticMarkup(
      createElement(ClusterPanel, {
        cluster: {
          nodes: [serving],
          summary: {
            healthy: true,
            nodes_online: 1,
            nodes_total: 1,
            nodes_serving: 1,
          },
        },
        metrics: { decode_tok_per_s: 41.2, throughput_tok_per_s: 80.4, prefill_tok_per_s: 210 },
        engine: { kv_usage_pct: 0.84, kv_capacity_tokens: 1_694_725 },
      }),
    );
    expect(html).toContain("Temperature");
    expect(html).toContain("47°C");
    expect(html).toContain("Usage");
    expect(html).toContain("83%");
    expect(html).toContain("tok/s");
    expect(html).toContain("41.2");
    expect(html).toContain("Prefill");
    expect(html).toContain("210");
    expect(html).toContain("Per-stream decode rate");
    expect(html).toContain("all streams 80.4 tok/s");
    // 0.84 % of the pool is 0.8 %, never 84 %
    expect(html).toContain("KV 0.8%");
    expect(html).not.toContain("KV 84");
    expect(html).toContain("108.4");
    expect(html).toContain("91.9 GiB engine reservation + 16.5 GiB other used · 13.3 GiB available");
    expect(html).toContain("swap 8.1");
    expect(html).not.toContain("—");
  });

  test("a TP worker shows its rank, not a second copy of the endpoint rate", () => {
    const html = renderToStaticMarkup(
      createElement(ClusterPanel, {
        cluster: {
          nodes: [serving, worker],
          summary: { healthy: true, nodes_online: 2, nodes_total: 2, nodes_serving: 2 },
        },
        metrics: { decode_tok_per_s: 41.2 },
      }),
    );
    expect(html.match(/41\.2/g)?.length).toBe(2); // head hero + load strip, not the worker
    expect(html).toContain("TP rank 1");
  });

  test("between requests the last prefill is dimmed and labelled 'last', never shown as live", () => {
    const html = renderToStaticMarkup(
      createElement(ClusterPanel, {
        cluster: {
          nodes: [serving],
          summary: { healthy: true, nodes_online: 1, nodes_total: 1, nodes_serving: 1 },
        },
        metrics: { decode_tok_per_s: null, prefill_tok_per_s: null, last_prefill: { tok_per_s: 1234, at: Date.now() - 95_000 } },
      }),
    );
    expect(html).toContain("last 1234");
    expect(html).toContain("1 m 35 s ago — not live");
    expect(html).not.toMatch(/text-lab-ok">1234</);
  });

  test("a remote node serving on its own never shows the local endpoint's rates or KV", () => {
    const remote: ClusterNode = { ...serving, id: "spark2", label: "spark2", hostname: "spark2", local: false };
    const html = renderToStaticMarkup(
      createElement(ClusterPanel, {
        cluster: {
          nodes: [{ ...serving, state: "idle" }, remote],
          summary: { healthy: true, nodes_online: 2, nodes_total: 2, nodes_serving: 1 },
        },
        metrics: { decode_tok_per_s: 41.2, prefill_tok_per_s: 210 },
        engine: { kv_usage_pct: 12, kv_capacity_tokens: 1_694_725 },
      }),
    );
    expect(html).not.toContain("41.2");
    expect(html).not.toContain("210");
    expect(html).not.toContain("KV 12%");
  });

  test("usage tooltip leaves out a GPU reading it does not have — never 'GPU 0%'", () => {
    const html = renderToStaticMarkup(
      createElement(ClusterPanel, {
        cluster: {
          nodes: [{ ...serving, gpu_util_pct: null, cpu_util_pct: 37, cpu: "Cortex-X925" }],
          summary: { healthy: true, nodes_online: 1, nodes_total: 1, nodes_serving: 1 },
        },
      }),
    );
    expect(html).toContain('title="CPU 37% (Cortex-X925)"');
    expect(html).not.toContain("GPU 0%");
  });

  test("an ssh failure on a pinging host reads differently from a host that is down", () => {
    const html = renderToStaticMarkup(
      createElement(ClusterPanel, {
        cluster: {
          nodes: [
            { id: "a", label: "a", state: "unreachable" },
            { id: "b", label: "b", state: "offline" },
          ],
          summary: { healthy: false, nodes_online: 0, nodes_total: 2 },
        },
      }),
    );
    expect(html).toContain("SSH failed");
    expect(html).toContain("Offline");
  });

  test("idle Spark keeps tok/s and prefill as None, not zero", () => {
    const html = renderToStaticMarkup(
      createElement(ClusterPanel, {
        cluster: {
          nodes: [idle],
          summary: {
            healthy: true,
            nodes_online: 1,
            nodes_total: 1,
            nodes_serving: 0,
          },
        },
      }),
    );
    expect(html).toContain("tok/s");
    expect(html).toContain("Prefill");
    expect(html).toContain("None");
    expect(html).not.toContain("0 tok");
    expect(html).not.toContain(">0%<");
  });
});
