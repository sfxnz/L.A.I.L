/**
 * Front page: Spark instruments, the served-model panel and the "Last decode bench" card (the retired
 * DecodeBench panel's replacement — the bench itself lives on /bench).
 */
import { describe, expect, test } from "bun:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ClusterPanel, flowDurS } from "../components/ClusterPanel";
import { LastSyncCard } from "../components/bench/LastSyncCard";
import { EndpointHero, hermesBases } from "../components/status/EndpointHero";
import type { ClusterNode, RunRow } from "./api";
import { lastSync, latestDecodeRuns } from "./bench/last-sync";
import { CONCURRENCY_LEVELS, PACK_LABELS, sortConcurrencies } from "./bench/levels";
import type { DecodeResult } from "./bench/result";


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

describe("Last decode bench (Status card)", () => {
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
    expect(empty).toContain("Run a decode bench to get the first curve.");
    expect(empty).toContain('href="/bench"');
    expect(empty).not.toContain("—");
    const unbenched = renderToStaticMarkup(createElement(LastSyncCard, { runs: rows, loading: false, servingModel: "org/new-model" }));
    expect(unbenched).toContain("No bench for this model yet");

    const html = renderToStaticMarkup(createElement(LastSyncCard, { runs: rows, loading: false, servingModel: MODEL }));
    expect(html).toContain("Last decode bench");
    expect(html).toContain("80.5");
    expect(html).toContain("peak aggregate @ ×4");
    expect(html).toContain("+12 % vs previous");
    expect(html).toContain('href="/bench?run=d2"');
    expect(html).toContain("Run again");
  });
});

describe("Sparks panel (per-node hardware)", () => {
  const NOW = 1_790_000_000_000;
  const serving: ClusterNode = {
    id: "spark1",
    label: "spark1",
    state: "serving",
    local: true,
    hostname: "spark1",
    cpu: "10× Cortex-X925 + 10× Cortex-A725",
    sampled_at: NOW - 400,
    temperature_c: 47,
    soc_temp_c: 51,
    nic_temp_c: 50,
    nvme_temp_c: 43,
    gpu_util_pct: 83,
    cpu_util_pct: 12,
    power_w: 32.1,
    available_gib: 13.3,
    ram_gib: 121.7,
    engine_reserved_gib: 91.9,
    swap_total_gib: 16,
    swap_used_gib: 8.1,
    mem_pressure: "ok",
    tp_rank: 0,
    rail_rates: { enp1s0f1np1: { tx_bps: 118_000_000, rx_bps: 117_000_000 } },
  };
  const worker: ClusterNode = {
    id: "spark2",
    label: "spark2",
    state: "serving_worker",
    online: true,
    hostname: "spark2",
    tp_rank: 1,
    sampled_at: NOW - 900,
    temperature_c: 42,
  };
  const link = (iface: string, extra = {}) => ({
    from: "spark1",
    to: "spark2",
    iface,
    target_ip: "10.0.0.2",
    ok: true,
    rtt_ms: 0.6,
    from_speed_mbps: 200_000,
    ...extra,
  });
  const render = (props: Parameters<typeof ClusterPanel>[0]) => renderToStaticMarkup(createElement(ClusterPanel, props));

  test("a serving Spark shows its hardware truth: temps, power, GPU util, CPU, memory split and swap", () => {
    const html = render({
      cluster: { nodes: [serving], summary: { healthy: true, nodes_online: 1, nodes_total: 1, nodes_serving: 1 } },
      serverNow: NOW,
    });
    for (const s of ["GPU temp", "47", "Power", "32.1", "GPU util", "83", "CPU", "12%", "SoC 51°C", "NIC 50°C", "NVMe 43°C"]) {
      expect(html).toContain(s);
    }
    expect(html).toContain("108.4");
    expect(html).toContain("91.9 GiB engine reservation + 16.5 GiB other used · 13.3 GiB available");
    expect(html).toContain("engine 91.9");
    expect(html).toContain("free 13.3");
    expect(html).toContain("swap 8.1 / 16");
    expect(html).toContain(">live<");
    // the endpoint rate is not on node cards (it lives once, in the served-model panel)
    expect(html).not.toContain("tok/s");
    expect(html).not.toContain("—");
  });

  test("a TP worker is labelled with its rank; the fabric shows every rail and its real RDMA traffic", () => {
    const html = render({
      cluster: {
        nodes: [serving, worker],
        fabric: { ok: true, links: [link("enp1s0f1np1"), link("enP2p1s0f1np1", { target_ip: "10.0.1.2" })] },
        summary: { healthy: true, nodes_online: 2, nodes_total: 2, nodes_serving: 2, multi: { mode: "multi_aligned", tensor_parallel_hint: 2 } },
      },
      serverNow: NOW,
    });
    expect(html).toContain("TP worker · rank 1");
    expect(html).toContain("Tensor parallel");
    // each fact once: the online count, no TP= or model message repeated from the hero
    expect(html).toContain("2/2 online");
    expect(html).not.toContain("TP=");
    expect(html).not.toContain("serving</");
    expect(html).toContain("enp1s0f1np1");
    expect(html).toContain("enP2p1s0f1np1");
    expect(html).toContain("↑ 118 MB/s");
    // the busy rail flows, the rail with no counters reading does not
    expect(html.match(/data-flowing="true"/g)?.length).toBe(1);
  });

  test("nothing flows while the data is stale, even with traffic on the last reading", () => {
    const html = render({
      cluster: { nodes: [serving, worker], fabric: { ok: true, links: [link("enp1s0f1np1")] }, summary: { nodes_online: 2, nodes_total: 2 } },
      serverNow: NOW,
      stale: true,
    });
    expect(html).not.toContain('data-flowing="true"');
  });

  test("a node whose telemetry is old is dimmed with its age, not shown as live", () => {
    const html = render({
      cluster: { nodes: [{ ...worker, sampled_at: NOW - 12_000 }], summary: { nodes_online: 1, nodes_total: 1 } },
      serverNow: NOW,
    });
    expect(html).toContain("12 s old");
    expect(html).not.toContain(">live<");
    expect(html).toContain("opacity-55");
  });

  test("ssh failure reads differently from a host that is down; neither shows numbers", () => {
    const html = render({
      cluster: {
        nodes: [
          { id: "a", label: "a", state: "unreachable", temperature_c: 40 },
          { id: "b", label: "b", state: "offline" },
        ],
        summary: { healthy: false, nodes_online: 0, nodes_total: 2 },
      },
      serverNow: NOW,
    });
    expect(html).toContain("SSH failed");
    expect(html).toContain("Offline");
    expect(html).not.toContain(">40<");
  });

  test("while the first inventory runs, the panel says it is probing — never a red 0/0", () => {
    const html = render({ cluster: { nodes: [], pending: true, summary: { healthy: false } } });
    expect(html).toContain("probing the Sparks");
    expect(html).not.toContain("0/0");
  });

  test("the fabric's flow speed moves in steps, so 1 s rate jitter never re-phases the animation", () => {
    // 120 vs 132 MB/s on a 200 Gb/s rail: same step
    expect(flowDurS(120e6, 200_000)).toBe(flowDurS(132e6, 200_000));
    expect(flowDurS(1e6, 200_000)).toBe(2.4);
    expect(flowDurS(5e9, 200_000)).toBe(0.45);
    expect(flowDurS(120e6, null)).toBe(1.2);
  });

  test("three nodes: every card renders and every link gets a row", () => {
    const n3 = { ...worker, id: "spark3", label: "spark3", tp_rank: 2 };
    const html = render({
      cluster: {
        nodes: [serving, worker, n3],
        fabric: { ok: true, links: [link("a"), { ...link("b"), to: "spark3" }, { ...link("c"), from: "spark2", to: "spark3" }] },
        summary: { nodes_online: 3, nodes_total: 3 },
      },
      serverNow: NOW,
    });
    expect(html).toContain("spark3");
    expect(html).toContain("Fabric · 3 links");
    expect(html).toContain("spark2 → spark3");
  });
});

describe("Served model panel", () => {
  const NOW = 1_790_000_000_000;
  const base = {
    healthy: true,
    model_id: "nvidia/Qwen3.8-Flash-Next-NVFP4",
    base_url: "http://127.0.0.1:8000",
    engine: { kv_usage_pct: 0.84, kv_capacity_tokens: 1_694_725, requests_running: 1, requests_waiting: 0, max_model_len: 262_144, version: "0.30.0", flags_fingerprint: "8488b677", flags: ["--port", "8000"] },
    cluster: { nodes: [], summary: { multi: { tensor_parallel_hint: 2 } } },
  };
  const render = (metrics: Record<string, unknown>, extra: Record<string, unknown> = {}) =>
    renderToStaticMarkup(
      createElement(EndpointHero, {
        serve: { ...base, metrics, ...extra } as never,
        endpoint: [],
        serverNow: NOW,
      }),
    );

  test("live: per-stream decode, throughput, TTFT, requests, spec acceptance and KV — each once", () => {
    const html = render({ decode_tok_per_s: 84.1, throughput_tok_per_s: 160.4, ttft_s: 0.174, spec_accept_rate: 0.84, spec_tokens_per_step: 3.52, prefill_tok_per_s: 402 });
    expect(html.match(/84\.1/g)?.length).toBe(1);
    expect(html).toContain("160");
    expect(html).toContain("174 ms");
    expect(html).toContain("84%");
    expect(html).toContain("3.52/step");
    expect(html).toContain("402 tok/s");
    expect(html).toContain('1 <span class="text-lab-muted">running</span>');
    // 0.84 % of the pool is 0.8 %, never 84 %
    expect(html).toContain("0.8%");
    expect(html).toContain("TP=2");
    expect(html).toContain("NVFP4");
    expect(html).toContain("http://127.0.0.1:8000/v1");
    expect(html).toContain("copy env");
  });

  test("idle: the last burst is dimmed and labelled with its age, never shown as live", () => {
    const html = render(
      { decode_tok_per_s: null, throughput_tok_per_s: 0, last_burst: { decode_tok_per_s: 83.37, tokens: 128, ended_at: NOW - 12_000 }, last_prefill: { tok_per_s: 1234, at: NOW - 95_000 }, spec_accept_rate_lifetime: 0.8432 },
      { engine: { ...base.engine, requests_running: 0 } },
    );
    expect(html).toContain("83.4");
    expect(html).toContain("last burst · 128 tok · 12 s ago");
    expect(html).toContain("last 1234 · 1 m 35 s ago");
    expect(html).toContain("84% lifetime");
    expect(html).toContain("text-lab-muted");
  });

  test("mid-decode TTFT: no request started this second, so the last one shows dimmed with its age — never 'Idle'", () => {
    const html = render({ decode_tok_per_s: 46.7, throughput_tok_per_s: 46.7, ttft_s: null, last_ttft: { s: 0.174, at: NOW - 4_000 } });
    expect(html).toContain("last 174 ms · 4 s ago");
    expect(html).not.toContain(">Idle<");
  });

  test("hermesBases follows the port that answers and adds the page host when it is not loopback", () => {
    expect(hermesBases("http://127.0.0.1:8888", "127.0.0.1")).toEqual({ local: "http://127.0.0.1:8888/v1", remote: null });
    expect(hermesBases("http://127.0.0.1:8000", "spark1.tailnet")).toEqual({
      local: "http://127.0.0.1:8000/v1",
      remote: "http://spark1.tailnet:8000/v1",
    });
    expect(hermesBases(null, "x")).toBeNull();
  });
});
