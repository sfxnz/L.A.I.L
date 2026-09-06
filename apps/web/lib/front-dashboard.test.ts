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
import { decodeRunLabel, type DecodeResult } from "./bench/result";

const webRoot = join(import.meta.dir, "..");

const row = (run_id: string, created_at: string, kind: string, summary: Record<string, unknown> = {}): RunRow => ({
  run_id,
  created_at,
  kind,
  intent: null,
  model_id: "nvidia/Qwen3.8-Flash-Next-NVFP4",
  summary,
  path: `${run_id}.json`,
});
const HEADLINE = (c1: number, peak: number, at: number) => ({
  decode_tok_per_s_median_c1: c1,
  aggregate_peak_tok_per_s: peak,
  aggregate_peak_concurrency: at,
  ttft_p50_s_c1: 0.21,
  concurrencies: [1, 2, 4],
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

  test("legacy run label reads summary.workload, never perf_workflow", () => {
    expect(decodeRunLabel({ workload: "structured" })).toBe("Structured");
    expect(decodeRunLabel({ workload: "perf_workflow" })).toBeNull();
    expect(decodeRunLabel({ kind: "perf_workflow" })).toBeNull();
  });
});

describe("Last synchronization (Status card)", () => {
  const rows = [
    row("old-prefill", "2026-09-06T10:00:00Z", "prefill", { prefill_tok_per_s_sustained: 2874.9 }),
    row("d1", "2026-09-06T11:00:00Z", "decode", HEADLINE(34.2, 71.8, 4)),
    row("tool", "2026-09-06T11:30:00Z", "tool_eval"),
    row("d2", "2026-09-06T12:00:00Z", "decode", HEADLINE(34.0, 80.5, 4)),
  ];

  test("picks the newest decode run and the one before it, ignoring other kinds", () => {
    expect(latestDecodeRuns(rows).map((r) => r.run_id)).toEqual(["d2", "d1"]);
  });

  test("hero, delta vs previous and both hrefs come straight from the index rows", () => {
    const s = lastSync(rows, { current: null, previous: null });
    expect(s).not.toBeNull();
    expect(s!.id).toBe("d2");
    expect(s!.peak).toBe(80.5);
    expect(s!.peakAt).toBe(4);
    expect(s!.c1).toBe(34.0);
    // (80.5 − 71.8) / 71.8 = +12.1 %
    expect(s!.delta).toBeCloseTo(0.1212, 3);
    expect(s!.openHref).toBe("/bench?run=d2");
    expect(s!.runAgainHref).toBe("/bench?tab=decode&pack=prose&levels=1%2C2%2C4&tokens=512");
  });

  test("the envelope, when loaded, supplies pack, tokens and arms for the mini curve", () => {
    const arm = (concurrency: number, aggregate: number, perStream: number) => ({
      concurrency,
      aggregate,
      perStream,
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
      source: "history",
      levels: [1, 2, 4],
      arms: [arm(1, 34, 34), arm(2, 52, 26), arm(4, 80.5, 20.1)],
    };
    const s = lastSync(rows, { current, previous: null })!;
    expect(s.pack).toBe("code");
    expect(s.arms).toHaveLength(3);
    expect(s.previousArms).toBeNull();
    expect(s.delta).toBeCloseTo(0.1212, 3); // previous still from its index row
    expect(s.runAgainHref).toBe("/bench?tab=decode&pack=code&levels=1%2C2%2C4&tokens=128");
  });

  test("no decode runs → null; a lone run has no delta", () => {
    expect(lastSync([rows[0], rows[2]], { current: null, previous: null })).toBeNull();
    const lone = lastSync([rows[3]], { current: null, previous: null })!;
    expect(lone.delta).toBeNull();
  });

  test("renders the empty state with a Bench CTA, and the hero when a run exists", () => {
    const empty = renderToStaticMarkup(createElement(LastSyncCard, { runs: [], loading: false }));
    expect(empty).toContain("No sequences yet. Run a decode sync to draw the first slice.");
    expect(empty).toContain('href="/bench"');
    expect(empty).not.toContain("—");

    const html = renderToStaticMarkup(createElement(LastSyncCard, { runs: rows, loading: false }));
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
    hostname: "spark1",
    temperature_c: 47,
    gpu_util_pct: 83,
    power_w: 32.1,
    available_gib: 80,
    gen_tok_per_s: 41.2,
    prompt_tok_per_s: 210,
  };
  const idle: ClusterNode = {
    id: "spark2",
    label: "spark2",
    state: "idle",
    hostname: "spark2",
  };

  test("serving Spark shows temperature, usage, tok/s, and prefill", () => {
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
    expect(html).toContain("Live serve decode rate");
    expect(html).not.toContain("—");
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
