/**
 * Structural tests: front-page Spark instruments and decode-bench controls.
 */
import { describe, expect, test } from "bun:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { readFileSync } from "fs";
import { join } from "path";
import { ClusterPanel } from "../components/ClusterPanel";
import { DecodeBench } from "../components/DecodeBench";
import type { ClusterNode } from "./api";
import {
  CONCURRENCY_LEVELS,
  WORKLOAD_KINDS,
  WORKLOAD_LABELS,
  sortConcurrencies,
} from "./decode-bench";

const webRoot = join(import.meta.dir, "..");

describe("decode bench domain", () => {
  test("workload kinds are structured, prose, code, json", () => {
    expect([...WORKLOAD_KINDS]).toEqual(["structured", "prose", "code", "json"]);
    expect(WORKLOAD_LABELS.structured).toBe("Structured");
    expect(WORKLOAD_LABELS.json).toBe("JSON");
  });

  test("concurrency levels range from 1 to 32 inclusive", () => {
    expect(CONCURRENCY_LEVELS).toHaveLength(32);
    expect(CONCURRENCY_LEVELS[0]).toBe(1);
    expect(CONCURRENCY_LEVELS[31]).toBe(32);
    expect(CONCURRENCY_LEVELS).toEqual(
      Array.from({ length: 32 }, (_, i) => i + 1),
    );
  });

  test("selected concurrencies run in ascending order", () => {
    expect(sortConcurrencies(new Set([16, 1, 4]))).toEqual([1, 4, 16]);
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
    expect(panel).toContain('word = "Awaiting"');
    expect(panel).toContain('word?: "Awaiting" | "None"');
    expect(panel).toContain("<Nil");
    expect(panel).toContain("TrafficValue");
    expect(panel).not.toContain('{"—"}');
  });
});

describe("shipped front-page decode bench", () => {
  const status = readFileSync(join(webRoot, "app/status/page.tsx"), "utf8");
  const bench = readFileSync(join(webRoot, "components/DecodeBench.tsx"), "utf8");

  test("decode bench is mounted on Status, not only /evals", () => {
    expect(status).toContain("DecodeBench");
    expect(status).toContain('label="Decode bench"');
    expect(status).toContain("Token required");
    expect(status).not.toContain('href="/evals" className={btnClass("primary"');
  });

  test("workload buttons cover structured, prose, code, JSON", () => {
    const kinds = readFileSync(join(webRoot, "lib/decode-bench.ts"), "utf8");
    expect(kinds).toContain('"structured"');
    expect(kinds).toContain('"prose"');
    expect(kinds).toContain('"code"');
    expect(kinds).toContain('"json"');
    expect(bench).toContain("WORKLOAD_KINDS.map");
    expect(bench).toContain("{WORKLOAD_LABELS[k]}");
  });

  test("concurrency buttons map CONCURRENCY_LEVELS 1 through 32", () => {
    expect(bench).toContain("CONCURRENCY_LEVELS.map");
    expect(bench).toContain("Concurrency 1 to 32");
    expect(bench).toContain("workload: kind");
    expect(bench).toContain("concurrencies: levels");
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

describe("shipped DecodeBench render", () => {
  test("renders structured/prose/code/JSON and buttons 1 through 32", () => {
    const html = renderToStaticMarkup(
      createElement(DecodeBench, { healthy: true, runs: [] }),
    );
    expect(html).toContain("Structured");
    expect(html).toContain("Prose");
    expect(html).toContain("Code");
    expect(html).toContain("JSON");
    for (const n of CONCURRENCY_LEVELS) {
      expect(html).toContain(`>${n}<`);
    }
    expect(html).toContain("Run decode");
  });
});
