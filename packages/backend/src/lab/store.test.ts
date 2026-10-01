import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { config } from "../config";
import { listLabRuns, listLabRunsByFingerprint } from "./store";

let dir: string;
let prev: string;

beforeAll(() => {
  prev = config.dataDir;
  dir = mkdtempSync(join(tmpdir(), "lail-lab-"));
  config.dataDir = dir;
  const root = join(dir, "lab-runs");
  // 30 newer "page" runs on top of 2 older "chart" runs, as a busy gallery looks
  for (let k = 0; k < 32; k++) {
    const id = `20260901T0000${String(k).padStart(2, "0")}Z_aaaaaa`;
    const chart = k < 2;
    mkdirSync(join(root, id), { recursive: true });
    writeFileSync(
      join(root, id, "meta.json"),
      JSON.stringify({
        id,
        kind: "hermes_task",
        task_type: chart ? "chart" : "page",
        title: id,
        model_id: chart ? "org/old-model" : "org/new-model",
        created_at: id,
        entry: "index.html",
        share: { public: false, slug: null },
        tags: [],
        task_fingerprint: chart ? "fp-chart" : "fp-page",
      }),
    );
  }
});

afterAll(() => {
  config.dataDir = prev;
  rmSync(dir, { recursive: true, force: true });
});

describe("lab run listing", () => {
  test("filters apply before the limit", () => {
    expect(listLabRuns(10)).toHaveLength(10);
    const charts = listLabRuns(10, (m) => m.task_type === "chart");
    expect(charts.map((r) => r.task_type)).toEqual(["chart", "chart"]);
    expect(listLabRuns(10, (m) => m.model_id.includes("old"))).toHaveLength(2);
    expect(listLabRunsByFingerprint("fp-chart", 5)).toHaveLength(2);
    expect(listLabRunsByFingerprint("fp-page", 5)).toHaveLength(5);
  });
});
