import { describe, expect, test } from "bun:test";
import { DESYNC_MS, STALL_MS, itlStats, percentile, stallIndices, strandDesync } from "./stalls";

describe("stall markers and desync", () => {
  test("stallIndices flags gaps at or above STALL_MS", () => {
    expect(stallIndices([18, 20, 2400, 19, STALL_MS, 17])).toEqual([2, 4]);
    expect(stallIndices(undefined)).toEqual([]);
    expect(stallIndices([])).toEqual([]);
  });

  test("percentile is nearest-rank like the engine", () => {
    const s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    expect(percentile(s, 50)).toBe(5);
    expect(percentile(s, 95)).toBe(10);
    expect(percentile([], 50)).toBeNull();
  });

  test("itlStats gives p50/p95/max and the stall count", () => {
    expect(itlStats([20, 18, 22, 3000, 19])).toEqual({ p50: 20, p95: 3000, max: 3000, stalls: 1 });
    expect(itlStats(undefined)).toEqual({ p50: null, p95: null, max: null, stalls: 0 });
  });

  test("error is a desync at once; a decode gap ≥ DESYNC_MS is a desync; prefill never is", () => {
    expect(strandDesync({ state: "error", error: "HTTP 500" }, 0)).toMatchObject({ desync: true, reason: "HTTP 500" });
    expect(strandDesync({ state: "decode", at_last_delta: 1000 }, 1000 + DESYNC_MS - 1)).toMatchObject({
      desync: false,
      stalledMs: DESYNC_MS - 1,
    });
    expect(strandDesync({ state: "decode", at_last_delta: 1000 }, 1000 + DESYNC_MS)).toMatchObject({
      desync: true,
      stalledMs: DESYNC_MS,
    });
    expect(strandDesync({ state: "decode", at_decode: 0 }, 9000).desync).toBe(true);
    expect(strandDesync({ state: "prefill", at_decode: 0 }, 99_000).desync).toBe(false);
    expect(strandDesync({ state: "done" }, 99_000).desync).toBe(false);
  });
});
