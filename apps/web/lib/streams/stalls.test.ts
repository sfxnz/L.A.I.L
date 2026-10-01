import { describe, expect, test } from "bun:test";
import { DESYNC_MS, STALL_MS, itlStats, stallIndices, strandDesync } from "./stalls";

describe("stall markers and desync", () => {
  test("stallIndices flags gaps at or above STALL_MS", () => {
    expect(stallIndices([18, 20, 2400, 19, STALL_MS, 17])).toEqual([2, 4]);
    expect(stallIndices(undefined)).toEqual([]);
    expect(stallIndices([])).toEqual([]);
  });

  test("itlStats: per-token p50/p95 from decode steps, the longest gap, and stalls on the raw gap", () => {
    // without token counts every step is one token
    expect(itlStats([20, 18, 22, 3000, 19])).toEqual({ p50: 20, p95: 3000, max: 3000, stalls: 1 });
    // MTP: 46 ms steps of 4 tokens are 11.5 ms per token; a 2.4 s gap that carried 4 tokens
    // is still a stall (no output for 2.4 s), though it is 600 ms per token
    const st = itlStats([46, 46, 46, 46, 2400], [4, 4, 4, 4, 4]);
    expect(st).toEqual({ p50: 11.5, p95: 600, max: 2400, stalls: 1 });
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
