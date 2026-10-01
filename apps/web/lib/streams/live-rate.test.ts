import { describe, expect, test } from "bun:test";
import { RATE_KEEP_MS, RATE_WINDOW_MS, pushSample, stackShares, windowRate, type RateSample } from "./live-rate";

function series(points: Array<[number, number]>): RateSample[] {
  return points.reduce<RateSample[]>((acc, [at, tokens]) => pushSample(acc, at, tokens), []);
}

describe("per-strand live rate window", () => {
  test("pushSample keeps the ring bounded to RATE_KEEP_MS", () => {
    const s = series([
      [0, 1],
      [1000, 5],
      [2000, 9],
      [4000, 12],
    ]);
    expect(s.map((p) => p.at)).toEqual([1000, 2000, 4000]);
    expect(s[0].at).toBeGreaterThanOrEqual(4000 - RATE_KEEP_MS);
  });

  test("steady stream: token growth over the time it grew", () => {
    // a coalesced delta every 100 ms carrying 4 tokens → 40 tok/s
    const pts: Array<[number, number]> = [];
    for (let k = 0; k <= 30; k++) pts.push([k * 100, k * 4]);
    expect(windowRate(series(pts), 3000)).toBe(40);
  });

  test("coalesced samples: growth is divided by the time since the baseline sample, not the window", () => {
    // deltas every 400 ms carrying 20 tokens (50 tok/s). At 3000 the window starts at 1500;
    // the baseline is the sample at 1200, so 1800 ms of growth (90 tokens) — not 1500 ms.
    const pts: Array<[number, number]> = [];
    for (let k = 0; k <= 7; k++) pts.push([k * 400, k * 20]);
    const s = [...series(pts), { at: 3000, tokens: 150 }];
    expect(windowRate(s, 3000)).toBe(50);
  });

  test("first second reads a rate from the earliest sample instead of zero", () => {
    const s = series([
      [0, 0],
      [80, 1],
      [160, 2],
      [240, 3],
    ]);
    // 3 tokens over max(250, 240) ms → 12 tok/s
    expect(windowRate(s, 240)).toBe(12);
    expect(windowRate([{ at: 0, tokens: 4 }], 500)).toBe(0);
  });

  test("a strand that stopped decays to zero once the window has passed", () => {
    const s = series([
      [0, 0],
      [500, 5],
      [1000, 10],
    ]);
    expect(windowRate(s, 1000)).toBeGreaterThan(0);
    expect(windowRate(s, 1000 + RATE_WINDOW_MS + 1)).toBe(0);
  });

  test("stackShares scales the split so the top edge equals the engine aggregate", () => {
    const shares = stackShares([10, 30], 100, [true, true]);
    expect(shares).toEqual([25, 75]);
    expect(shares.reduce((a, b) => a + b, 0)).toBe(100);
  });

  test("stackShares splits evenly across active strands when the client saw no growth yet", () => {
    expect(stackShares([0, 0, 0], 60, [true, false, true])).toEqual([30, 0, 30]);
    expect(stackShares([0, 0], 60, [false, false])).toEqual([0, 0]);
    expect(stackShares([5, 5], 0, [true, true])).toEqual([0, 0]);
  });
});
