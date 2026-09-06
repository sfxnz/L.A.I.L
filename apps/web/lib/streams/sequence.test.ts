import { describe, expect, test } from "bun:test";
import { sequenceLayout, toRunMs, type SeqInput } from "./sequence";

const clock = { t_ms: 5000, at: 105_000 }; // run-ms 5000 arrived at client stamp 105000
const helloAt = 100_000; // → run-ms 0

describe("sequence view geometry", () => {
  test("toRunMs maps client stamps through the run clock, else hello-relative, else 0", () => {
    expect(toRunMs(105_000, clock, helloAt)).toBe(5000);
    expect(toRunMs(104_000, clock, helloAt)).toBe(4000);
    expect(toRunMs(100_400, null, helloAt)).toBe(400);
    expect(toRunMs(123, null, null)).toBe(0);
  });

  test("waiting → ttft → decode → done, in run-ms", () => {
    const s: SeqInput = { i: 0, state: "done", at_prefill: 100_250, at_decode: 100_650, at_end: 103_000, at_last_delta: 102_900, stalls: [] };
    const { rows, t_min, t_max } = sequenceLayout([s], clock, helloAt, 106_000);
    expect(t_min).toBe(0);
    expect(rows[0].segments).toEqual([
      { kind: "wait", x0: 0, x1: 250 },
      { kind: "ttft", x0: 250, x1: 650 },
      { kind: "decode", x0: 650, x1: 3000 },
    ]);
    expect(rows[0].end).toBe(3000);
    expect(t_max).toBe(6000); // now
  });

  test("a live decode runs to now; observed stalls and an open stall are hatched", () => {
    const s: SeqInput = {
      i: 1,
      state: "decode",
      at_prefill: 100_000,
      at_decode: 100_300,
      at_last_delta: 103_000,
      stalls: [{ from: 101_000, to: 103_000 }],
    };
    const { rows } = sequenceLayout([s], clock, helloAt, 106_000);
    expect(rows[0].segments).toEqual([
      { kind: "ttft", x0: 0, x1: 300 },
      { kind: "decode", x0: 300, x1: 6000 },
      { kind: "stall", x0: 1000, x1: 3000 },
      { kind: "stall", x0: 3000, x1: 6000 },
    ]);
  });

  test("a strand never seen leaving the queue is one waiting bar; prefill without a first token is all TTFT", () => {
    const waiting: SeqInput = { i: 0, state: "waiting", stalls: [] };
    const prefill: SeqInput = { i: 1, state: "prefill", at_prefill: 100_500, stalls: [] };
    const { rows } = sequenceLayout([waiting, prefill], clock, helloAt, 102_000);
    expect(rows[0].segments).toEqual([{ kind: "wait", x0: 0, x1: 2000 }]);
    expect(rows[1].segments).toEqual([
      { kind: "wait", x0: 0, x1: 500 },
      { kind: "ttft", x0: 500, x1: 2000 },
    ]);
  });

  test("t_max covers the latest segment even after the run finished", () => {
    const s: SeqInput = { i: 0, state: "done", at_prefill: 100_000, at_decode: 100_100, at_end: 110_000, stalls: [] };
    const { t_max } = sequenceLayout([s], clock, helloAt, 108_000);
    expect(t_max).toBe(10_000);
  });
});
