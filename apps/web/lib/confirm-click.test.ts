import { describe, expect, test } from "bun:test";
import { CONFIRM_MIN_MS, CONFIRM_WINDOW_MS, confirmStep } from "./confirm-click";

describe("confirmStep", () => {
  test("arm, then a double-click's second click is ignored, then a deliberate click confirms", () => {
    const t0 = 10_000;
    expect(confirmStep(null, t0)).toBe("arm");
    expect(confirmStep(t0, t0 + 120)).toBe("ignore"); // same double-click
    expect(confirmStep(t0, t0 + CONFIRM_MIN_MS - 1)).toBe("ignore");
    expect(confirmStep(t0, t0 + CONFIRM_MIN_MS)).toBe("confirm");
    expect(confirmStep(t0, t0 + CONFIRM_WINDOW_MS)).toBe("confirm");
  });

  test("a second click after the window re-arms instead of stopping", () => {
    expect(confirmStep(0, CONFIRM_WINDOW_MS + 1)).toBe("arm");
  });
});
