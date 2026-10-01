/**
 * Two-click confirm for a destructive button (Serve's Stop). The first click arms it;
 * a second click confirms only after CONFIRM_MIN_MS — a double-click's second click
 * lands sooner and is ignored — and within CONFIRM_WINDOW_MS, after which it disarms.
 */
export const CONFIRM_MIN_MS = 400;
export const CONFIRM_WINDOW_MS = 4000;

export function confirmStep(armedAt: number | null, now: number): "arm" | "ignore" | "confirm" {
  if (armedAt == null || now - armedAt > CONFIRM_WINDOW_MS) return "arm";
  return now - armedAt < CONFIRM_MIN_MS ? "ignore" : "confirm";
}
