import { describe, expect, test } from "bun:test";
import {
  CHORD_WINDOW_MS,
  GLOBAL_SHORTCUTS,
  ROUTE_SHORTCUTS,
  createChordMachine,
  formatShortcut,
  routeShortcuts,
} from "./shortcuts";

describe("chord state machine", () => {
  test("g then b navigates to /bench inside the window", () => {
    const m = createChordMachine();
    expect(m.feed({ key: "g" }, 0)).toBeNull();
    expect(m.pending()).toBe(true);
    expect(m.feed({ key: "b" }, 500)).toEqual({ type: "navigate", href: "/bench" });
    expect(m.pending()).toBe(false);
  });

  test("every nav room has a chord", () => {
    const m = createChordMachine();
    const hrefs = ["s", "v", "b", "l", "e", "c"].map((k) => {
      m.feed({ key: "g" }, 0);
      return (m.feed({ key: k }, 1) as { href: string }).href;
    });
    expect(hrefs).toEqual(["/status", "/server", "/bench", "/streams", "/evals", "/configure"]);
  });

  test("chord times out after CHORD_WINDOW_MS and the key falls through", () => {
    const m = createChordMachine();
    m.feed({ key: "g" }, 0);
    expect(m.feed({ key: "b" }, CHORD_WINDOW_MS + 1)).toBeNull();
    expect(m.pending()).toBe(false);
  });

  test("any other key cancels the chord without being swallowed", () => {
    const m = createChordMachine();
    m.feed({ key: "g" }, 0);
    // `r` belongs to the page (Bench/Streams run) — must return null so the page still sees it.
    expect(m.feed({ key: "r" }, 100)).toBeNull();
    expect(m.pending()).toBe(false);
    expect(m.feed({ key: "b" }, 200)).toBeNull();
  });

  test("a Shift press on the way to ? does not cancel or fire", () => {
    const m = createChordMachine();
    expect(m.feed({ key: "Shift", shift: true }, 0)).toBeNull();
    expect(m.feed({ key: "?", shift: true }, 10)).toEqual({ type: "sheet" });
  });

  test("page single keys are never global", () => {
    const m = createChordMachine();
    for (const key of ["r", "f", "t", "1", "9", "[", "]", "c", "Enter"]) {
      expect(m.feed({ key }, 0)).toBeNull();
    }
    expect(m.feed({ key: "S", shift: true }, 0)).toBeNull();
  });

  test("input focus guards single keys but not ⌘K", () => {
    const m = createChordMachine();
    expect(m.feed({ key: "g", editable: true }, 0)).toBeNull();
    expect(m.feed({ key: "b", editable: true }, 10)).toBeNull();
    expect(m.feed({ key: "?", shift: true, editable: true }, 20)).toBeNull();
    expect(m.feed({ key: "k", meta: true, editable: true }, 30)).toEqual({ type: "palette" });
  });

  test("typing g in an input then b outside does not navigate", () => {
    const m = createChordMachine();
    m.feed({ key: "g", editable: true }, 0);
    expect(m.feed({ key: "b" }, 10)).toBeNull();
  });

  test("an open dialog swallows everything except ⌘K", () => {
    const m = createChordMachine();
    expect(m.feed({ key: "g", dialogOpen: true }, 0)).toBeNull();
    expect(m.feed({ key: "b", dialogOpen: true }, 10)).toBeNull();
    expect(m.feed({ key: "?", shift: true, dialogOpen: true }, 20)).toBeNull();
    expect(m.feed({ key: "L", meta: true, shift: true, dialogOpen: true }, 30)).toBeNull();
    expect(m.feed({ key: "k", meta: true, dialogOpen: true }, 40)).toEqual({ type: "palette" });
  });

  test("modifier chords: ⌘K / Ctrl+K palette, ⌘⇧L theme, ⌘⇧W wall", () => {
    const m = createChordMachine();
    expect(m.feed({ key: "k", meta: true })).toEqual({ type: "palette" });
    expect(m.feed({ key: "k", ctrl: true })).toEqual({ type: "palette" });
    expect(m.feed({ key: "L", meta: true, shift: true })).toEqual({ type: "theme" });
    expect(m.feed({ key: "W", meta: true, shift: true })).toEqual({ type: "wall" });
    // The pages' own ⌘. / ⌘⇧C / ⌘/ pass through.
    expect(m.feed({ key: ".", meta: true })).toBeNull();
    expect(m.feed({ key: "C", meta: true, shift: true })).toBeNull();
    expect(m.feed({ key: "/", meta: true })).toBeNull();
  });
});

describe("shortcut display", () => {
  test("formatShortcut prints macOS glyphs and Ctrl elsewhere", () => {
    expect(formatShortcut(["⌘", "K"], true)).toBe("⌘K");
    expect(formatShortcut(["⌘", "K"], false)).toBe("Ctrl+K");
    expect(formatShortcut(["g", "b"], true)).toBe("g b");
    expect(formatShortcut(["⌘", "⇧", "L"], true)).toBe("⌘⇧L");
  });

  test("route tables cover bench and streams and resolve by prefix", () => {
    expect(routeShortcuts("/bench")?.title).toBe("Bench");
    expect(routeShortcuts("/streams/x")?.title).toBe("Streams");
    expect(routeShortcuts("/status")).toBeNull();
    const labels = (r: string) => ROUTE_SHORTCUTS[r].rows.map((x) => x.keys.join(""));
    expect(labels("/streams")).toEqual(expect.arrayContaining(["r", "⌘.", "⌘⇧C", "f", "t", "↩", "⇧S"]));
    expect(labels("/bench")).toEqual(expect.arrayContaining(["r", "⌘.", "⌘/", "⌘⇧C", "[", "]"]));
    expect(GLOBAL_SHORTCUTS.map((g) => g.keys.join(""))).toContain("⌘K");
  });
});
