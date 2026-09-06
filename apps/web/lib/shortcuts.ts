"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef } from "react";
import { create } from "zustand";
import { THEME_CHOICES, THEME_STORAGE_KEY, applyTheme, readStoredChoice, type ThemeChoice } from "./theme";

/**
 * Global keyboard chords (design brief §5.9 — the operator is on macOS).
 *
 *   ⌘K            palette
 *   g s/v/b/l/e/c Status / Serve / Bench / Streams / Evals / Configure
 *   ?             shortcut sheet
 *   ⌘⇧L           cycle theme
 *   ⌘⇧W           wall mode (on /streams)
 *
 * Only `g` (a chord starter) and `?` are global single keys; every other
 * single key belongs to the page under the cursor (`r`, `f`, `t`, digits…).
 * The hook listens in the capture phase so a consumed chord key never reaches
 * a page handler; everything else passes through untouched.
 */

export const CHORD_WINDOW_MS = 1200;

export const CHORD_TARGETS: Record<string, { href: string; label: string }> = {
  s: { href: "/status", label: "Status" },
  v: { href: "/server", label: "Serve" },
  b: { href: "/bench", label: "Bench" },
  l: { href: "/streams", label: "Streams" },
  e: { href: "/evals", label: "Evals" },
  c: { href: "/configure", label: "Configure" },
};

export type KeyInput = {
  key: string;
  meta?: boolean;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  /** focus is in an input / textarea / contenteditable */
  editable?: boolean;
  /** a Radix dialog (palette, sheet, transcript) is open */
  dialogOpen?: boolean;
};

export type ShortcutAction =
  | { type: "palette" }
  | { type: "sheet" }
  | { type: "navigate"; href: string }
  | { type: "theme" }
  | { type: "wall" };

export type ChordMachine = {
  /** Feed one keydown; returns the action to perform (caller consumes the event) or null. */
  feed: (k: KeyInput, now?: number) => ShortcutAction | null;
  /** True while a `g` is pending (for tests / a future hint). */
  pending: () => boolean;
};

/** Pure state machine — no DOM, so the chord logic is unit-testable. */
export function createChordMachine(windowMs = CHORD_WINDOW_MS): ChordMachine {
  let gAt: number | null = null;

  return {
    pending: () => gAt !== null,
    feed(k, now = Date.now()) {
      const mod = !!(k.meta || k.ctrl);

      // Modifier chords work everywhere — including inside inputs — because
      // no page or field claims them.
      if (mod && !k.alt && (k.key === "k" || k.key === "K")) {
        gAt = null;
        return { type: "palette" };
      }
      if (k.dialogOpen) {
        gAt = null;
        return null;
      }
      if (mod && k.shift && !k.alt && (k.key === "L" || k.key === "l")) {
        gAt = null;
        return { type: "theme" };
      }
      if (mod && k.shift && !k.alt && (k.key === "W" || k.key === "w")) {
        gAt = null;
        return { type: "wall" };
      }

      // Single keys never fire while typing.
      if (k.editable || mod || k.alt) {
        gAt = null;
        return null;
      }

      // Pure modifier presses (Shift on the way to `?`) don't cancel a chord.
      if (k.key === "Shift" || k.key === "Meta" || k.key === "Control" || k.key === "Alt") return null;

      if (gAt !== null) {
        const inWindow = now - gAt <= windowMs;
        gAt = null;
        if (inWindow && !k.shift) {
          const t = CHORD_TARGETS[k.key];
          if (t) return { type: "navigate", href: t.href };
        }
        // Any other key cancels the chord and is NOT swallowed — fall through
        // so `g` then `r` still runs the page's `r`.
      }

      if (k.key === "g" && !k.shift) {
        gAt = now;
        return null;
      }
      if (k.key === "?") return { type: "sheet" };
      return null;
    },
  };
}

export function isEditableTarget(t: EventTarget | null): boolean {
  const el = t as HTMLElement | null;
  if (!el || typeof el.tagName !== "string") return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || !!el.isContentEditable;
}

export function isDialogOpen(doc: Document = document): boolean {
  return !!doc.querySelector('[role="dialog"][data-state="open"]');
}

/** True on macOS/iOS — where the palette prints ⌘, elsewhere Ctrl. */
export function isMacPlatform(nav: { platform?: string; userAgent?: string } | undefined = typeof navigator === "undefined" ? undefined : navigator): boolean {
  if (!nav) return true;
  return /Mac|iPhone|iPad|iPod/i.test(nav.platform || "") || /Mac OS X/i.test(nav.userAgent || "");
}

/** Display form of a shortcut: ["⌘","K"] → "⌘K"; ["g","b"] → "g b"; "⌘" → "Ctrl" off-Mac. */
export function formatShortcut(keys: readonly string[], mac = isMacPlatform()): string {
  const mapped = keys.map((k) => (!mac && k === "⌘" ? "Ctrl" : k));
  const chord = mapped.every((k) => k.length === 1 && /[a-z0-9?]/i.test(k));
  if (chord && mapped.length > 1) return mapped.join(" ");
  if (!mac) return mapped.join("+");
  return mapped.join("");
}

// ── Sheet tables ───────────────────────────────────────────────────────────

export type ShortcutRow = { keys: string[]; label: string };

export const GLOBAL_SHORTCUTS: ShortcutRow[] = [
  { keys: ["⌘", "K"], label: "Command palette" },
  { keys: ["g", "s"], label: "Go to Status" },
  { keys: ["g", "v"], label: "Go to Serve" },
  { keys: ["g", "b"], label: "Go to Bench" },
  { keys: ["g", "l"], label: "Go to Streams" },
  { keys: ["g", "e"], label: "Go to Evals" },
  { keys: ["g", "c"], label: "Go to Configure" },
  { keys: ["⌘", "⇧", "L"], label: "Cycle theme" },
  { keys: ["⌘", "⇧", "W"], label: "Wall mode (Streams)" },
  { keys: ["?"], label: "This sheet" },
  { keys: ["esc"], label: "Close palette / sheet" },
];

/** Per-route tables mirror the pages' own keydown handlers (app/bench, app/streams). */
export const ROUTE_SHORTCUTS: Record<string, { title: string; rows: ShortcutRow[] }> = {
  "/bench": {
    title: "Bench",
    rows: [
      { keys: ["r"], label: "Run" },
      { keys: ["⌘", "↩"], label: "Run" },
      { keys: ["⌘", "."], label: "Stop run" },
      { keys: ["1"], label: "…6 — toggle level ×1 ×2 ×4 ×8 ×16 ×32" },
      { keys: ["["], label: "Previous run" },
      { keys: ["]"], label: "Next run" },
      { keys: ["⌘", "/"], label: "Toggle details" },
      { keys: ["⌘", "⇧", "C"], label: "Copy Markdown" },
    ],
  },
  "/streams": {
    title: "Streams",
    rows: [
      { keys: ["r"], label: "Run pack" },
      { keys: ["⌘", "."], label: "Stop run" },
      { keys: ["⌘", "⇧", "C"], label: "Copy all transcripts" },
      { keys: ["1"], label: "…9 — focus strand" },
      { keys: ["f"], label: "Pin hovered strand" },
      { keys: ["t"], label: "Toggle thinking" },
      { keys: ["c"], label: "Copy strand" },
      { keys: ["↩"], label: "Open transcript" },
      { keys: ["⇧", "S"], label: "Helix ⇄ Sequence view" },
      { keys: ["esc"], label: "Unfocus / exit wall" },
    ],
  },
};

export function routeShortcuts(pathname: string): { title: string; rows: ShortcutRow[] } | null {
  for (const [route, table] of Object.entries(ROUTE_SHORTCUTS)) {
    if (pathname === route || pathname.startsWith(route + "/")) return table;
  }
  return null;
}

// ── Theme ──────────────────────────────────────────────────────────────────

const THEME_LABEL: Record<ThemeChoice, string> = { light: "Light", dark: "Dark", system: "System" };

export function currentThemeChoice(): ThemeChoice {
  const c = document.documentElement.dataset.themeChoice;
  return c === "light" || c === "dark" || c === "system" ? c : readStoredChoice();
}

/**
 * Pick a theme through the ThemeToggle's own radio when it is mounted, so the
 * toggle's state and the crossfade stay in one place; fall back to applying
 * it directly (wall mode hides the header).
 */
export function pickTheme(next: ThemeChoice): void {
  const radio = document.querySelector<HTMLButtonElement>(
    `[role="radiogroup"][aria-label="Colour theme"] [role="radio"][aria-label="${THEME_LABEL[next]}"]`,
  );
  if (radio) {
    radio.click();
    return;
  }
  applyTheme(next);
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, next);
  } catch {
    /* private mode */
  }
}

export function cycleTheme(): ThemeChoice {
  const cur = currentThemeChoice();
  const next = THEME_CHOICES[(THEME_CHOICES.indexOf(cur) + 1) % THEME_CHOICES.length];
  pickTheme(next);
  return next;
}

/** Flip `?wall=1` on the current /streams URL (the page reads it via useSearchParams). */
export function toggleWall(pathname: string, replace: (href: string) => void, on?: boolean): void {
  const q = new URLSearchParams(window.location.search);
  const next = on ?? q.get("wall") !== "1";
  if (next) q.set("wall", "1");
  else q.delete("wall");
  const qs = q.toString();
  replace(qs ? `${pathname}?${qs}` : pathname);
}

// ── UI store shared by the hook, the palette and the sheet ─────────────────

type CommandUi = {
  paletteOpen: boolean;
  sheetOpen: boolean;
  setPalette: (open: boolean) => void;
  setSheet: (open: boolean) => void;
};

export const useCommandUi = create<CommandUi>((set) => ({
  paletteOpen: false,
  sheetOpen: false,
  setPalette: (paletteOpen) => set({ paletteOpen, sheetOpen: false }),
  setSheet: (sheetOpen) => set({ sheetOpen, paletteOpen: false }),
}));

/** Mounted once in AppShell. */
export function useGlobalShortcuts(): void {
  const router = useRouter();
  const pathname = usePathname();
  const machine = useRef<ChordMachine | null>(null);
  const latest = useRef(pathname);
  latest.current = pathname;

  useEffect(() => {
    machine.current ??= createChordMachine();
    const onKey = (e: KeyboardEvent) => {
      if (e.isComposing || e.repeat) return;
      const ui = useCommandUi.getState();
      const action = machine.current!.feed({
        key: e.key,
        meta: e.metaKey,
        ctrl: e.ctrlKey,
        shift: e.shiftKey,
        alt: e.altKey,
        editable: isEditableTarget(e.target),
        dialogOpen: isDialogOpen(),
      });
      if (!action) return;
      e.preventDefault();
      e.stopPropagation();
      const path = latest.current;
      switch (action.type) {
        case "palette":
          ui.setPalette(!ui.paletteOpen);
          break;
        case "sheet":
          ui.setSheet(true);
          break;
        case "navigate":
          router.push(action.href);
          break;
        case "theme":
          cycleTheme();
          break;
        case "wall":
          if (path?.startsWith("/streams")) toggleWall(path, router.replace);
          break;
      }
    };
    window.addEventListener("keydown", onKey, { capture: true });
    return () => window.removeEventListener("keydown", onKey, { capture: true });
  }, [router]);
}
