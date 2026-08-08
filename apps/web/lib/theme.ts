"use client";

import { useCallback, useEffect, useState } from "react";

/**
 * Animus theme engine — light ("reconstruction"), dark ("in-simulation"), system.
 *
 * The resolved theme lives on <html data-theme>, which is what globals.css keys
 * every token off. The choice (which may be "system") lives on data-theme-choice
 * so the toggle can render the user's actual selection, not the resolved one.
 */

export type ThemeChoice = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

export const THEME_STORAGE_KEY = "lail-theme";
export const THEME_CHOICES: ThemeChoice[] = ["light", "dark", "system"];

const DARK_QUERY = "(prefers-color-scheme: dark)";

function isChoice(v: unknown): v is ThemeChoice {
  return v === "light" || v === "dark" || v === "system";
}

export function systemTheme(): ResolvedTheme {
  if (typeof window === "undefined") return "dark";
  return window.matchMedia(DARK_QUERY).matches ? "dark" : "light";
}

export function resolveTheme(choice: ThemeChoice): ResolvedTheme {
  return choice === "system" ? systemTheme() : choice;
}

export function readStoredChoice(): ThemeChoice {
  if (typeof window === "undefined") return "system";
  try {
    const raw = window.localStorage.getItem(THEME_STORAGE_KEY);
    return isChoice(raw) ? raw : "system";
  } catch {
    return "system";
  }
}

export function applyTheme(choice: ThemeChoice): ResolvedTheme {
  const resolved = resolveTheme(choice);
  const el = document.documentElement;
  el.dataset.theme = resolved;
  el.dataset.themeChoice = choice;
  el.style.colorScheme = resolved;
  return resolved;
}

/**
 * Boot script — must run before first paint or the wrong theme flashes.
 * Inlined into <head>; keep it dependency-free and exception-safe.
 */
export const THEME_BOOT_SCRIPT = `(function(){try{var k=${JSON.stringify(
  THEME_STORAGE_KEY,
)};var c=localStorage.getItem(k);if(c!=="light"&&c!=="dark"&&c!=="system")c="system";var r=c==="system"?(window.matchMedia("${DARK_QUERY}").matches?"dark":"light"):c;var e=document.documentElement;e.dataset.theme=r;e.dataset.themeChoice=c;e.style.colorScheme=r;}catch(_){var e2=document.documentElement;e2.dataset.theme="dark";e2.dataset.themeChoice="system";}})();`;

export function useTheme() {
  const [choice, setChoiceState] = useState<ThemeChoice>("system");
  const [resolved, setResolved] = useState<ResolvedTheme>("dark");
  const [mounted, setMounted] = useState(false);

  // Adopt whatever the boot script already put on <html> — no second flash.
  useEffect(() => {
    const el = document.documentElement;
    const bootChoice = el.dataset.themeChoice;
    const c = isChoice(bootChoice) ? bootChoice : readStoredChoice();
    setChoiceState(c);
    setResolved(resolveTheme(c));
    setMounted(true);
  }, []);

  // Follow the OS only while the choice is "system".
  useEffect(() => {
    if (!mounted || choice !== "system") return;
    const mq = window.matchMedia(DARK_QUERY);
    const onChange = () => setResolved(applyTheme("system"));
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [choice, mounted]);

  const setChoice = useCallback((next: ThemeChoice) => {
    setChoiceState(next);
    setResolved(applyTheme(next));
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      /* private mode — theme still applies for this session */
    }
  }, []);

  return { choice, resolved, setChoice, mounted };
}
