"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useCallback, useEffect, useRef } from "react";
import { useTheme, type ThemeChoice } from "@/lib/theme";
import { cn } from "@/lib/utils";

const THEMING_CLASS = "animus-theming";
/** Matches the crossfade duration declared on html.animus-theming in globals.css. */
const THEMING_MS = 300;

const OPTIONS: Array<{
  value: ThemeChoice;
  label: string;
  Icon: React.ComponentType<{ className?: string; strokeWidth?: number }>;
}> = [
  { value: "light", label: "Light", Icon: Sun },
  { value: "dark", label: "Dark", Icon: Moon },
  { value: "system", label: "System", Icon: Monitor },
];

/**
 * Animus world switch — light (reconstruction) / dark (in simulation) / system.
 *
 * Renders neutral until `mounted`, so the server markup and the first client
 * paint agree; the real choice is adopted by useTheme() on the first effect.
 * <html data-theme> is already set pre-paint by THEME_BOOT_SCRIPT.
 */
export function ThemeToggle() {
  const { choice, resolved, setChoice, mounted } = useTheme();
  const fadeTimer = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (fadeTimer.current !== null) window.clearTimeout(fadeTimer.current);
      document.documentElement.classList.remove(THEMING_CLASS);
    },
    [],
  );

  const pick = useCallback(
    (next: ThemeChoice) => {
      // Dissolve between worlds instead of hard-cutting.
      document.documentElement.classList.add(THEMING_CLASS);
      if (fadeTimer.current !== null) window.clearTimeout(fadeTimer.current);
      fadeTimer.current = window.setTimeout(() => {
        document.documentElement.classList.remove(THEMING_CLASS);
        fadeTimer.current = null;
      }, THEMING_MS);
      setChoice(next);
    },
    [setChoice],
  );

  return (
    <div
      role="radiogroup"
      aria-label="Colour theme"
      className="flex shrink-0 items-center gap-px border border-[color:var(--animus-hairline)] bg-lab-input p-px"
    >
      {OPTIONS.map(({ value, label, Icon }) => {
        const active = mounted && choice === value;
        return (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={label}
            title={
              mounted && value === "system" ? `System — following ${resolved}` : label
            }
            onClick={() => pick(value)}
            className={cn(
              "relative flex items-center gap-1.5 px-2 py-[5px] font-[family-name:var(--font-display)] text-[11px] font-semibold uppercase leading-none tracking-[0.14em] transition-colors duration-200 focus-visible:z-10",
              active
                ? "text-lab-text"
                : "text-lab-muted hover:bg-[color:var(--animus-accent-wash)] hover:text-lab-text-dim",
            )}
          >
            {active && (
              <>
                <span
                  aria-hidden
                  className="animus-notch absolute inset-0 bg-[image:var(--animus-selection-fade)] opacity-50"
                />
                <span aria-hidden className="absolute inset-y-0 left-0 w-[2px] bg-lab-accent" />
              </>
            )}
            <Icon className="relative h-3.5 w-3.5 shrink-0" strokeWidth={1.75} aria-hidden />
            <span className="relative hidden lg:inline">{label}</span>
          </button>
        );
      })}
    </div>
  );
}
