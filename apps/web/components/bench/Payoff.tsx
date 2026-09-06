"use client";

import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { Eyebrow, HeroNumber, Nil } from "@/components/ui";
import { cn } from "@/lib/utils";

/*
  The payoff — once per run, never on re-render:
    60–80 ms hit-stop (everything freezes)
    → ring closes 500 ms
    → hero counts up 700 ms
    → secondaries stagger in ≤ 320 ms
    → interpretation fades in.
  Reduced motion: one 150 ms crossfade, numbers snapped. A result loaded from
  history skips straight to "settled" — the payoff is earned by a run.
*/

export type PayoffStage = "none" | "hitstop" | "ring" | "hero" | "secondaries" | "settled";

export const PAYOFF = { hitstop: 70, ring: 500, heroLead: 180, hero: 700, secondaries: 320 } as const;

function reducedMotion() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function usePayoff(key: string | null, play: boolean): PayoffStage {
  const [stage, setStage] = useState<PayoffStage>("none");
  const played = useRef<string | null>(null);
  useEffect(() => {
    if (!key) {
      played.current = null;
      setStage("none");
      return;
    }
    if (played.current === key) return;
    played.current = key;
    if (!play || reducedMotion()) {
      setStage("settled");
      return;
    }
    setStage("hitstop");
    const t: ReturnType<typeof setTimeout>[] = [];
    let at = PAYOFF.hitstop;
    t.push(setTimeout(() => setStage("ring"), at));
    at += PAYOFF.ring;
    t.push(setTimeout(() => setStage("hero"), at));
    at += PAYOFF.heroLead;
    t.push(setTimeout(() => setStage("secondaries"), at));
    at += PAYOFF.hero;
    t.push(setTimeout(() => setStage("settled"), at));
    return () => t.forEach(clearTimeout);
  }, [key, play]);
  return stage;
}

const ORDER: Record<PayoffStage, number> = { none: -1, hitstop: 0, ring: 1, hero: 2, secondaries: 3, settled: 4 };
export function reached(stage: PayoffStage, at: PayoffStage): boolean {
  return ORDER[stage] >= ORDER[at];
}

/** Diamond ring (the sync glyph, large) that draws itself closed over 500 ms. */
export function ClosingRing({ closing, done, size = 64 }: { closing: boolean; done: boolean; size?: number }) {
  const D = "M32 3 61 32 32 61 3 32Z";
  return (
    <svg viewBox="0 0 64 64" width={size} height={size} aria-hidden className="shrink-0">
      <path d={D} fill="none" className="stroke-lab-border" strokeWidth="1.5" />
      {(closing || done) && (
        <path d={D} fill="none" pathLength={1} className={cn("stroke-lab-ok", closing && !done && "bench-ring-close")} strokeWidth="2" />
      )}
      {done && <path d="M32 18 46 32 32 46 18 32Z" className="fill-lab-ok bench-settle" />}
    </svg>
  );
}

export type Secondary = { label: string; value: ReactNode; tone?: "target" | "muted" };

export function ResultHero({
  stage,
  value,
  format,
  unit,
  at,
  secondaries,
  sentence,
  className,
}: {
  stage: PayoffStage;
  value: number | null;
  format: (n: number) => string;
  unit: string;
  /** "aggregate @ ×8" */
  at: string;
  secondaries: Secondary[];
  sentence: string;
  className?: string;
}) {
  const showHero = reached(stage, "hero");
  const showSecondaries = reached(stage, "secondaries");
  const settled = reached(stage, "settled");
  // Count up only while the payoff is playing; a settled/loaded result renders its number at rest.
  const animate = stage !== "settled";
  return (
    <div className={cn("grid gap-4 md:grid-cols-[auto_minmax(0,1fr)]", className)}>
      <div className="flex items-start gap-4">
        <ClosingRing closing={stage === "ring" || stage === "hero" || stage === "secondaries"} done={settled} />
        <div className="min-w-0">
          <Eyebrow>{at}</Eyebrow>
          <div className="mt-1 flex items-baseline gap-2">
            {value === null ? (
              <span className="animus-hero text-lab-muted">
                <Nil word="None" />
              </span>
            ) : showHero ? (
              <HeroNumber value={value} from={animate ? 0 : undefined} format={format} label={`${format(value)} ${unit}`} />
            ) : (
              <span className="animus-hero opacity-0" aria-hidden>
                {format(value)}
              </span>
            )}
            <span className="font-[family-name:var(--font-display)] text-[14px] font-semibold uppercase tracking-[0.14em] text-lab-text-dim">
              {unit}
            </span>
          </div>
        </div>
      </div>
      <div className="min-w-0">
        <dl className={cn("grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-3", !showSecondaries && "invisible")}>
          {secondaries.map((s, i) => (
            <div key={s.label} className={cn(showSecondaries && !settled && "bench-secondary")} style={{ "--i": i } as CSSProperties}>
              <dt className="animus-eyebrow text-[9px]">{s.label}</dt>
              <dd
                className={cn(
                  "lab-num mt-0.5 font-[family-name:var(--font-display)] text-[20px] font-semibold leading-none",
                  s.tone === "target" ? "text-lab-target" : s.tone === "muted" ? "text-lab-muted" : "text-lab-text",
                )}
              >
                {s.value}
              </dd>
            </div>
          ))}
        </dl>
        <p className={cn("mt-3 max-w-[60ch] text-[13px] leading-snug text-lab-text-dim", settled ? "bench-fade" : "invisible")}>{sentence}</p>
      </div>
    </div>
  );
}
