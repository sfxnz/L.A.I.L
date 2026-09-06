"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";

/** Scroll-up past this many px releases the pin; scrolling back to the end re-pins. */
export const RELEASE_PX = 64;

/**
 * Mono transcript pinned to the bottom while it streams. The user takes over
 * by scrolling up more than RELEASE_PX; reaching the bottom again hands the
 * pin back. `children` renders before the text (prompt, thinking block).
 */
export function Transcript({
  text,
  empty,
  children,
  className,
  live,
}: {
  text: string;
  empty?: ReactNode;
  children?: ReactNode;
  className?: string;
  live?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [pinned, setPinned] = useState(true);
  const pinnedRef = useRef(true);

  useEffect(() => {
    const el = ref.current;
    if (!el || !pinnedRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [text, children]);

  const onScroll = () => {
    const el = ref.current;
    if (!el) return;
    const fromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const next = fromBottom <= RELEASE_PX ? true : pinnedRef.current && fromBottom > RELEASE_PX ? false : pinnedRef.current;
    if (next !== pinnedRef.current) {
      pinnedRef.current = next;
      setPinned(next);
    }
  };

  return (
    <div className="relative min-h-0 flex-1">
      <div
        ref={ref}
        onScroll={onScroll}
        className={cn(
          "strand-transcript h-full overflow-y-auto overflow-x-hidden px-3 py-2 font-mono text-lab-text-dim",
          className,
        )}
        aria-live={live ? "polite" : undefined}
      >
        {children}
        {text ? <span className="whitespace-pre-wrap break-words">{text}</span> : empty}
      </div>
      {!pinned && (
        <button
          type="button"
          onClick={() => {
            pinnedRef.current = true;
            setPinned(true);
            const el = ref.current;
            if (el) el.scrollTop = el.scrollHeight;
          }}
          className="animus-chamfer-sm absolute right-2 bottom-2 border border-lab-border bg-lab-panel px-2 py-1 font-[family-name:var(--font-display)] text-[10px] font-semibold uppercase tracking-[0.14em] text-lab-text-dim hover:border-lab-line hover:text-lab-text"
        >
          ↓ follow
        </button>
      )}
    </div>
  );
}
