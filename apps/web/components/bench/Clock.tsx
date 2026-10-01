"use client";

import { useEffect, useState } from "react";
import { fmtDuration } from "@/lib/bench/format";

/**
 * Wall clock that ticks only while `active`, in the component that reads it — so a
 * running timer re-renders a few characters, not the whole bench room.
 */
export function useNow(active: boolean, intervalMs = 250): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const t = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(t);
  }, [active, intervalMs]);
  return now;
}

/** Elapsed run time, ticking while the run is live. */
export function Elapsed({ startedAt }: { startedAt: number }) {
  const now = useNow(true);
  return <span>{fmtDuration(Math.max(0, now - startedAt))}</span>;
}
