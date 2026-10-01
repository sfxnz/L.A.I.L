"use client";

import { STALE_AFTER_S, snapshotAge, useLabStatusStore, useNow } from "@/lib/lab-status-store";
import { fmtAge } from "@/lib/status/format";
import { Eyebrow } from "@/components/ui";
import { cn } from "@/lib/utils";

/**
 * How current the live data is, in words: "live" while samples arrive every second,
 * "polling" on the fallback transport, "not updating · 14 s" once the newest sample
 * is older than STALE_AFTER_S. Ticks on its own clock, so it moves even when
 * nothing else on the page re-renders.
 */
export function LiveAge({ className, quiet }: { className?: string; /** say nothing while live */ quiet?: boolean }) {
  const now = useNow(1000);
  useLabStatusStore((s) => s.receivedAt);
  const s = useLabStatusStore.getState();
  const age = snapshotAge(s, Math.max(now, Date.now()));
  if (age == null) return null;
  const stale = s.unreachable || age > STALE_AFTER_S;
  if (quiet && !stale && s.transport !== "poll") return null;
  return (
    <Eyebrow
      className={cn("lab-num shrink-0 whitespace-nowrap", stale ? "text-lab-warn" : "text-lab-muted", className)}
      title={`Newest sample ${age.toFixed(1)} s old${s.transport === "poll" ? " · stream unavailable, polling every 2 s" : ""}`}
    >
      {stale ? `not updating · ${fmtAge(age)}` : s.transport === "poll" ? "polling" : "live"}
    </Eyebrow>
  );
}
