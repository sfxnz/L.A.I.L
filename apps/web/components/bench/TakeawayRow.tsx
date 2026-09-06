"use client";

import { useEffect, useState } from "react";
import { benchJson, benchMarkdown } from "@/lib/bench/export";
import type { BenchResult } from "@/lib/bench/result";
import { Btn, Eyebrow } from "@/components/ui";
import { cn } from "@/lib/utils";

/** Copy as Markdown (⌘⇧C) · Copy JSON · Run ×16/×32 · Run again · Details (⌘/). */

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

export function useCopy(): [string | null, (label: string, text: string) => void] {
  const [flash, setFlash] = useState<string | null>(null);
  useEffect(() => {
    if (!flash) return;
    const t = setTimeout(() => setFlash(null), 1600);
    return () => clearTimeout(t);
  }, [flash]);
  const copy = (label: string, text: string) => {
    void copyText(text).then((ok) => setFlash(ok ? `${label} copied` : "Clipboard blocked — select the details table instead"));
  };
  return [flash, copy];
}

export function TakeawayRow({
  result,
  envelope,
  suggest,
  onRunLevels,
  onRunAgain,
  detailsOpen,
  onToggleDetails,
  disabled,
  copyRequest,
  className,
}: {
  result: BenchResult;
  envelope?: Record<string, unknown> | null;
  /** levels worth adding when saturation was not reached (decode) */
  suggest?: number[];
  onRunLevels?: (levels: number[]) => void;
  onRunAgain: () => void;
  detailsOpen: boolean;
  onToggleDetails: () => void;
  disabled?: boolean;
  /** bumps when ⌘⇧C fires so the copy runs from the keyboard too */
  copyRequest?: number;
  className?: string;
}) {
  const [flash, copy] = useCopy();
  useEffect(() => {
    if (copyRequest) copy("Markdown", benchMarkdown(result));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [copyRequest]);
  return (
    <div className={cn("flex flex-wrap items-center gap-2", className)}>
      <Btn variant="secondary" size="sm" onClick={() => copy("Markdown", benchMarkdown(result))} title="Copy as Markdown (⌘⇧C)">
        Copy Markdown
      </Btn>
      <Btn variant="secondary" size="sm" onClick={() => copy("JSON", benchJson(result, envelope))} title="Copy the run envelope as JSON">
        Copy JSON
      </Btn>
      {result.kind === "decode" && suggest && suggest.length > 0 && onRunLevels && (
        <Btn size="sm" onClick={() => onRunLevels(suggest)} disabled={disabled} title="Saturation not reached — extend the sweep">
          Run {suggest.map((n) => `×${n}`).join("/")}
        </Btn>
      )}
      <Btn variant="secondary" size="sm" onClick={onRunAgain} disabled={disabled} title="Run again with this configuration (r)">
        Run again
      </Btn>
      <Btn variant="ghost" size="sm" onClick={onToggleDetails} aria-expanded={detailsOpen} title="Toggle details (⌘/)">
        {detailsOpen ? "Hide details" : "Details"}
      </Btn>
      {flash && (
        <Eyebrow className="text-lab-ok!" aria-live="polite">
          {flash}
        </Eyebrow>
      )}
    </div>
  );
}
