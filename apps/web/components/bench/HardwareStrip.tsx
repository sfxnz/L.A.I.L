"use client";

import { useEffect, useRef, useState } from "react";
import { appendHardware, type HardwareSeries } from "@/lib/bench/live";
import { useLabStatusStore } from "@/lib/lab-status-store";
import { Eyebrow, SparkStat, Tick } from "@/components/ui";
import { cn } from "@/lib/utils";

/**
 * THE hardware strip: per node GPU util · power · temperature on the run's
 * clock. Samples come from the shared lab-status ring buffer (one sampler for
 * the whole app) while `active`; the series freezes once the run ends so the
 * strip stays a record of what the hardware did under the curve.
 */
export function useHardwareSamples(active: boolean, runKey: string | null): HardwareSeries[] {
  const samples = useLabStatusStore((s) => s.samples);
  const status = useLabStatusStore((s) => s.status);
  const [series, setSeries] = useState<HardwareSeries[]>([]);
  const key = useRef<string | null>(null);
  const since = useRef<number>(0);
  useEffect(() => {
    if (runKey !== key.current) {
      key.current = runKey;
      since.current = Date.now();
      setSeries([]);
    }
  }, [runKey]);
  useEffect(() => {
    if (!active) return;
    const nodes = status?.cluster?.nodes ?? status?.serve?.cluster?.nodes;
    setSeries((prev) => appendHardware(prev, samples, nodes, since.current));
  }, [active, samples, status]);
  return series;
}

export function HardwareStrip({ series, className }: { series: HardwareSeries[]; className?: string }) {
  if (!series.length) return null;
  return (
    <div className={cn("flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-lab-border-subtle pt-3", className)} aria-label="Hardware during the run">
      {series.map((s, idx) => {
        const util = s.samples.map((x) => x.util).filter((v): v is number => v !== null);
        const power = s.samples.map((x) => x.power).filter((v): v is number => v !== null);
        const temp = s.samples.map((x) => x.temp).filter((v): v is number => v !== null);
        const hot = temp.length && temp[temp.length - 1] >= 80;
        return (
          <div key={s.id} className="flex flex-wrap items-center gap-x-4 gap-y-2">
            {idx > 0 && <Tick className="hidden h-6 sm:block" />}
            <Eyebrow className="w-16 truncate text-lab-text-dim!" title={s.label}>
              {s.label}
            </Eyebrow>
            <SparkStat label="GPU" unit="%" values={util} max={100} tone="text-lab-line" />
            <SparkStat label="Power" unit=" W" values={power} tone="text-lab-line-2" />
            <SparkStat label="Temp" unit="°C" values={temp} max={100} tone={hot ? "text-lab-warn" : "text-lab-muted"} />
          </div>
        );
      })}
    </div>
  );
}
