"use client";

import { useEffect, useRef, useState } from "react";
import { appendHardware, type HardwareSeries } from "@/lib/bench/live";
import { serverNow, useLabStatusStore } from "@/lib/lab-status-store";
import { Eyebrow, SparkStat, Tick } from "@/components/ui";
import { cn } from "@/lib/utils";

/**
 * THE hardware strip: per node GPU util · power · temperature on the run's
 * clock. Samples come from the shared lab-status series (one sampler for the
 * whole app) while `active`; the series freezes once the run ends so the strip
 * stays a record of what the hardware did under the curve.
 */
export function useHardwareSamples(active: boolean, runKey: string | null): HardwareSeries[] {
  // Subscribed only while a run is live: an idle Bench page must not re-render on
  // every 1 s sample.
  const samples = useLabStatusStore((s) => (active ? s.samples.nodes : null));
  const nodes = useLabStatusStore((s) => (active ? s.status?.serve?.cluster?.nodes : undefined));
  const [series, setSeries] = useState<HardwareSeries[]>([]);
  const key = useRef<string | null>(null);
  const since = useRef<number>(0);
  useEffect(() => {
    if (runKey !== key.current) {
      key.current = runKey;
      // the run starts now on the server's clock (sample timestamps are server time)
      since.current = serverNow(useLabStatusStore.getState()) ?? Date.now();
      setSeries([]);
    }
  }, [runKey]);
  useEffect(() => {
    if (!active || !samples) return;
    setSeries((prev) => appendHardware(prev, samples, nodes, since.current));
  }, [active, samples, nodes]);
  return series;
}

export function HardwareStrip({ series, className }: { series: HardwareSeries[]; className?: string }) {
  if (!series.length) return null;
  return (
    <div className={cn("flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-lab-border-subtle pt-3", className)} aria-label="Hardware during the run">
      {series.map((s, idx) => {
        const domain = s.samples.length ? ([s.samples[0].t, s.samples[s.samples.length - 1].t] as const) : undefined;
        const util = s.samples.map((x) => ({ t: x.t, v: x.util }));
        const power = s.samples.map((x) => ({ t: x.t, v: x.power }));
        const temp = s.samples.map((x) => ({ t: x.t, v: x.temp }));
        const lastTemp = s.samples[s.samples.length - 1]?.temp;
        const hot = lastTemp != null && lastTemp >= 80;
        return (
          <div key={s.id} className="flex flex-wrap items-center gap-x-4 gap-y-2">
            {idx > 0 && <Tick className="hidden h-6 sm:block" />}
            <Eyebrow className="w-16 truncate text-lab-text-dim!" title={s.label}>
              {s.label}
            </Eyebrow>
            <SparkStat label="GPU" unit="%" points={util} domain={domain} max={100} tone="text-lab-line" />
            <SparkStat label="Power" unit=" W" points={power} domain={domain} tone="text-lab-line-2" />
            <SparkStat label="Temp" unit="°C" points={temp} domain={domain} max={100} tone={hot ? "text-lab-warn" : "text-lab-muted"} />
          </div>
        );
      })}
    </div>
  );
}
