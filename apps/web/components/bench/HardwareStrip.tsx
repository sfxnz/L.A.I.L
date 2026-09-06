"use client";

import { useEffect, useRef, useState } from "react";
import { sampleHardware, type HardwareSeries } from "@/lib/bench/live";
import { useLabStatus } from "@/lib/lab-status-store";
import { Eyebrow, Nil, Sparkline, Tick } from "@/components/ui";
import { cn } from "@/lib/utils";

/**
 * Per node: GPU util · power · temperature on the run's clock, sampled from the
 * shared 2 s lab-status poll while `active`; frozen once the run ends so the
 * strip stays a record of what the hardware did under the curve.
 */
export function useHardwareSamples(active: boolean, runKey: string | null): HardwareSeries[] {
  const { status, lastGoodAt } = useLabStatus();
  const [series, setSeries] = useState<HardwareSeries[]>([]);
  const key = useRef<string | null>(null);
  useEffect(() => {
    if (runKey !== key.current) {
      key.current = runKey;
      setSeries([]);
    }
  }, [runKey]);
  useEffect(() => {
    if (!active || !lastGoodAt) return;
    const nodes = status?.cluster?.nodes ?? status?.serve?.cluster?.nodes;
    setSeries((prev) => sampleHardware(prev, nodes, lastGoodAt));
  }, [active, lastGoodAt, status]);
  return series;
}

function Cell({ label, unit, values, max, tone }: { label: string; unit: string; values: number[]; max?: number; tone: string }) {
  const last = values.length ? values[values.length - 1] : null;
  return (
    <div className="flex min-w-0 items-center gap-2">
      <div className="min-w-0">
        <Eyebrow className="block text-[9px]">{label}</Eyebrow>
        <div className="lab-num mt-0.5 font-mono text-[12px] text-lab-text">
          {last !== null ? (
            <>
              {Math.round(last)}
              <span className="text-lab-muted">{unit}</span>
            </>
          ) : (
            <Nil />
          )}
        </div>
      </div>
      <Sparkline points={values} width={72} height={22} min={0} max={max} className={cn("shrink-0", tone)} label={`${label} over the run`} />
    </div>
  );
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
            <Cell label="GPU" unit="%" values={util} max={100} tone="text-lab-line" />
            <Cell label="Power" unit=" W" values={power} tone="text-lab-line-2" />
            <Cell label="Temp" unit="°C" values={temp} max={100} tone={hot ? "text-lab-warn" : "text-lab-muted"} />
          </div>
        );
      })}
    </div>
  );
}
