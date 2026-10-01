"use client";

import { useEffect, useState } from "react";
import { api, type ServeEngine } from "@/lib/api";
import { Field, inputCls } from "@/components/ui";
import { cn } from "@/lib/utils";

/**
 * Engines Serve can launch, from the serve-engine (its adapters own the defaults and
 * the fields each engine translates). Until that answers — or if it cannot — the four
 * names are offered with every field shown, so the form never hides a flag on a guess.
 */
const FALLBACK: ServeEngine[] = [
  { name: "vllm", label: "vLLM", default_port: 8000 },
  { name: "sglang", label: "SGLang", default_port: 30000 },
  { name: "llamacpp", label: "llama.cpp", default_port: 8080 },
  { name: "tensorfold", label: "TensorFold", default_port: 8090 },
].map((e) => ({ ...e, default_image: "", image_env: "", max_tp: null, fields: [], notes: "" }));

export function useServeEngines(): ServeEngine[] {
  const [engines, setEngines] = useState<ServeEngine[]>(FALLBACK);
  useEffect(() => {
    let live = true;
    api
      .serveEngines()
      .then((r) => {
        if (live && r.engines?.length) setEngines(r.engines);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, []);
  return engines;
}

/** True when `engine` turns `field` into a flag (unknown engine list: show everything). */
export function engineHas(engine: ServeEngine | undefined, field: string): boolean {
  return !engine || engine.fields.length === 0 || engine.fields.includes(field);
}

/** The subset of `fields` the engine translates: what Start sends (a hidden field never is). */
export function engineFields(engine: ServeEngine | undefined, fields: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(fields).filter(([k]) => engineHas(engine, k)));
}

export function EngineSelect({
  engines,
  value,
  onChange,
  disabled,
}: {
  engines: ServeEngine[];
  value: string;
  onChange: (name: string) => void;
  disabled?: boolean;
}) {
  const current = engines.find((e) => e.name === value);
  return (
    <Field label="Engine" htmlFor="serve-engine">
      <select
        id="serve-engine"
        className={cn(inputCls, "font-mono")}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        title={current?.notes || undefined}
        aria-describedby="serve-engine-summary"
      >
        {engines.map((e) => (
          <option key={e.name} value={e.name}>
            {e.label}
          </option>
        ))}
      </select>
    </Field>
  );
}

/** One line under the target row: the selected engine's image, port, TP reach and notes. */
export function EngineSummary({ engine }: { engine: ServeEngine | undefined }) {
  if (!engine) return null;
  const parts = [
    engine.default_image && `image ${engine.default_image}`,
    `port ${engine.default_port}`,
    engine.max_tp === 1 ? "single node" : engine.max_tp ? `TP ≤ ${engine.max_tp}` : "TP across Sparks",
  ].filter(Boolean);
  return (
    <p id="serve-engine-summary" className="break-words font-mono text-[10.5px] leading-relaxed text-lab-muted">
      {engine.label} · {parts.join(" · ")}
      {engine.notes ? <span className="block font-sans text-[11px]">{engine.notes}</span> : null}
    </p>
  );
}
