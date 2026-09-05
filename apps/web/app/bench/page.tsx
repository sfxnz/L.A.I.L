"use client";

import { Eyebrow, Panel } from "@/components/ui";

/** Phase B2 fills this room: Decode · Prefill instrument. */
export default function BenchPage() {
  return (
    <div className="lab-fade-in space-y-4">
      <div className="page-header">
        <div className="min-w-0">
          <h1 className="page-title">Bench</h1>
          <p className="page-sub">
            Decode and prefill synchronization runs against the live endpoint, drawn as they happen.
          </p>
        </div>
      </div>
      <Panel title="Bench" padded>
        <Eyebrow>Compiling</Eyebrow>
        <p className="mt-2 text-[13px] leading-relaxed text-lab-muted">
          Bench instrument compiling — decode ×1…×32 and prefill 8k–256k land here.
        </p>
      </Panel>
    </div>
  );
}
