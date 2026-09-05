"use client";

import { Eyebrow, Panel } from "@/components/ui";

/** Phase B1 fills this room: Strands — the live concurrent-streams view. */
export default function StreamsPage() {
  return (
    <div className="lab-fade-in space-y-4">
      <div className="page-header">
        <div className="min-w-0">
          <h1 className="page-title">Streams</h1>
          <p className="page-sub">
            Every strand on the endpoint, live: text, TTFT, tok/s, and the aggregate they add up to.
          </p>
        </div>
      </div>
      <Panel title="Strands" padded>
        <Eyebrow>Compiling</Eyebrow>
        <p className="mt-2 text-[13px] leading-relaxed text-lab-muted">
          Streams instrument compiling — the Helix grid and its aggregate sparkline land here.
        </p>
      </Panel>
    </div>
  );
}
