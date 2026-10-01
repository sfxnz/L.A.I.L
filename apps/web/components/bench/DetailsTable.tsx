"use client";

import type { BenchRange } from "@lail/shared";
import { fmtInt, fmtMs, fmtPct, fmtSize, fmtTokS } from "@/lib/bench/format";
import type { DecodeArm, PrefillArm } from "@/lib/bench/result";
import { Nil } from "@/components/ui";
import { cn } from "@/lib/utils";

/** Collapsed under "Details" (⌘/). Errors are rows, never a footnote. */

function Num({ v, fmt }: { v: number | null; fmt: (n: number) => string }) {
  return v === null ? <Nil word="None" /> : <>{fmt(v)}</>;
}

/** min–max over repeats, under the median. */
function Range({ r }: { r: BenchRange | null }) {
  return r ? <div className="text-[10px] text-lab-muted">{`${fmtTokS(r[0])}–${fmtTokS(r[1])}`}</div> : null;
}

/** Repeats behind the medians, and foreign requests on the server while the level ran. */
function Samples({ samples, foreign }: { samples: number | null; foreign: number | null }) {
  return (
    <>
      {samples ?? <Nil word="None" />}
      {foreign !== null && foreign > 0 && (
        <span className="ml-1.5 text-lab-warn" title={`${foreign} request(s) that were not this bench's ran on the server`}>
          contended
        </span>
      )}
    </>
  );
}

export function DecodeDetails({ arms, className }: { arms: DecodeArm[]; className?: string }) {
  return (
    <div className={cn("overflow-x-auto", className)}>
      <table className="lab-table text-[12px]">
        <thead>
          <tr>
            <th scope="col">Level</th>
            <th scope="col" title="Wall-clock, median over waves (min–max)">Aggregate</th>
            <th scope="col" title="Σ decoded tokens ÷ decode span — the live gauge's measure">Decode span</th>
            <th scope="col" title="Median per-strand decode rate = 1 / TPOT (min–max)">Per-stream</th>
            <th scope="col">TTFT p50</th>
            <th scope="col" title="Shown from 5 strands up">p95</th>
            <th scope="col" title="Shown from 5 strands up">p99</th>
            <th scope="col">TPOT</th>
            <th scope="col" title="vLLM: accepted ÷ drafted speculative tokens over the level">Spec accept</th>
            <th scope="col">Waves</th>
            <th scope="col">Strands</th>
            <th scope="col">Errors</th>
          </tr>
        </thead>
        <tbody>
          {arms.map((a) => (
            <tr key={a.concurrency} className={cn(a.ok < a.requests && "text-lab-danger")}>
              <td className="font-mono text-lab-text">×{a.concurrency}</td>
              <td className="font-mono text-lab-text">
                <Num v={a.aggregate} fmt={(n) => `${fmtTokS(n)} tok/s`} />
                <Range r={a.aggregateRange} />
              </td>
              <td className="font-mono">
                <Num v={a.steady} fmt={(n) => `${fmtTokS(n)} tok/s`} />
              </td>
              <td className="font-mono">
                <Num v={a.perStream} fmt={(n) => `${fmtTokS(n)} tok/s`} />
                <Range r={a.perStreamRange} />
              </td>
              <td className="font-mono">
                <Num v={a.ttftP50} fmt={fmtMs} />
              </td>
              <td className="font-mono">
                <Num v={a.ttftP95} fmt={fmtMs} />
              </td>
              <td className="font-mono">
                <Num v={a.ttftP99} fmt={fmtMs} />
              </td>
              <td className="font-mono">
                <Num v={a.tpotMs} fmt={(n) => `${n.toFixed(1)} ms`} />
              </td>
              <td className="font-mono">
                <Num v={a.server?.spec_acceptance ?? null} fmt={(n) => fmtPct(n)} />
              </td>
              <td className="font-mono text-lab-text-dim">
                <Samples samples={a.samples} foreign={a.foreignMax} />
              </td>
              <td className={cn("font-mono", a.ok < a.requests ? "text-lab-danger" : "text-lab-text-dim")}>
                {a.ok}/{a.requests}
              </td>
              <td className="max-w-[22rem] whitespace-normal font-mono text-[11px]">
                {a.errors.length ? a.errors.join("; ") : <Nil word="None" />}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function PrefillDetails({ arms, className }: { arms: PrefillArm[]; className?: string }) {
  return (
    <div className={cn("overflow-x-auto", className)}>
      <table className="lab-table text-[12px]">
        <thead>
          <tr>
            <th scope="col">Context</th>
            <th scope="col">Prompt tokens</th>
            <th scope="col" title="prompt tokens ÷ TTFT, median over requests (min–max)">Prefill</th>
            <th scope="col" title="vLLM: computed prefill tokens ÷ prefill time (no queueing, no cache hits)">Server prefill</th>
            <th scope="col">TTFT</th>
            <th scope="col">Requests</th>
            <th scope="col">Notes</th>
          </tr>
        </thead>
        <tbody>
          {arms.map((a) => (
            <tr key={a.size} className={cn(a.skipped && "bench-hatched opacity-75", !a.skipped && a.ok < a.requests && "text-lab-danger")}>
              <td className="font-mono text-lab-text">{fmtSize(a.size)}</td>
              <td className="font-mono">
                <Num v={a.promptTokens} fmt={fmtInt} />
              </td>
              <td className="font-mono text-lab-text">
                <Num v={a.prefillTokS} fmt={(n) => `${fmtTokS(n)} tok/s`} />
                <Range r={a.prefillRange} />
              </td>
              <td className="font-mono">
                <Num v={a.server?.prefill_tok_s ?? null} fmt={(n) => `${fmtTokS(n)} tok/s`} />
              </td>
              <td className="font-mono">
                <Num v={a.ttftMs} fmt={fmtMs} />
              </td>
              <td className="font-mono text-lab-text-dim">
                {a.ok}/{a.requests}
                {a.foreignMax !== null && a.foreignMax > 0 && <span className="ml-1.5 text-lab-warn">contended</span>}
              </td>
              <td className="max-w-[22rem] whitespace-normal font-mono text-[11px]">
                {a.skipped ? <span className="text-lab-muted">skipped: {a.skipped}</span> : a.errors.length ? a.errors.join("; ") : <Nil word="None" />}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
