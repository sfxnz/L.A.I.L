"use client";

import { fmtInt, fmtMs, fmtSize, fmtTokS } from "@/lib/bench/format";
import type { DecodeArm, PrefillArm } from "@/lib/bench/result";
import { Nil } from "@/components/ui";
import { cn } from "@/lib/utils";

/** Collapsed under "Details" (⌘/). Errors are rows, never a footnote. */

function Num({ v, fmt }: { v: number | null; fmt: (n: number) => string }) {
  return v === null ? <Nil word="None" /> : <>{fmt(v)}</>;
}

export function DecodeDetails({ arms, className }: { arms: DecodeArm[]; className?: string }) {
  return (
    <div className={cn("overflow-x-auto", className)}>
      <table className="lab-table text-[12px]">
        <thead>
          <tr>
            <th scope="col">Level</th>
            <th scope="col">Aggregate</th>
            <th scope="col">Per-stream</th>
            <th scope="col">TTFT p50</th>
            <th scope="col">p95</th>
            <th scope="col">p99</th>
            <th scope="col">TPOT</th>
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
              </td>
              <td className="font-mono">
                <Num v={a.perStream} fmt={(n) => `${fmtTokS(n)} tok/s`} />
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
            <th scope="col">Prefill</th>
            <th scope="col">TTFT</th>
            <th scope="col">Strands</th>
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
              </td>
              <td className="font-mono">
                <Num v={a.ttftMs} fmt={fmtMs} />
              </td>
              <td className="font-mono text-lab-text-dim">
                {a.ok}/{a.requests}
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
