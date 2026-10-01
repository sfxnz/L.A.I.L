/** Telemetry formatting for the Streams instrument. Numbers render in `lab-num` (tabular). */

import { fmtRate as fmtRateOrEmpty } from "../status/format";

/** The console's one tok/s format; "—" where there is no value. */
export function fmtRate(v: number | null | undefined): string {
  return fmtRateOrEmpty(v) || "—";
}

export function fmtInt(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return Math.round(v).toLocaleString("en-US");
}

/** Milliseconds: "412 ms" below a second, "1.20 s" above. */
export function fmtMs(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (v < 1000) return `${Math.round(v)} ms`;
  return `${(v / 1000).toFixed(2)} s`;
}

/** Run duration: "48.1 s" under a minute, "1 m 17 s" above. */
export function fmtDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "—";
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  return `${m} m ${Math.round(s - m * 60)} s`;
}

export function fmtPct(share: number | null | undefined): string {
  if (share === null || share === undefined || !Number.isFinite(share)) return "—";
  return `${Math.round(share * 100)} %`;
}

/** "org/model" → "model". */
export function shortModel(id: string | null | undefined): string {
  return (id || "").split("/").pop() || "—";
}
