/** Number formatting for bench readouts — one rule per unit so every surface agrees. */

export function fmtTokS(v: number | null | undefined, digits?: number): string {
  if (v == null || !Number.isFinite(v)) return "";
  if (v >= 10_000) return `${(v / 1000).toFixed(1)}k`;
  if (v >= 1000) return `${(v / 1000).toFixed(2)}k`;
  if (digits != null) return v.toFixed(digits);
  return v >= 100 ? String(Math.round(v)) : v.toFixed(1);
}

/** ms under 1 s, seconds with two decimals above; never mixes units in one table. */
export function fmtMs(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return "";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function fmtDuration(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms) || ms < 0) return "";
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  return r ? `${m} m ${r} s` : `${m} m`;
}

/** 8192 → "8k", 262144 → "256k", 1500 → "1.5k", 512 → "512". */
export function fmtSize(tokens: number): string {
  if (tokens >= 1024 && tokens % 1024 === 0) return `${tokens / 1024}k`;
  if (tokens >= 1000) return `${(tokens / 1000).toFixed(1).replace(/\.0$/, "")}k`;
  return String(tokens);
}

export function fmtInt(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "";
  return new Intl.NumberFormat("en-US").format(Math.round(n));
}

export function fmtPct(frac: number | null | undefined, signed = false): string {
  if (frac == null || !Number.isFinite(frac)) return "";
  const pct = Math.round(frac * 100);
  return `${signed && pct > 0 ? "+" : ""}${pct} %`;
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const day = d.getDate();
  const mon = d.toLocaleString("en-GB", { month: "short" });
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${day} ${mon} ${hh}:${mm}`;
}

export function modelShort(id: string | null | undefined): string {
  return id?.split("/").pop() || "";
}
