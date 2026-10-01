/** Formatting shared by the Status cards. Numbers render inside `lab-num`. */

/** 4523 s → "1 h 15 m"; 95 → "1 m 35 s"; 3 d 2 h. */
export function fmtUptime(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s) || s < 0) return "";
  const sec = Math.floor(s);
  if (sec < 60) return `${sec} s`;
  const m = Math.floor(sec / 60);
  if (m < 60) return `${m} m ${sec % 60} s`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h} h ${m % 60} m`;
  const d = Math.floor(h / 24);
  return `${d} d ${h % 24} h`;
}

export function fmtGib(v: number | null | undefined, digits = 1): string {
  if (v == null || !Number.isFinite(v)) return "";
  return `${v.toFixed(digits).replace(/\.0$/, "")} GiB`;
}

export function fmtRate(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "";
  return v >= 100 ? String(Math.round(v)) : v.toFixed(1);
}

export function fmtTemp(v: number): string {
  return `${Math.round(v)}°C`;
}

export function fmtPct(v: number): string {
  return `${Math.round(v)}%`;
}

export function fmtWatts(v: number): string {
  return v >= 100 ? String(Math.round(v)) : v.toFixed(1);
}

/** Bytes per second as the link speaks it: "0 B/s", "812 KB/s", "118 MB/s", "2.4 GB/s". */
export function fmtBytesRate(bps: number): string {
  if (!Number.isFinite(bps) || bps <= 0) return "0 B/s";
  if (bps >= 1e9) return `${(bps / 1e9).toFixed(bps >= 1e10 ? 0 : 1)} GB/s`;
  if (bps >= 1e6) return `${Math.round(bps / 1e6)} MB/s`;
  if (bps >= 1e3) return `${Math.round(bps / 1e3)} KB/s`;
  return `${Math.round(bps)} B/s`;
}

/** Age of a reading: "now" under a second, then "4 s", "2 m 05 s"-style via fmtUptime. */
export function fmtAge(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return "";
  if (s < 1) return "now";
  return fmtUptime(s);
}

/** Display name of an engine / backend key. */
export function engineLabel(key: string | null | undefined): string {
  const k = (key || "").toLowerCase();
  if (k === "vllm") return "vLLM";
  if (k === "llamacpp") return "llama.cpp";
  if (k === "sglang") return "SGLang";
  return key || "";
}
