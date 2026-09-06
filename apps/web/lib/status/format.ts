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
