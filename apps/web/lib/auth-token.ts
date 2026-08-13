const KEY = "lail_token";

function store(): Storage | null {
  try {
    if (typeof sessionStorage === "undefined") return null;
    return sessionStorage;
  } catch {
    return null;
  }
}

export function getClientToken(): string {
  try {
    return (store()?.getItem(KEY) || "").trim();
  } catch {
    return "";
  }
}

export function setClientToken(token: string): void {
  const s = store();
  if (!s) return;
  try {
    const t = token.trim();
    if (t) s.setItem(KEY, t);
    else s.removeItem(KEY);
  } catch {
    /* */
  }
}

export function tokenQuery(url: string): string {
  const t = getClientToken();
  if (!t) return url;
  return url + (url.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(t);
}
