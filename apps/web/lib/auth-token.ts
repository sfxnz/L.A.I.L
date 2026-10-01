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

/**
 * The controller's own token gate (HTTP 401 + {error:"unauthorized"}), from an
 * ApiError thrown by lib/api.ts. Decided by status, not message text: a 401 the
 * controller relays from elsewhere (e.g. the HF Hub) or a body that merely
 * mentions "401" is a real failure, not "paste your token".
 */
export function isUnauthorizedError(err: unknown): boolean {
  const e = err as { status?: unknown; json?: { error?: unknown } | null } | null;
  return e?.status === 401 && e.json?.error === "unauthorized";
}
