/** Host bind + token policy for the controller. */

export class BindPolicyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BindPolicyError";
  }
}

export function isLoopbackHost(host: string): boolean {
  const h = host.trim().toLowerCase();
  return h === "127.0.0.1" || h === "localhost" || h === "::1" || h === "[::1]";
}

export function assertSafeBind(opts: {
  host: string;
  token: string;
  allowInsecure?: boolean;
}): void {
  if (isLoopbackHost(opts.host)) return;
  if ((opts.token || "").trim()) return;
  if (opts.allowInsecure) return;
  throw new BindPolicyError(
    `LAIL_HOST=${opts.host} is off-loopback and LAIL_TOKEN is unset. ` +
      `Bind 127.0.0.1, or set LAIL_TOKEN, or (compose only) LAIL_INSECURE_BIND=1 ` +
      `when host ports are published on 127.0.0.1.`,
  );
}

export function tokenMatches(
  req: Request,
  token: string,
  opts?: { allowQuery?: boolean },
): boolean {
  const expected = (token || "").trim();
  if (!expected) return true;
  const auth = req.headers.get("authorization") || "";
  if (auth.toLowerCase().startsWith("bearer ") && auth.slice(7).trim() === expected) {
    return true;
  }
  if ((req.headers.get("x-lail-token") || "") === expected) return true;
  if (opts?.allowQuery) {
    try {
      const u = new URL(req.url);
      if (u.searchParams.get("token") === expected) return true;
    } catch {
      /* */
    }
  }
  return false;
}

/** Query-string tokens are only for transports that cannot set headers. */
export function allowQueryToken(pathname: string): boolean {
  return /^\/api\/jobs\/[^/]+\/logs$/.test(pathname) || /^\/api\/streams\/runs\/[^/]+\/events$/.test(pathname);
}

export function isPublicUnauthedPath(pathname: string, method: string): boolean {
  if (pathname === "/api/health") return true;
  if (method !== "GET" && method !== "HEAD") return false;
  return (
    pathname.startsWith("/api/lab/p/") ||
    pathname.startsWith("/api/lab/play/") ||
    pathname.startsWith("/api/lab/public/") ||
    pathname.startsWith("/p/")
  );
}

/**
 * Never reflect an unknown Origin. Only the configured web origins are trusted;
 * other loopback ports (dev apps, artifact servers, bridges) are not.
 */
export function resolveCorsOrigin(
  origin: string | undefined,
  allow: string[],
): string | undefined {
  if (!origin) return allow[0];
  return allow.includes(origin) ? origin : undefined;
}

/**
 * Cross-site write guard, independent of LAIL_TOKEN. A browser request from a
 * foreign origin can still *reach* a handler as a CORS "simple request"
 * (text/plain or form body, or no body) even though it cannot read the reply.
 * Such a write is refused unless it is JSON — application/json forces a
 * preflight, which only the configured web origins pass. curl, Hermes and other
 * non-browser clients send no Origin and are unaffected.
 */
export function isCrossSiteWrite(req: Request, allow: string[]): boolean {
  if (req.method === "GET" || req.method === "HEAD" || req.method === "OPTIONS") return false;
  const origin = req.headers.get("origin");
  if (!origin) return false;
  if (req.headers.get("sec-fetch-site") === "same-origin") return false;
  if (resolveCorsOrigin(origin, allow) === origin) return false;
  const ct = (req.headers.get("content-type") || "").split(";")[0].trim().toLowerCase();
  return ct !== "application/json";
}
