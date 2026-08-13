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

export function tokenMatches(req: Request, token: string): boolean {
  const expected = (token || "").trim();
  if (!expected) return true;
  const auth = req.headers.get("authorization") || "";
  if (auth.toLowerCase().startsWith("bearer ") && auth.slice(7).trim() === expected) {
    return true;
  }
  if ((req.headers.get("x-lail-token") || "") === expected) return true;
  try {
    const u = new URL(req.url);
    if (u.searchParams.get("token") === expected) return true;
  } catch {
    /* */
  }
  return false;
}
