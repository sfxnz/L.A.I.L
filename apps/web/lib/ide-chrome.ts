/**
 * L.A.I.L console chrome — serve + evals (agent work lives in Hermes).
 */

export const WORKSPACE_NAV = [
  { href: "/status", label: "Status" as const },
  { href: "/server", label: "Serve" as const },
  { href: "/bench", label: "Bench" as const },
  { href: "/streams", label: "Streams" as const },
  { href: "/evals", label: "Evals" as const },
  { href: "/configure", label: "Configure" as const },
];

/** Prose pages keep the 1152px measure; every instrument page gets 1440px. */
export const PROSE_ROUTES = ["/configure"] as const;

export function isProseRoute(pathname: string): boolean {
  return PROSE_ROUTES.some((p) => pathname === p || pathname.startsWith(p + "/"));
}
