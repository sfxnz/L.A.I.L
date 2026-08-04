import type { NextConfig } from "next";

const api = process.env.LAIL_API_URL || process.env.NEXT_PUBLIC_LAIL_API || "http://127.0.0.1:8787";

// Next 16 blocks cross-origin access to dev resources (HMR / turbopack) unless listed.
// L.A.I.L is opened via localhost, 127.0.0.1, Tailscale IP, and hostname from Mac.
const allowedDevOrigins = [
  "127.0.0.1",
  "localhost",
  "0.0.0.0",
  "spark1",
  "spark1.home",
  "100.86.121.44",
  ...(process.env.LAIL_DEV_ORIGINS
    ? process.env.LAIL_DEV_ORIGINS.split(",").map((s) => s.trim()).filter(Boolean)
    : []),
];

const nextConfig: NextConfig = {
  transpilePackages: ["@lail/shared"],
  allowedDevOrigins,
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${api}/api/:path*` },
      { source: "/v1/:path*", destination: `${api}/v1/:path*` },
      // public artifact share (static only — proxied to controller)
      { source: "/p/:path*", destination: `${api}/p/:path*` },
    ];
  },
};

export default nextConfig;
