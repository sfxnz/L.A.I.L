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
  // The dev rewrite proxy honours the browser's `Accept-Encoding: gzip` and
  // compresses proxied responses — including `text/event-stream`. gzip buffers,
  // so an SSE connection delivers ZERO bytes to the browser until the stream
  // closes: the Serve job dock sits on "running · 0 log bytes" for an entire
  // 15-30 min model load, while curl (no Accept-Encoding by default) streams
  // the same URL fine. Next has no per-route compression switch, so turn the
  // dev proxy's compression off; the controller never gzips these itself.
  compress: false,
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
