import type { NextConfig } from "next";

const api = process.env.LAIL_API_URL || process.env.NEXT_PUBLIC_LAIL_API || "http://127.0.0.1:8787";

const nextConfig: NextConfig = {
  transpilePackages: ["@lail/shared"],
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${api}/api/:path*` },
      { source: "/v1/:path*", destination: `${api}/v1/:path*` },
    ];
  },
};

export default nextConfig;
