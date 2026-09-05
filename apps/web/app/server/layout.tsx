import type { Metadata } from "next";

export const metadata: Metadata = { title: "Serve" };

export default function ServeLayout({ children }: { children: React.ReactNode }) {
  return children;
}
