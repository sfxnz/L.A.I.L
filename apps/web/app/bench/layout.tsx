import type { Metadata } from "next";

export const metadata: Metadata = { title: "Bench" };

export default function BenchLayout({ children }: { children: React.ReactNode }) {
  return children;
}
