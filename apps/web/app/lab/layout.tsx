import type { Metadata } from "next";

export const metadata: Metadata = { title: "Lab" };

export default function LabLayout({ children }: { children: React.ReactNode }) {
  return children;
}
