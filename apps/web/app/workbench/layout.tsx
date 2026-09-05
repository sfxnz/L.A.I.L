import type { Metadata } from "next";

export const metadata: Metadata = { title: "Workbench" };

export default function WorkbenchLayout({ children }: { children: React.ReactNode }) {
  return children;
}
