import type { Metadata } from "next";
import { AppShell } from "@/components/layout/AppShell";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "L.A.I.L — Serve & Evals",
    template: "%s · L.A.I.L",
  },
  description: "Local AI Lab on DGX Spark: serve models, run evals, connect Hermes",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full overflow-hidden antialiased">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
