import type { Metadata } from "next";
import { AppShell } from "@/components/layout/AppShell";
import "./globals.css";

export const metadata: Metadata = {
  title: "L.A.I.L — Local AI Lab",
  description: "Local-first AI lab: serve, compose, models, usage",
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
