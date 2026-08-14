import type { Metadata } from "next";
import { Barlow_Condensed } from "next/font/google";
import { AppShell } from "@/components/layout/AppShell";
import { THEME_BOOT_SCRIPT } from "@/lib/theme";
import "./globals.css";

const animusDisplay = Barlow_Condensed({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-animus-display",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "L.A.I.L — Serve & Evals",
    template: "%s · L.A.I.L",
  },
  description: "Serve and eval any model on your own hardware.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`h-full ${animusDisplay.variable}`} suppressHydrationWarning>
      <head>
        {/* Resolve the theme before first paint — otherwise the wrong world flashes. */}
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOT_SCRIPT }} />
      </head>
      <body className="h-full overflow-hidden antialiased animus-grain animus-vignette animus-scanlines">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
