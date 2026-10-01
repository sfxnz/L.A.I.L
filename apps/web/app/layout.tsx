import type { Metadata } from "next";
import { Barlow_Condensed, JetBrains_Mono } from "next/font/google";
import { AppShell } from "@/components/layout/AppShell";
import { THEME_BOOT_SCRIPT } from "@/lib/theme";
import "./globals.css";

const animusDisplay = Barlow_Condensed({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-animus-display",
  display: "swap",
});

/* Every numeral, id, path and timestamp rides this face (→ --font-mono). */
const animusMono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-animus-mono",
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
    <html
      lang="en"
      className={`h-full ${animusDisplay.variable} ${animusMono.variable}`}
      suppressHydrationWarning
    >
      <head>
        {/* Resolve the theme before first paint — otherwise the wrong world flashes. */}
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOT_SCRIPT }} />
      </head>
      <body className="h-full overflow-hidden antialiased animus-vignette">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
