import type { Metadata } from "next";

/*
  Page titles are Next metadata, nothing else. A client `document.title = …`
  (the old AppShell PAGE_TITLES effect and lib/usePageTitle) is overwritten a
  few ms later when React commits the streamed root <title>, so the tab never
  showed the page name on first load. Metadata is SSR-correct and survives.
*/
export const metadata: Metadata = { title: "Status" };

export default function StatusLayout({ children }: { children: React.ReactNode }) {
  return children;
}
