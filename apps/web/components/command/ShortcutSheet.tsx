"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { usePathname } from "next/navigation";
import { Eyebrow } from "@/components/ui";
import { GLOBAL_SHORTCUTS, routeShortcuts, useCommandUi, type ShortcutRow } from "@/lib/shortcuts";
import { Keys } from "./Keys";
import "./command.css";

function Column({ title, rows }: { title: string; rows: ShortcutRow[] }) {
  return (
    <div className="cmd-sheet-col">
      <h3 className="flex items-center gap-2">
        <span aria-hidden className="h-3 w-px bg-lab-accent" />
        <Eyebrow>{title}</Eyebrow>
      </h3>
      {rows.map((r) => (
        <div key={r.label} className="cmd-sheet-row">
          <span>{r.label}</span>
          <Keys keys={r.keys} />
        </div>
      ))}
    </div>
  );
}

/** `?` — two columns: global chords and the current route's own keys. */
export function ShortcutSheet() {
  const pathname = usePathname() || "/";
  const open = useCommandUi((s) => s.sheetOpen);
  const setSheet = useCommandUi((s) => s.setSheet);
  const route = routeShortcuts(pathname);

  return (
    <Dialog.Root open={open} onOpenChange={setSheet}>
      <Dialog.Portal>
        <Dialog.Overlay className="cmd-overlay" />
        <Dialog.Content className="cmd-surface cmd-sheet animus-bracketed" aria-describedby={undefined}>
          <div className="cmd-input-row">
            <Dialog.Title className="font-[family-name:var(--font-display)] text-[13px] font-semibold uppercase tracking-[0.14em] text-lab-text">
              Keyboard shortcuts
            </Dialog.Title>
            <span className="ml-auto text-[11px] text-lab-muted">macOS keys · ⌘K lists every command</span>
            <Dialog.Close
              className="cmd-sheet-close flex h-6 w-6 items-center justify-center text-lab-muted hover:text-lab-text"
              aria-label="Close"
            >
              <X className="h-3.5 w-3.5" strokeWidth={1.75} />
            </Dialog.Close>
          </div>
          <div className="cmd-sheet-body">
            <Column title="Everywhere" rows={GLOBAL_SHORTCUTS} />
            <Column
              title={route ? `This page — ${route.title}` : "This page"}
              rows={route?.rows ?? [{ keys: ["⌘", "K"], label: "No page keys here — open the palette" }]}
            />
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
