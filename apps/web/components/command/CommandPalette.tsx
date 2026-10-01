"use client";

import { Command } from "cmdk";
import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type RunRow } from "@/lib/api";
import {
  COMMANDS,
  filterCommands,
  groupCommands,
  themeSetter,
  visibleCommands,
  type Command as Cmd,
  type CommandContext,
} from "@/lib/commands";
import { useLabStatusStore } from "@/lib/lab-status-store";
import { runHref } from "@/lib/run-href";
import { useCommandUi } from "@/lib/shortcuts";
import { Eyebrow } from "@/components/ui";
import { Keys } from "./Keys";
import { ShortcutSheet } from "./ShortcutSheet";
import "./command.css";

const RUNS_DEBOUNCE_MS = 200;
const TOAST_MS = 2200;

function Row({ label, hint, shortcut }: { label: string; hint?: string; shortcut?: readonly string[] }) {
  return (
    <>
      <span aria-hidden className="cmd-item-notch animus-notch" />
      <span aria-hidden className="cmd-item-bar" />
      <span aria-hidden className="cmd-glyph" />
      <span className="cmd-label">
        <span>{label}</span>
        {hint && <span className="cmd-hint">{hint}</span>}
      </span>
      {shortcut && <Keys keys={shortcut} />}
    </>
  );
}

/**
 * ⌘K — one palette for navigation, actions and run search. Every row prints
 * its shortcut so the palette teaches until it is obsolete. Filtering is ours
 * (lib/commands.ts) so the same scorer is unit-tested; cmdk owns focus,
 * arrow keys, Enter and Esc.
 */
export function CommandPalette() {
  const router = useRouter();
  const pathname = usePathname() || "/";
  const open = useCommandUi((s) => s.paletteOpen);
  const setPalette = useCommandUi((s) => s.setPalette);
  const setSheet = useCommandUi((s) => s.setSheet);
  const { status, needToken, liveRun } = useLabStatusStore();
  // `liveRun` is published by useStreamRun while a Streams/Bench run streams.
  const strands = useMemo(
    () => ({ running: liveRun?.running ?? 0, waiting: liveRun?.waiting ?? 0 }),
    [liveRun],
  );

  const [query, setQuery] = useState("");
  const [runs, setRuns] = useState<RunRow[] | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const toastTimer = useRef<number | null>(null);

  const showToast = useCallback((message: string) => {
    setToast(message);
    if (toastTimer.current !== null) window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(null), TOAST_MS);
  }, []);
  useEffect(() => () => {
    if (toastTimer.current !== null) window.clearTimeout(toastTimer.current);
  }, []);

  // Reset the query each time the palette opens.
  useEffect(() => {
    if (open) setQuery("");
  }, [open]);

  // Sequences: fetch the recent run index once per open, debounced behind the first keystroke.
  useEffect(() => {
    if (!open || runs !== null || query.trim().length < 2) return;
    const t = window.setTimeout(() => {
      api
        .runs({ limit: 20 })
        .then(setRuns)
        .catch(() => setRuns([]));
    }, RUNS_DEBOUNCE_MS);
    return () => window.clearTimeout(t);
  }, [open, query, runs]);
  useEffect(() => {
    if (!open) setRuns(null);
  }, [open]);

  const ctx = useMemo<CommandContext>(
    () => ({
      router: { push: router.push, replace: router.replace },
      pathname,
      status,
      needToken,
      strands,
      theme: themeSetter,
      toast: showToast,
      openSheet: () => setSheet(true),
    }),
    [router, pathname, status, needToken, strands, showToast, setSheet],
  );

  const groups = useMemo(() => groupCommands(filterCommands(visibleCommands(COMMANDS, ctx), query)), [ctx, query]);
  const sequences = useMemo(() => {
    if (!runs || query.trim().length < 2) return [];
    // Only runs with a view to open: legacy_* and other kinds have none.
    const rows = runs.flatMap((r) => {
      const href = runHref(r);
      return href
        ? [{ run: r, href, label: r.run_id, hint: [r.kind, r.model_id?.split("/").pop()].filter(Boolean).join(" · "), group: "Sequences" }]
        : [];
    });
    return filterCommands(rows, query).slice(0, 8);
  }, [runs, query]);

  const execute = useCallback(
    (cmd: Cmd) => {
      setPalette(false);
      // Let the dialog release focus before the command touches the page.
      window.setTimeout(() => void cmd.run(ctx), 0);
    },
    [ctx, setPalette],
  );

  const empty = groups.length === 0 && sequences.length === 0;

  return (
    <>
      <Command.Dialog
        open={open}
        onOpenChange={setPalette}
        label="Command palette"
        shouldFilter={false}
        loop
        overlayClassName="cmd-overlay"
        contentClassName="cmd-surface animus-bracketed"
      >
        <div className="cmd-input-row">
          <Eyebrow className="shrink-0 tracking-[0.18em]">⌘K</Eyebrow>
          <Command.Input
            className="cmd-input"
            value={query}
            onValueChange={setQuery}
            placeholder="Go to, run, copy, theme… or a run id"
            autoFocus
          />
        </div>
        <Command.List className="cmd-list">
          {empty && <Command.Empty className="cmd-empty">Nothing matches “{query}”.</Command.Empty>}
          {groups.map(([group, list]) => (
            <Command.Group key={group} heading={<Eyebrow>{group}</Eyebrow>}>
              {list.map((c) => (
                <Command.Item key={c.id} value={c.id} className="cmd-item" onSelect={() => execute(c)}>
                  <Row label={c.label} hint={c.hint} shortcut={c.shortcut} />
                </Command.Item>
              ))}
            </Command.Group>
          ))}
          {sequences.length > 0 && (
            <Command.Group heading={<Eyebrow>Sequences</Eyebrow>}>
              {sequences.map((s) => (
                <Command.Item
                  key={s.run.run_id}
                  value={`run:${s.run.run_id}`}
                  className="cmd-item"
                  onSelect={() => {
                    setPalette(false);
                    router.push(s.href);
                  }}
                >
                  <Row label={s.label} hint={s.hint} />
                </Command.Item>
              ))}
            </Command.Group>
          )}
        </Command.List>
        <div className="cmd-foot">
          <span>
            <kbd className="cmd-kbd">↑↓</kbd>move
          </span>
          <span>
            <kbd className="cmd-kbd">↩</kbd>run
          </span>
          <span>
            <kbd className="cmd-kbd">esc</kbd>close
          </span>
          <span className="ml-auto">
            <kbd className="cmd-kbd">?</kbd>all shortcuts
          </span>
        </div>
      </Command.Dialog>
      <ShortcutSheet />
      {toast && (
        <div className="cmd-toast" role="status">
          {toast}
        </div>
      )}
    </>
  );
}
