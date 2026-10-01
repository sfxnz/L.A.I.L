import { api, type LabStatus } from "./api";
import { serveHealthy } from "./lab-status-store";
import { copyText } from "./streams/clipboard";
import { CHORD_TARGETS, pickTheme, toggleWall } from "./shortcuts";
import type { ThemeChoice } from "./theme";

/**
 * Command registry for the ⌘K palette. Every command prints its shortcut so
 * the palette teaches until it is obsolete (brief §1.2 #2, §5.1).
 * Pure data + a tiny fuzzy filter; the UI lives in components/command.
 */

export type CommandGroup = "Navigate" | "Run" | "Copy" | "Theme" | "View" | "Help";

export const COMMAND_GROUPS: readonly CommandGroup[] = ["Navigate", "Run", "Copy", "Theme", "View", "Help"];

export type CommandContext = {
  router: { push: (href: string) => void; replace: (href: string) => void };
  pathname: string;
  status: LabStatus | null;
  needToken: boolean;
  /** Strand counts of the live run (from the store's `liveRun`); zeros when idle. */
  strands: { running: number; waiting: number };
  theme: { setChoice: (c: ThemeChoice) => void };
  toast: (message: string) => void;
  openSheet: () => void;
};

export type Command = {
  id: string;
  group: CommandGroup;
  label: string;
  hint?: string;
  /** Display form, e.g. ["⌘","K"] or ["g","b"]. */
  shortcut?: string[];
  when?: (ctx: CommandContext) => boolean;
  run: (ctx: CommandContext) => void | Promise<void>;
};

const NAV_EXTRA = [
  { href: "/connect", label: "Connect" },
  { href: "/usage", label: "Usage" },
  { href: "/lab", label: "Lab gallery" },
];

/** Bench query params are the page's own (`lib/bench/levels.ts` decodeConfigToQuery). */
export const BENCH_DECODE_HREF = "/bench?tab=decode&pack=prose&levels=1,2,4&tokens=512";
export const BENCH_PREFILL_HREF = "/bench?tab=prefill&sizes=8192,16384,32768,65536,131072";

function servedModel(status: LabStatus | null): string | null {
  const id = status?.serve?.model_id;
  return serveHealthy(status) && id && id !== "auto" && id !== "default" ? id : null;
}

function endpointUrl(status: LabStatus | null): string | null {
  return status?.serve?.base_url || status?.openAiBase || null;
}

async function startLoad(ctx: CommandContext, n: number, maxTokens: number) {
  try {
    const { run_id } = await api.startStreamRun({ mode: "load", pack: "prose", n, max_tokens: maxTokens });
    ctx.router.push(`/streams?run=${encodeURIComponent(run_id)}`);
  } catch (e) {
    ctx.toast(`Could not start: ${e instanceof Error ? e.message : String(e)}`);
  }
}

export const COMMANDS: Command[] = [
  // Navigate — the six nav rooms carry their `g` chord.
  ...Object.entries(CHORD_TARGETS).map<Command>(([key, t]) => ({
    id: `nav:${t.href}`,
    group: "Navigate",
    label: t.label,
    hint: t.href,
    shortcut: ["g", key],
    run: (ctx) => ctx.router.push(t.href),
  })),
  ...NAV_EXTRA.map<Command>((t) => ({
    id: `nav:${t.href}`,
    group: "Navigate",
    label: t.label,
    hint: t.href,
    run: (ctx) => ctx.router.push(t.href),
  })),

  // Run
  {
    id: "run:bench-decode",
    group: "Run",
    label: "Bench: decode ×1 ×2 ×4 prose",
    hint: "512 tokens per stream, then press r",
    when: (ctx) => serveHealthy(ctx.status),
    run: (ctx) => ctx.router.push(BENCH_DECODE_HREF),
  },
  {
    id: "run:bench-prefill",
    group: "Run",
    label: "Bench: prefill 8k–128k",
    hint: "opens the prefill tab configured, then press r",
    when: (ctx) => serveHealthy(ctx.status),
    run: (ctx) => ctx.router.push(BENCH_PREFILL_HREF),
  },
  {
    id: "run:streams-prose-8",
    group: "Run",
    label: "Streams: load prose ×8",
    hint: "starts on the default endpoint · 256 tokens",
    when: (ctx) => serveHealthy(ctx.status) && ctx.strands.running === 0,
    run: (ctx) => startLoad(ctx, 8, 256),
  },
  {
    id: "run:streams-smoke",
    group: "Run",
    label: "Streams: smoke ×2 (48 tokens)",
    hint: "tiny load to prove the endpoint answers",
    when: (ctx) => serveHealthy(ctx.status) && ctx.strands.running === 0,
    run: (ctx) => startLoad(ctx, 2, 48),
  },
  {
    id: "run:stop",
    group: "Run",
    label: "Stop current run",
    hint: "cancels every live streams / bench run",
    shortcut: ["⌘", "."],
    when: (ctx) => ctx.strands.running > 0,
    run: async (ctx) => {
      const rows = await api.listStreamRuns().catch(() => []);
      const live = rows.filter((r) => r.status === "running");
      await Promise.all(live.map((r) => api.stopStreamRun(r.run_id).catch(() => null)));
      ctx.toast(live.length ? `Stopped ${live.length} run${live.length === 1 ? "" : "s"}` : "No live run");
    },
  },

  // Copy
  {
    id: "copy:endpoint",
    group: "Copy",
    label: "Copy endpoint URL",
    hint: "the served OpenAI-compatible base URL",
    when: (ctx) => !!endpointUrl(ctx.status),
    run: async (ctx) => {
      const url = endpointUrl(ctx.status)!;
      ctx.toast((await copyText(url)) ? `Copied ${url}` : "Clipboard unavailable");
    },
  },
  {
    id: "copy:model",
    group: "Copy",
    label: "Copy model id",
    when: (ctx) => !!servedModel(ctx.status),
    run: async (ctx) => {
      const id = servedModel(ctx.status)!;
      ctx.toast((await copyText(id)) ? `Copied ${id}` : "Clipboard unavailable");
    },
  },

  // Theme
  {
    id: "theme:light",
    group: "Theme",
    label: "White Room",
    hint: "light — reconstruction",
    shortcut: ["⌘", "⇧", "L"],
    run: (ctx) => ctx.theme.setChoice("light"),
  },
  {
    id: "theme:dark",
    group: "Theme",
    label: "Helix",
    hint: "dark — in simulation",
    shortcut: ["⌘", "⇧", "L"],
    run: (ctx) => ctx.theme.setChoice("dark"),
  },
  {
    id: "theme:system",
    group: "Theme",
    label: "System",
    hint: "follow the OS",
    shortcut: ["⌘", "⇧", "L"],
    run: (ctx) => ctx.theme.setChoice("system"),
  },

  // View
  {
    id: "view:wall",
    group: "View",
    label: "Wall mode",
    hint: "read-only, chrome hidden — esc exits",
    shortcut: ["⌘", "⇧", "W"],
    when: (ctx) => ctx.pathname.startsWith("/streams"),
    run: (ctx) => toggleWall(ctx.pathname, ctx.router.replace, true),
  },
  {
    id: "view:details",
    group: "View",
    label: "Toggle details",
    hint: "raw table under the bench result",
    shortcut: ["⌘", "/"],
    when: (ctx) => ctx.pathname.startsWith("/bench"),
    // The page owns this key; hand it the same event it already listens for.
    run: () => window.dispatchEvent(new KeyboardEvent("keydown", { key: "/", metaKey: true, bubbles: true })),
  },

  // Help
  {
    id: "help:shortcuts",
    group: "Help",
    label: "Keyboard shortcuts",
    shortcut: ["?"],
    run: (ctx) => ctx.openSheet(),
  },
  {
    id: "help:token",
    group: "Help",
    label: "Paste LAIL_TOKEN",
    hint: "focus the banner — 401 is not an outage",
    when: (ctx) => ctx.needToken,
    run: () => document.querySelector<HTMLInputElement>('input[aria-label="LAIL_TOKEN"]')?.focus(),
  },
];

/** Default theme setter for the context — routes through the ThemeToggle. */
export const themeSetter = { setChoice: pickTheme };

export function visibleCommands(cmds: readonly Command[], ctx: CommandContext): Command[] {
  return cmds.filter((c) => !c.when || c.when(ctx));
}

function norm(s: string): string {
  return s.toLowerCase().replace(/[×x]/g, "x").replace(/\s+/g, " ").trim();
}

/**
 * Fuzzy score over label + hint + group: 0 = no match. Prefix on the label
 * wins, then a word prefix, then substring, then an in-order subsequence.
 */
export type Searchable = { label: string; hint?: string; group: string };

export function scoreCommand(c: Searchable, query: string): number {
  const q = norm(query);
  if (!q) return 1;
  const label = norm(c.label);
  if (label.startsWith(q)) return 5;
  if (label.split(/[\s:]+/).some((w) => w.startsWith(q))) return 4;
  const hay = `${label} ${norm(c.hint ?? "")} ${norm(c.group)}`;
  if (hay.includes(q)) return 3;
  let i = 0;
  for (const ch of hay) if (ch === q[i]) i++;
  return i === q.length ? 1 : 0;
}

export function filterCommands<T extends Searchable>(cmds: readonly T[], query: string): T[] {
  return cmds
    .map((c, i) => ({ c, i, s: scoreCommand(c, query) }))
    .filter((x) => x.s > 0)
    .sort((a, b) => b.s - a.s || a.i - b.i)
    .map((x) => x.c);
}

/** Group in registry order, dropping empty groups. */
export function groupCommands(cmds: readonly Command[]): Array<[CommandGroup, Command[]]> {
  return COMMAND_GROUPS.map<[CommandGroup, Command[]]>((g) => [g, cmds.filter((c) => c.group === g)]).filter(
    ([, list]) => list.length > 0,
  );
}
