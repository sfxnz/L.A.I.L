/**
 * User-visible chrome contract from L.A.I.L inspo screenshots.
 * Imported by AppShell / Workbench so label checks bind to shipped UI.
 */

export const SIDEBAR_SECTION_LABELS = [
  "Search",
  "Workspace",
  "Pinned",
  "Tasks",
  "Projects",
] as const;

export const WORKSPACE_NAV_LABELS = [
  "Status",
  "Workbench",
  "Models",
  "Configure",
  "Usage",
  "Integrations",
  "Server",
] as const;

export const WORKSPACE_NAV = [
  { href: "/status", label: "Status" as const },
  { href: "/workbench", label: "Workbench" as const },
  { href: "/models", label: "Models" as const },
  { href: "/configure", label: "Configure" as const },
  { href: "/usage", label: "Usage" as const },
  { href: "/integrations", label: "Integrations" as const },
  { href: "/server", label: "Server" as const },
];

export const COMPOSER_PLACEHOLDER = "Ask for follow-up changes";

export const AGENT_MODES = ["plan", "ask", "agent"] as const;
export const AGENT_MODE_LABELS = { plan: "Plan", ask: "Ask", agent: "Agent" } as const;

export const STREAM_MARKERS = {
  thought: "Thought",
  working: "Working",
  ran: "Ran",
  creating: "Creating",
  status: "Status",
  proposed: "Proposed",
} as const;

export type TimelineKind =
  | "thought"
  | "status"
  | "tool"
  | "file"
  | "assistant"
  | "error"
  | "ran"
  | "patch";

export type TimelineInput = {
  kind: TimelineKind | string;
  text: string;
  meta?: Record<string, unknown>;
};

export type StreamBlock =
  | { type: "ran"; count: number; detail: string; output?: string }
  | { type: "thought"; text: string }
  | { type: "status"; text: string }
  | { type: "file"; path: string; creating?: boolean }
  | { type: "patch"; path: string }
  | { type: "error"; text: string }
  | { type: "assistant"; text: string };

/**
 * Map agent timeline events → inspo-style stream blocks (Thought / Ran N / Creating).
 *
 * Counting: only `kind === "ran"` (tool_end) increments the command count.
 * `kind === "tool"` is tool_start — kept for optional live UI but does not count
 * a finished command (otherwise tool_start+tool_end would show "Ran 2 commands").
 */
export function groupTimeline(items: TimelineInput[]): StreamBlock[] {
  const out: StreamBlock[] = [];
  let ranBuf: { count: number; detail: string; output?: string } | null = null;

  const flushRan = () => {
    if (ranBuf && ranBuf.count > 0) {
      out.push({ type: "ran", ...ranBuf });
    }
    ranBuf = null;
  };

  for (const it of items) {
    // tool_start: ignore for Ran N aggregation (live "working" can use status)
    if (it.kind === "tool") {
      continue;
    }
    if (it.kind === "ran") {
      if (!ranBuf) ranBuf = { count: 0, detail: it.text };
      ranBuf.count += 1;
      ranBuf.detail = it.text;
      if (it.meta?.output != null) ranBuf.output = String(it.meta.output);
      continue;
    }
    flushRan();
    if (it.kind === "thought") out.push({ type: "thought", text: it.text });
    else if (it.kind === "status") out.push({ type: "status", text: it.text });
    else if (it.kind === "file")
      out.push({ type: "file", path: it.text, creating: !!it.meta?.creating });
    else if (it.kind === "patch") out.push({ type: "patch", path: it.text });
    else if (it.kind === "error") out.push({ type: "error", text: it.text });
    // assistant is rendered from messages[], not the stream (avoids duplicate bubbles)
  }
  flushRan();
  return out;
}

/** User-visible label for a ran-commands block (e.g. "Ran 2 commands"). */
export function ranLabel(count: number, detail?: string): string {
  const base = `Ran ${count} command${count === 1 ? "" : "s"}`;
  return detail ? `${base} · ${detail}` : base;
}

/** User-visible Creating/Wrote label for file events. */
export function fileWriteLabel(path: string, creating?: boolean): string {
  return `${creating ? STREAM_MARKERS.creating : "Wrote"} ${path}`;
}
