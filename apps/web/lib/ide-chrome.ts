/**
 * L.A.I.L console chrome — serve + evals (agent work lives in Hermes).
 */

export const SIDEBAR_SECTION_LABELS = ["Lab"] as const;

export const WORKSPACE_NAV_LABELS = [
  "Status",
  "Serve",
  "Evals",
  "Configure",
] as const;

export const WORKSPACE_NAV = [
  { href: "/status", label: "Status" as const },
  { href: "/server", label: "Serve" as const },
  { href: "/evals", label: "Evals" as const },
  { href: "/configure", label: "Configure" as const },
];

/** @deprecated Composer retired from primary nav — agent = Hermes */
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
    if (it.kind === "tool") continue;
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
  }
  flushRan();
  return out;
}

export function ranLabel(count: number, detail?: string): string {
  const base = `Ran ${count} command${count === 1 ? "" : "s"}`;
  return detail ? `${base} · ${detail}` : base;
}

export function fileWriteLabel(path: string, creating?: boolean): string {
  return `${creating ? STREAM_MARKERS.creating : "Wrote"} ${path}`;
}
