"use client";

import type { ContextMention } from "@/lib/api";

function mentionLabel(m: ContextMention): string {
  if (m.type === "search") return `@search ${m.query}`;
  return `@${m.type} ${m.path}`;
}

export function ContextChips({
  mentions,
  openTabCount,
}: {
  mentions: ContextMention[];
  openTabCount: number;
}) {
  if (mentions.length === 0 && openTabCount === 0) return null;

  return (
    <div
      className="mb-1.5 flex flex-wrap items-center gap-1.5"
      data-testid="context-chips"
      aria-label="Context chips"
    >
      {openTabCount > 0 && (
        <span className="rounded-full border border-[#2a2a2a] bg-[#1a1a1a] px-2 py-0.5 text-[10px] text-[#888]">
          {openTabCount} open tab{openTabCount === 1 ? "" : "s"}
        </span>
      )}
      {mentions.map((m) => (
        <span
          key={m.type === "search" ? `search:${m.query}` : `${m.type}:${m.path}`}
          className="max-w-[220px] truncate rounded-full border border-[#2a3a2a] bg-[#152015] px-2 py-0.5 font-mono text-[10px] text-[#8fbcbb]"
          title={mentionLabel(m)}
        >
          {mentionLabel(m)}
        </span>
      ))}
    </div>
  );
}
