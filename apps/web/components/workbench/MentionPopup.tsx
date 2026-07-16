"use client";

import { useEffect, useMemo, useState } from "react";
import type { TreeNode } from "@/lib/api";
import { cn } from "@/lib/utils";

export function flattenTreePaths(nodes: TreeNode[], out: string[] = []): string[] {
  for (const n of nodes) {
    if (n.type === "file") out.push(n.path);
    if (n.children?.length) flattenTreePaths(n.children, out);
  }
  return out;
}

/** Detect incomplete @token at end of text (for popup). Returns query after last bare @. */
export function mentionQueryAtCursor(text: string, cursor: number): { start: number; query: string } | null {
  const before = text.slice(0, cursor);
  const at = before.lastIndexOf("@");
  if (at < 0) return null;
  const afterAt = before.slice(at + 1);
  // Already a typed keyword mention with space — not incomplete bare @
  if (/^(file|folder|search|code)\s/i.test(afterAt)) return null;
  // No whitespace after @ (incomplete token)
  if (/\s/.test(afterAt)) return null;
  return { start: at, query: afterAt };
}

export function MentionPopup({
  open,
  query,
  paths,
  onSelect,
  onClose,
}: {
  open: boolean;
  query: string;
  paths: string[];
  onSelect: (path: string) => void;
  onClose: () => void;
}) {
  const [active, setActive] = useState(0);

  const filtered = useMemo(() => {
    const q = query.toLowerCase();
    const list = q
      ? paths.filter((p) => p.toLowerCase().includes(q))
      : paths;
    return list.slice(0, 12);
  }, [paths, query]);

  useEffect(() => {
    setActive(0);
  }, [query, open]);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActive((i) => Math.min(i + 1, Math.max(filtered.length - 1, 0)));
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setActive((i) => Math.max(i - 1, 0));
        return;
      }
      if (e.key === "Enter" && filtered[active]) {
        e.preventDefault();
        e.stopPropagation();
        onSelect(filtered[active]);
      }
    }
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open, filtered, active, onSelect, onClose]);

  if (!open || filtered.length === 0) return null;

  return (
    <div
      className="absolute bottom-full left-0 z-20 mb-1 max-h-48 w-full max-w-md overflow-auto rounded-lg border border-[#2a2a2a] bg-[#1a1a1a] py-1 shadow-lg"
      role="listbox"
      aria-label="Mention file"
      data-testid="mention-popup"
    >
      {filtered.map((path, i) => (
        <button
          key={path}
          type="button"
          role="option"
          aria-selected={i === active}
          className={cn(
            "flex w-full px-3 py-1.5 text-left font-mono text-[11px]",
            i === active
              ? "bg-[#2a2a2a] text-[#e8e8e8]"
              : "text-[#999] hover:bg-[#222] hover:text-[#ccc]",
          )}
          onMouseEnter={() => setActive(i)}
          onClick={() => onSelect(path)}
        >
          <span className="truncate">{path}</span>
        </button>
      ))}
    </div>
  );
}
