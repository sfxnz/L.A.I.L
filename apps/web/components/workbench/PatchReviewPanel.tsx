"use client";

import { useState } from "react";
import { Check, CheckCheck, FileDiff, X } from "lucide-react";
import { api } from "@/lib/api";
import type { Patch } from "@/lib/api";
import { useLabStore } from "@/lib/store";
import { cn } from "@/lib/utils";

function truncate(s: string, n = 80): string {
  const t = s.replace(/\s+/g, " ").trim();
  if (t.length <= n) return t;
  return `${t.slice(0, n)}…`;
}

function PatchCard({
  patch,
  onAccept,
  onReject,
  busy,
}: {
  patch: Patch;
  onAccept: () => void;
  onReject: () => void;
  busy: boolean;
}) {
  return (
    <div className="rounded-md border border-[#2a2a2a] bg-[#181818] p-2.5">
      <div className="flex items-start gap-1.5">
        <FileDiff className="mt-0.5 h-3 w-3 shrink-0 text-[#8fbcbb]" strokeWidth={1.5} />
        <div className="min-w-0 flex-1">
          <div className="truncate font-mono text-[11px] text-[#ddd]" title={patch.path}>
            {patch.path}
          </div>
          <div className="mt-0.5 text-[10px] uppercase tracking-wide text-[#666]">
            {patch.op}
          </div>
        </div>
      </div>
      {(patch.oldString || patch.newString) && (
        <div className="mt-2 space-y-1 font-mono text-[10px] leading-snug">
          {patch.oldString ? (
            <div className="truncate text-[#c07070]" title={patch.oldString}>
              − {truncate(patch.oldString)}
            </div>
          ) : null}
          {patch.newString ? (
            <div className="truncate text-[#7cb87c]" title={patch.newString}>
              + {truncate(patch.newString)}
            </div>
          ) : null}
        </div>
      )}
      <div className="mt-2 flex items-center gap-1.5">
        <button
          type="button"
          disabled={busy}
          onClick={onAccept}
          className={cn(
            "inline-flex flex-1 items-center justify-center gap-1 rounded border border-[#2a4a2a] bg-[#1a2a1a] px-2 py-1 text-[11px] text-[#8fbc8f] hover:bg-[#223322]",
            busy && "opacity-50",
          )}
        >
          <Check className="h-3 w-3" strokeWidth={1.5} />
          Accept
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={onReject}
          className={cn(
            "inline-flex flex-1 items-center justify-center gap-1 rounded border border-[#4a2a2a] bg-[#1a1010] px-2 py-1 text-[11px] text-[#e07070] hover:bg-[#2a1515]",
            busy && "opacity-50",
          )}
        >
          <X className="h-3 w-3" strokeWidth={1.5} />
          Reject
        </button>
      </div>
    </div>
  );
}

export function PatchReviewPanel() {
  const { pendingPatches, upsertPatch, openFile, workspace, session } = useLabStore();
  const [busyId, setBusyId] = useState<string | null>(null);

  if (pendingPatches.length === 0) return null;

  async function acceptOne(id: string) {
    setBusyId(id);
    try {
      const updated = await api.patches.accept(id);
      upsertPatch(updated);
      if (workspace && updated.path) {
        try {
          const f = await api.workspaces.readFile(workspace.id, updated.path);
          openFile({ path: f.path, content: f.content });
        } catch {
          /* file may not exist for delete */
        }
      }
    } catch (e) {
      console.error(e);
    } finally {
      setBusyId(null);
    }
  }

  async function rejectOne(id: string) {
    setBusyId(id);
    try {
      const updated = await api.patches.reject(id);
      upsertPatch(updated);
    } catch (e) {
      console.error(e);
    } finally {
      setBusyId(null);
    }
  }

  async function acceptAll() {
    setBusyId("__all__");
    try {
      const body: { sessionId?: string; runId?: string } = {};
      if (session?.id) body.sessionId = session.id;
      const updated = await api.patches.acceptAll(body);
      for (const p of updated) {
        upsertPatch(p);
        if (workspace && p.path && p.status === "accepted") {
          try {
            const f = await api.workspaces.readFile(workspace.id, p.path);
            openFile({ path: f.path, content: f.content });
          } catch {
            /* */
          }
        }
      }
    } catch (e) {
      console.error(e);
    } finally {
      setBusyId(null);
    }
  }

  return (
    <aside
      className="flex w-[280px] shrink-0 flex-col border-l border-[#1f1f1f] bg-[#111111]"
      data-testid="patch-review"
      aria-label="Patch review"
    >
      <div className="flex h-8 items-center justify-between border-b border-[#1f1f1f] px-2.5">
        <span className="text-[11px] font-medium text-[#ccc]">
          Patches
          <span className="ml-1.5 text-[#666]">{pendingPatches.length}</span>
        </span>
        <button
          type="button"
          disabled={busyId !== null || pendingPatches.length === 0}
          onClick={() => void acceptAll()}
          className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-[#8fbc8f] hover:bg-[#1a2a1a] disabled:opacity-40"
        >
          <CheckCheck className="h-3 w-3" strokeWidth={1.5} />
          Accept all
        </button>
      </div>
      <div className="flex-1 space-y-2 overflow-y-auto p-2">
        {pendingPatches.map((p) => (
          <PatchCard
            key={p.id}
            patch={p}
            busy={busyId !== null}
            onAccept={() => void acceptOne(p.id)}
            onReject={() => void rejectOne(p.id)}
          />
        ))}
      </div>
    </aside>
  );
}
