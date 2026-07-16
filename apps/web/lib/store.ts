"use client";

import { create } from "zustand";
import type { AgentMode, Patch, Session, Workspace } from "./api";

export type TimelineItem = {
  id: string;
  kind: "thought" | "status" | "tool" | "file" | "assistant" | "error" | "ran";
  text: string;
  meta?: Record<string, unknown>;
  ts: number;
};

export type OpenFile = {
  path: string;
  content: string;
  dirty?: boolean;
};

type ShellApproval = {
  runId: string;
  approvalId: string;
  command: string;
};

type LabStore = {
  workspace: Workspace | null;
  session: Session | null;
  sessions: Session[];
  timeline: TimelineItem[];
  activeRunId: string | null;
  openFiles: OpenFile[];
  activeFile: string | null;
  editorOpen: boolean;
  statusPanelOpen: boolean;
  modelLabel: string | null;
  agentMode: AgentMode;
  pendingPatches: Patch[];
  shellApproval: ShellApproval | null;
  streamingText: string;
  setWorkspace: (w: Workspace | null) => void;
  setSession: (s: Session | null) => void;
  setSessions: (s: Session[]) => void;
  pushTimeline: (item: Omit<TimelineItem, "id" | "ts">) => void;
  clearTimeline: () => void;
  setActiveRunId: (id: string | null) => void;
  openFile: (f: OpenFile) => void;
  updateFileContent: (path: string, content: string, dirty?: boolean) => void;
  closeFile: (path: string) => void;
  setActiveFile: (path: string | null) => void;
  setEditorOpen: (v: boolean) => void;
  setStatusPanelOpen: (v: boolean) => void;
  setModelLabel: (m: string | null) => void;
  setAgentMode: (m: AgentMode) => void;
  setPendingPatches: (p: Patch[]) => void;
  upsertPatch: (p: Patch) => void;
  setShellApproval: (v: ShellApproval | null) => void;
  appendStreamingText: (t: string) => void;
  clearStreamingText: () => void;
};

export const useLabStore = create<LabStore>((set) => ({
  workspace: null,
  session: null,
  sessions: [],
  timeline: [],
  activeRunId: null,
  openFiles: [],
  activeFile: null,
  editorOpen: true,
  statusPanelOpen: true,
  modelLabel: null,
  agentMode: "agent",
  pendingPatches: [],
  shellApproval: null,
  streamingText: "",
  setWorkspace: (workspace) => set({ workspace }),
  setSession: (session) => set({ session }),
  setSessions: (sessions) => set({ sessions }),
  pushTimeline: (item) =>
    set((s) => ({
      timeline: [
        ...s.timeline,
        { ...item, id: crypto.randomUUID(), ts: Date.now() },
      ].slice(-300),
    })),
  clearTimeline: () => set({ timeline: [] }),
  setActiveRunId: (activeRunId) => set({ activeRunId }),
  openFile: (f) =>
    set((s) => {
      const exists = s.openFiles.find((x) => x.path === f.path);
      if (exists) {
        return {
          openFiles: s.openFiles.map((x) =>
            x.path === f.path ? { ...x, content: f.content, dirty: false } : x,
          ),
          activeFile: f.path,
          editorOpen: true,
        };
      }
      return {
        openFiles: [...s.openFiles, f],
        activeFile: f.path,
        editorOpen: true,
      };
    }),
  updateFileContent: (path, content, dirty = true) =>
    set((s) => ({
      openFiles: s.openFiles.map((f) =>
        f.path === path ? { ...f, content, dirty } : f,
      ),
    })),
  closeFile: (path) =>
    set((s) => {
      const openFiles = s.openFiles.filter((f) => f.path !== path);
      const activeFile =
        s.activeFile === path ? openFiles[openFiles.length - 1]?.path ?? null : s.activeFile;
      return { openFiles, activeFile };
    }),
  setActiveFile: (activeFile) => set({ activeFile }),
  setEditorOpen: (editorOpen) => set({ editorOpen }),
  setStatusPanelOpen: (statusPanelOpen) => set({ statusPanelOpen }),
  setModelLabel: (modelLabel) => set({ modelLabel }),
  setAgentMode: (agentMode) => set({ agentMode }),
  setPendingPatches: (pendingPatches) => set({ pendingPatches }),
  upsertPatch: (p) =>
    set((s) => {
      // Keep only pending patches in the list; remove when status changes away from pending
      if (p.status !== "pending") {
        return {
          pendingPatches: s.pendingPatches.filter((x) => x.id !== p.id),
        };
      }
      const idx = s.pendingPatches.findIndex((x) => x.id === p.id);
      if (idx >= 0) {
        const pendingPatches = s.pendingPatches.slice();
        pendingPatches[idx] = p;
        return { pendingPatches };
      }
      return { pendingPatches: [...s.pendingPatches, p] };
    }),
  setShellApproval: (shellApproval) => set({ shellApproval }),
  appendStreamingText: (t) =>
    set((s) => ({ streamingText: s.streamingText + t })),
  clearStreamingText: () => set({ streamingText: "" }),
}));
