"use client";

import { create } from "zustand";
import { api, type LabStatus } from "./api";
import { isUnauthorizedError } from "./auth-token";

/**
 * The ONE lab-status poll. AppShell owns the cadence (2s while the tab is
 * visible, paused while hidden); every page reads from here instead of
 * running its own interval. 401 (`needToken`) and unreachable are kept apart —
 * a fresh browser with no token is not an outage.
 */

export const LAB_STATUS_POLL_MS = 2000;

export type StrandCounts = { running: number; waiting: number };

type LabStatusStore = {
  status: LabStatus | null;
  /** true until the first poll settles (success or failure) */
  loading: boolean;
  /** controller answered 401 — paste LAIL_TOKEN */
  needToken: boolean;
  /** controller did not answer at all */
  unreachable: boolean;
  error: string | null;
  /** epoch ms of the last successful poll */
  lastGoodAt: number | null;
  /**
   * Hook point for the streams engine (Phase B): live strand counts shown in
   * the header instrument strip. Zero until a run is feeding them.
   */
  strands: StrandCounts;
  refresh: () => Promise<void>;
  setStrands: (s: StrandCounts) => void;
};

export const useLabStatusStore = create<LabStatusStore>((set) => ({
  status: null,
  loading: true,
  needToken: false,
  unreachable: false,
  error: null,
  lastGoodAt: null,
  strands: { running: 0, waiting: 0 },
  refresh: async () => {
    try {
      const status = await api.labStatus();
      set({
        status,
        loading: false,
        needToken: false,
        unreachable: false,
        error: null,
        lastGoodAt: Date.now(),
      });
    } catch (e) {
      const unauthorized = isUnauthorizedError(e);
      set({
        loading: false,
        needToken: unauthorized,
        unreachable: !unauthorized,
        error: unauthorized ? null : String((e as Error).message || e),
      });
    }
  },
  setStrands: (strands) => set({ strands }),
}));

/** Read-only view for pages. Re-renders on every poll, like a page-local poll did. */
export function useLabStatus() {
  return useLabStatusStore();
}

/**
 * Start the poll. Returns a stop function. Only AppShell calls this; ticks
 * pause while `document.hidden` and resume (with an immediate tick) on return.
 */
export function startLabStatusPolling(): () => void {
  const { refresh } = useLabStatusStore.getState();
  let timer: ReturnType<typeof setInterval> | null = null;

  const stop = () => {
    if (timer) clearInterval(timer);
    timer = null;
  };
  const start = () => {
    if (timer || document.hidden) return;
    void refresh();
    timer = setInterval(() => void refresh(), LAB_STATUS_POLL_MS);
  };
  const onVisibility = () => (document.hidden ? stop() : start());

  start();
  document.addEventListener("visibilitychange", onVisibility);
  return () => {
    stop();
    document.removeEventListener("visibilitychange", onVisibility);
  };
}

/** Serve is healthy when the engine answered, was reachable, and reports a live endpoint. */
export function serveHealthy(status: LabStatus | null): boolean {
  const s = status?.serve;
  return !!(s && !s.unreachable && s.healthy);
}
