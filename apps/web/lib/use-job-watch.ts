"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, watchJob } from "./api";

/**
 * One job tracker for every surface that starts a serve-engine job (Serve,
 * Bench, Evals). Owns the SSE closer (closed on unmount / re-track / clear),
 * batches log appends per animation frame, and speaks one state vocabulary:
 * the backend emits `running` / `completed` / `failed` (`queued` / `cancelled`
 * once the cancel route lands); legacy `done` / `error` are normalised.
 */

export type JobState = "queued" | "running" | "completed" | "failed" | "cancelled";

export type JobWatch = {
  id: string | null;
  status: JobState | null;
  progress: number;
  message: string;
  logs: string;
  result: unknown;
};

const EMPTY: JobWatch = { id: null, status: null, progress: 0, message: "", logs: "", result: null };
const LOG_CAP = 80_000;

export function normalizeJobStatus(raw: string): JobState {
  switch (raw) {
    case "queued":
    case "running":
    case "completed":
    case "failed":
    case "cancelled":
      return raw;
    case "done":
      return "completed";
    case "error":
      return "failed";
    default:
      return "running";
  }
}

export function isTerminalJobState(s: JobState | null): boolean {
  return s === "completed" || s === "failed" || s === "cancelled";
}

export function useJobWatch(opts?: { onSettled?: () => void }) {
  const [job, setJob] = useState<JobWatch>(EMPTY);
  const closer = useRef<null | (() => void)>(null);
  const onSettled = useRef(opts?.onSettled);
  onSettled.current = opts?.onSettled;

  // Log chunks arrive faster than React should re-render; coalesce per frame (on a
  // timer while the tab is hidden — rAF does not run there).
  const pendingLog = useRef("");
  const logRaf = useRef(0);
  const logTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const flushLogs = useCallback(() => {
    if (logRaf.current) cancelAnimationFrame(logRaf.current);
    if (logTimer.current) clearTimeout(logTimer.current);
    logRaf.current = 0;
    logTimer.current = null;
    const chunk = pendingLog.current;
    pendingLog.current = "";
    if (!chunk) return;
    setJob((j) => ({ ...j, logs: (j.logs + chunk).slice(-LOG_CAP) }));
  }, []);

  const close = useCallback(() => {
    closer.current?.();
    closer.current = null;
    if (logRaf.current) cancelAnimationFrame(logRaf.current);
    if (logTimer.current) clearTimeout(logTimer.current);
    logRaf.current = 0;
    logTimer.current = null;
    pendingLog.current = "";
  }, []);

  const track = useCallback(
    (jobId: string) => {
      close();
      setJob({ ...EMPTY, id: jobId, status: "running", message: "starting…" });
      let settled = false;
      const settle = (patch: Partial<JobWatch>) => {
        if (settled) return;
        settled = true;
        flushLogs();
        setJob((j) => ({ ...j, ...patch }));
        onSettled.current?.();
      };
      closer.current = watchJob(
        jobId,
        (chunk) => {
          pendingLog.current += chunk;
          if (document.hidden) {
            if (!logTimer.current) logTimer.current = setTimeout(flushLogs, 500);
          } else if (!logRaf.current) logRaf.current = requestAnimationFrame(flushLogs);
        },
        (s) => {
          const status = normalizeJobStatus(s.status);
          setJob((j) => ({ ...j, status, progress: s.progress ?? 0, message: s.message ?? "" }));
          if (isTerminalJobState(status)) settle({ status });
        },
        (result) => settle({ result }),
      );
    },
    [close, flushLogs],
  );

  // The runner flips the row to `cancelled` at its next check; the stream reports it.
  // Only a job that was already terminal (or orphaned) settles here.
  const cancel = useCallback(async () => {
    if (!job.id) return;
    const r = await api.cancelJob(job.id);
    const status = normalizeJobStatus(r.status ?? "running");
    setJob((j) => (isTerminalJobState(j.status) ? j : { ...j, status, message: isTerminalJobState(status) ? status : "cancelling…" }));
  }, [job.id]);

  const clear = useCallback(() => {
    close();
    setJob(EMPTY);
  }, [close]);

  useEffect(() => close, [close]);

  const running = job.status === "running" || job.status === "queued";
  return {
    job,
    track,
    cancel,
    clear,
    running,
    done: job.status === "completed",
    failed: job.status === "failed" || job.status === "cancelled",
  };
}
