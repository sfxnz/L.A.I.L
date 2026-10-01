"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { useJobWatch } from "@/lib/use-job-watch";
import { Badge, Btn, Callout, LogView, Panel, ProgressBar, SegmentedControl, btnClass } from "@/components/ui";

type Preset = "short" | "full" | "hardmode" | "coding";
type ToolEval = { available: boolean; path?: string | null; version?: string | null; install?: string; repo?: string };

/**
 * The tool-calling suites, launched where their results are read: golden tools (a
 * 12-case selection smoke) and tool-eval-bench (the full quality suite). Their job
 * log streams here; a suite still running when the page opens is re-attached.
 */
export function AgenticPanel({ healthy, onFinished }: { healthy: boolean; onFinished?: () => void }) {
  const [err, setErr] = useState<string | null>(null);
  const [preset, setPreset] = useState<Preset>("short");
  const [teb, setTeb] = useState<ToolEval | null>(null);
  const finished = useRef(onFinished);
  finished.current = onFinished;
  const watch = useJobWatch({ onSettled: () => finished.current?.() });
  const { job, running } = watch;

  useEffect(() => {
    api.toolEvalStatus().then(setTeb).catch(() => {});
  }, []);

  // Re-attach to an agentic suite that is still running (started here or by Hermes).
  useEffect(() => {
    let cancelled = false;
    api
      .jobs()
      .then((list) => {
        const live = list.find((j) => (j.status === "running" || j.status === "queued") && j.kind.startsWith("agentic_"));
        if (live && !cancelled) watch.track(live.job_id);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mount-only attach
  }, []);

  const start = async (body: Parameters<typeof api.benchAgentic>[0]) => {
    setErr(null);
    try {
      const { job_id } = await api.benchAgentic(body);
      watch.track(job_id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  const available = !!teb?.available;
  const blocked = !healthy ? "Start a model on Serve first" : running ? "A suite is already running" : null;

  return (
    <Panel>
      <div className="space-y-4 p-4">
        {err && (
          <Callout tone="danger" title="Could not start" onDismiss={() => setErr(null)}>
            {err}
          </Callout>
        )}
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="space-y-2.5">
            <div className="flex flex-wrap items-center gap-2">
              <span className="animus-eyebrow text-lab-text-dim">Golden tools</span>
              <Badge tone="muted">12 cases</Badge>
            </div>
            <p className="text-[12px] leading-relaxed text-lab-muted">
              Does the model pick the right tool (or none)? Needs{" "}
              <code className="font-mono text-[11px] text-lab-text-dim">--enable-auto-tool-choice</code> and a tool-call
              parser on the serve.
            </p>
            <Btn onClick={() => void start({ suite: "golden" })} disabled={!!blocked} title={blocked ?? undefined}>
              Run golden tools
            </Btn>
          </div>

          <div className="space-y-2.5">
            <div className="flex flex-wrap items-center gap-2">
              <span className="animus-eyebrow text-lab-text-dim">tool-eval-bench</span>
              <Badge tone={available ? "ok" : "warn"} dot>
                {teb == null ? "checking…" : available ? `installed${teb.version ? ` · ${teb.version}` : ""}` : "not installed"}
              </Badge>
            </div>
            <p className="text-[12px] leading-relaxed text-lab-muted">
              Selection, parameters, multi-step, restraint, safety, structured output — scored 0–100 with safety gating
              (
              <a
                href="https://github.com/SeraphimSerapis/tool-eval-bench"
                target="_blank"
                rel="noreferrer"
                className="text-lab-accent-bright underline-offset-2 hover:underline"
              >
                SeraphimSerapis/tool-eval-bench
              </a>
              ). Runs with <code className="font-mono text-[11px]">--no-think</code>.
            </p>
            {teb && !available ? (
              <Callout tone="warn" title="Not installed">
                <span>Install on this host, then restart <code className="font-mono text-[11px]">bun run dev</code>:</span>
                <pre className="mt-2 overflow-x-auto rounded-[2px] border border-l-2 border-lab-border border-l-lab-warn bg-lab-editor p-2.5 font-mono text-[11px] text-lab-text-dim">
                  {teb.install || "uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git"}
                </pre>
              </Callout>
            ) : (
              <div className="flex flex-wrap items-center gap-2">
                <SegmentedControl
                  ariaLabel="tool-eval-bench preset"
                  value={preset}
                  onChange={setPreset}
                  options={[
                    { id: "short", label: "Short (15)" },
                    { id: "full", label: "Full (69)" },
                    { id: "hardmode", label: "Hard mode" },
                    { id: "coding", label: "Coding" },
                  ]}
                />
                <Btn
                  onClick={() => void start({ suite: "tool_eval", preset })}
                  disabled={!!blocked || !available}
                  title={blocked ?? (!available ? "Install tool-eval-bench first" : undefined)}
                >
                  Run {preset}
                </Btn>
                <Link href="/evals/tool" className={btnClass("secondary", "md")}>
                  Results board →
                </Link>
              </div>
            )}
          </div>
        </div>

        {job.id && (
          <div className="space-y-2.5 border-t border-[color:var(--animus-hairline)] pt-3">
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone={watch.done ? "ok" : watch.failed ? "danger" : "accent"} dot={running}>
                {job.status ?? "running"}
              </Badge>
              {running && (
                <Btn variant="danger" size="sm" onClick={() => void watch.cancel().catch((e) => setErr(String(e)))}>
                  Cancel
                </Btn>
              )}
              {!running && (
                <Btn variant="ghost" size="sm" onClick={watch.clear}>
                  Dismiss
                </Btn>
              )}
            </div>
            <ProgressBar
              value={Math.round((job.progress || 0) * 100)}
              indeterminate={running && !(job.progress > 0)}
              label={job.message || (running ? "Working…" : "Last suite")}
            />
            <LogView text={job.logs} live={running} empty="The suite's output streams here." />
          </div>
        )}
      </div>
    </Panel>
  );
}
