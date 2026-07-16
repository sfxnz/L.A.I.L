"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowUp,
  Brain,
  ChevronRight,
  Columns2,
  FileDiff,
  FilePenLine,
  Loader2,
  PanelRight,
  Plus,
  Save,
  Square,
  Terminal,
  X,
} from "lucide-react";
import { ModeToggle } from "@/components/workbench/ModeToggle";
import { PatchReviewPanel } from "@/components/workbench/PatchReviewPanel";
import { ShellApprovalBanner } from "@/components/workbench/ShellApprovalBanner";
import { api, type Patch } from "@/lib/api";
import { useLabStore } from "@/lib/store";
import {
  COMPOSER_PLACEHOLDER,
  STREAM_MARKERS,
  fileWriteLabel,
  groupTimeline,
  ranLabel,
  type StreamBlock,
} from "@/lib/ide-chrome";
import { onWsEvent, wsSubscribe } from "@/lib/ws";
import { cn } from "@/lib/utils";

export default function WorkbenchPage() {
  const {
    workspace,
    session,
    setSession,
    setSessions,
    timeline,
    pushTimeline,
    clearTimeline,
    activeRunId,
    setActiveRunId,
    openFiles,
    activeFile,
    setActiveFile,
    openFile,
    updateFileContent,
    closeFile,
    editorOpen,
    setEditorOpen,
    statusPanelOpen,
    setStatusPanelOpen,
    modelLabel,
    agentMode,
    setAgentMode,
    pendingPatches,
    upsertPatch,
    shellApproval,
    setShellApproval,
    streamingText,
    appendStreamingText,
    clearStreamingText,
  } = useLabStore();

  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [cmdCount, setCmdCount] = useState(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  const active = openFiles.find((f) => f.path === activeFile) || null;

  useEffect(() => {
    (async () => {
      let s = session;
      if (!s) {
        const list = await api.sessions.list();
        if (list[0]) s = list[0];
        else s = await api.sessions.create("Composer", workspace?.id);
        setSession(s);
        setSessions(await api.sessions.list());
      }
      if (s) {
        const full = await api.sessions.get(s.id);
        setMessages(full.messages.map((m) => ({ role: m.role, content: m.content })));
      }
    })().catch(console.error);
  }, [session?.id, workspace?.id]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [timeline, messages, busy, streamingText]);

  useEffect(() => {
    return onWsEvent((event, channel) => {
      if (!channel.startsWith("agent")) return;
      const type = String(event.type || "");
      const runId = String(event.runId || "");
      if (activeRunId && runId && runId !== activeRunId) return;

      if (type === "token") {
        appendStreamingText(String(event.text || ""));
      } else if (type === "thought") {
        pushTimeline({ kind: "thought", text: String(event.text || "") });
      } else if (type === "status") {
        pushTimeline({ kind: "status", text: String(event.text || "") });
      } else if (type === "tool_start") {
        // Live indicator only — do not count as a finished command (tool_end does).
        pushTimeline({
          kind: "status",
          text: `Running ${String(event.tool || "tool")}…`,
          meta: { phase: "start", args: event.args, tool: event.tool },
        });
      } else if (type === "tool_end") {
        setCmdCount((c) => c + 1);
        pushTimeline({
          kind: "ran",
          text: String(event.summary || event.tool || "command"),
          meta: { phase: "end", output: event.output, tool: event.tool },
        });
      } else if (type === "file_write") {
        const path = String(event.path || "");
        pushTimeline({
          kind: "file",
          text: path,
          meta: { bytes: event.bytes, creating: true },
        });
        if (workspace && path) {
          api.workspaces
            .readFile(workspace.id, path)
            .then((f) => openFile({ path: f.path, content: f.content }))
            .catch(() => {});
        }
      } else if (type === "patch_proposed" || type === "patch_updated") {
        const patch = event.patch as Patch | undefined;
        if (patch && typeof patch === "object" && patch.id) {
          upsertPatch(patch);
          if (type === "patch_proposed") {
            pushTimeline({
              kind: "patch",
              text: String(patch.path || ""),
              meta: { patchId: patch.id, op: patch.op },
            });
          }
        }
      } else if (type === "shell_approval_required") {
        setShellApproval({
          runId: runId || String(event.runId || ""),
          approvalId: String(event.approvalId || ""),
          command: String(event.command || ""),
        });
      } else if (type === "assistant") {
        // Messages pane only — do not also push timeline (avoids duplicate bubbles).
        const text = String(event.text || "");
        setMessages((m) => [...m, { role: "assistant", content: text }]);
        clearStreamingText();
      } else if (type === "error") {
        pushTimeline({ kind: "error", text: String(event.message || "error") });
        setBusy(false);
        clearStreamingText();
      } else if (type === "cancelled") {
        setBusy(false);
        setActiveRunId(null);
        clearStreamingText();
        pushTimeline({ kind: "status", text: "Run cancelled" });
      } else if (type === "done") {
        setBusy(false);
        setActiveRunId(null);
        clearStreamingText();
        if (session) {
          api.sessions.get(session.id).then((full) => {
            setMessages(full.messages.map((m) => ({ role: m.role, content: m.content })));
          });
        }
      }
    });
  }, [
    activeRunId,
    pushTimeline,
    session,
    setActiveRunId,
    workspace,
    openFile,
    appendStreamingText,
    clearStreamingText,
    upsertPatch,
    setShellApproval,
  ]);

  async function send(text?: string) {
    const message = (text ?? input).trim();
    if (!message || !session) return;
    setInput("");
    if (taRef.current) taRef.current.style.height = "auto";
    setBusy(true);
    setCmdCount(0);
    clearTimeline();
    clearStreamingText();
    setMessages((m) => [...m, { role: "user", content: message }]);
    try {
      const { runId } = await api.agentRun(
        session.id,
        message,
        workspace?.id,
        agentMode,
      );
      setActiveRunId(runId);
      wsSubscribe(`agent:${runId}`);
      wsSubscribe("agent");
    } catch (e) {
      pushTimeline({
        kind: "error",
        text: e instanceof Error ? e.message : String(e),
      });
      setBusy(false);
    }
  }

  async function cancelRun() {
    if (!activeRunId) return;
    try {
      await api.cancelAgentRun(activeRunId);
    } catch (e) {
      console.error(e);
    }
  }

  async function newChat() {
    const s = await api.sessions.create("Composer", workspace?.id);
    setSession(s);
    setSessions(await api.sessions.list());
    setMessages([]);
    clearTimeline();
    clearStreamingText();
  }

  async function saveActive() {
    if (!workspace || !active) return;
    await api.workspaces.writeFile(workspace.id, active.path, active.content);
    updateFileContent(active.path, active.content, false);
  }

  const streamBlocks = useMemo(() => groupTimeline(timeline), [timeline]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-9 shrink-0 items-center gap-1 border-b border-[#1f1f1f] bg-[#141414] px-2">
        <div className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto">
          <TabChip active label="Composer" />
          {openFiles.map((f) => (
            <TabChip
              key={f.path}
              label={f.path.split("/").pop() || f.path}
              active={activeFile === f.path && editorOpen}
              dirty={f.dirty}
              onClick={() => {
                setActiveFile(f.path);
                setEditorOpen(true);
              }}
              onClose={() => closeFile(f.path)}
            />
          ))}
        </div>
        <button
          type="button"
          title="Toggle editor"
          aria-label="Toggle editor"
          onClick={() => setEditorOpen(!editorOpen)}
          className="rounded p-1.5 text-[#777] hover:bg-[#222] hover:text-[#ccc]"
        >
          <Columns2 className="h-3.5 w-3.5" strokeWidth={1.5} />
        </button>
        <button
          type="button"
          title="Toggle Status"
          aria-label="Toggle Status"
          onClick={() => setStatusPanelOpen(!statusPanelOpen)}
          className="rounded p-1.5 text-[#777] hover:bg-[#222] hover:text-[#ccc]"
        >
          <PanelRight className="h-3.5 w-3.5" strokeWidth={1.5} />
        </button>
      </div>

      <div className="flex min-h-0 flex-1">
        <div
          className={cn(
            "flex min-w-0 flex-col border-r border-[#1f1f1f]",
            editorOpen && openFiles.length ? "w-[46%]" : "flex-1",
          )}
        >
          <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
            {messages.length === 0 && timeline.length === 0 && !streamingText && (
              <div className="mx-auto max-w-lg pt-20 text-center">
                <p className="text-[13px] font-medium text-[#ddd]">Workbench</p>
                <p className="mt-2 text-[12px] leading-relaxed text-[#666]">
                  Local agentic IDE. Thought → Ran commands → Creating files — backed by vLLM /
                  llama.cpp. Models and Server remain available from the sidebar.
                </p>
              </div>
            )}

            <div className="mx-auto max-w-2xl space-y-4">
              {messages.map((m, i) =>
                m.role === "user" ? (
                  <div key={i} className="flex justify-end">
                    <div className="max-w-[90%] rounded-2xl bg-[#2a2a2a] px-3.5 py-2 text-[13px] leading-relaxed text-[#e8e8e8]">
                      {m.content}
                    </div>
                  </div>
                ) : (
                  <div
                    key={i}
                    className="text-[13px] leading-relaxed text-[#cfcfcf] whitespace-pre-wrap"
                  >
                    {m.content}
                  </div>
                ),
              )}

              {(busy || streamBlocks.length > 0 || streamingText) && (
                <div className="space-y-2">
                  {streamBlocks.map((b, i) => (
                    <StreamBlockView key={i} block={b} />
                  ))}
                  {streamingText ? (
                    <div
                      className="text-[13px] leading-relaxed text-[#cfcfcf] whitespace-pre-wrap"
                      data-testid="streaming-draft"
                    >
                      {streamingText}
                      {busy && (
                        <span className="ml-0.5 inline-block h-3 w-1 animate-pulse bg-[#888] align-middle" />
                      )}
                    </div>
                  ) : null}
                  {busy && (
                    <div className="flex items-center gap-2 text-[12px] text-[#777]">
                      <Loader2 className="h-3 w-3 animate-spin" />
                      {STREAM_MARKERS.working}
                      {cmdCount > 0 ? ` · ${cmdCount} tools` : ""}
                      {shellApproval ? " · waiting for shell approval" : ""}
                    </div>
                  )}
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          </div>

          <div className="shrink-0 border-t border-[#1f1f1f] bg-[#0e0e0e] px-3 py-3">
            <div className="mx-auto max-w-2xl">
              <ShellApprovalBanner />
              <div className="rounded-2xl border border-[#2a2a2a] bg-[#181818] focus-within:border-[#3d3d3d]">
                <textarea
                  ref={taRef}
                  rows={2}
                  disabled={busy}
                  value={input}
                  placeholder={COMPOSER_PLACEHOLDER}
                  aria-label={COMPOSER_PLACEHOLDER}
                  className="w-full resize-none bg-transparent px-3.5 py-2.5 text-[13px] text-[#e4e4e4] outline-none placeholder:text-[#555]"
                  onChange={(e) => {
                    setInput(e.target.value);
                    e.target.style.height = "auto";
                    e.target.style.height = `${Math.min(e.target.scrollHeight, 140)}px`;
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      void send();
                    }
                  }}
                />
                <div className="flex items-center justify-between px-2.5 pb-2">
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => void newChat()}
                      className="rounded-md p-1.5 text-[#666] hover:bg-[#222] hover:text-[#ccc]"
                      title="New chat"
                    >
                      <Plus className="h-3.5 w-3.5" />
                    </button>
                    <ModeToggle value={agentMode} onChange={setAgentMode} />
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className="max-w-[160px] truncate font-mono text-[11px] text-[#666]"
                      data-testid="model-label"
                    >
                      {modelLabel?.split("/").pop() || "model"}
                    </span>
                    {busy ? (
                      <button
                        type="button"
                        onClick={() => void cancelRun()}
                        aria-label="Cancel"
                        title="Cancel run"
                        className="flex h-7 items-center gap-1 rounded-full border border-[#3a3a3a] bg-[#2a2a2a] px-2.5 text-[11px] text-[#ccc] hover:bg-[#333]"
                      >
                        <Square className="h-2.5 w-2.5 fill-current" />
                        Cancel
                      </button>
                    ) : (
                      <button
                        type="button"
                        disabled={!input.trim()}
                        onClick={() => void send()}
                        aria-label="Send"
                        className={cn(
                          "flex h-7 w-7 items-center justify-center rounded-full transition",
                          !input.trim()
                            ? "bg-[#2a2a2a] text-[#555]"
                            : "bg-white text-black hover:opacity-90",
                        )}
                      >
                        <ArrowUp className="h-3.5 w-3.5" strokeWidth={2.2} />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {editorOpen && openFiles.length > 0 && (
          <div className="flex min-w-0 flex-1 flex-col bg-[#0c0c0c]" data-testid="file-editor">
            <div className="flex h-8 items-center justify-between border-b border-[#1f1f1f] px-2">
              <span className="truncate font-mono text-[11px] text-[#888]">
                {active?.path || "—"}
                {active?.dirty ? " · modified" : ""}
              </span>
              <button
                type="button"
                onClick={() => void saveActive()}
                disabled={!active?.dirty}
                className="flex items-center gap-1 rounded px-2 py-0.5 text-[11px] text-[#888] hover:bg-[#1a1a1a] hover:text-white disabled:opacity-30"
              >
                <Save className="h-3 w-3" />
                Save
              </button>
            </div>
            <textarea
              className="min-h-0 flex-1 resize-none bg-transparent p-3 font-mono text-[12px] leading-relaxed text-[#d4d4d4] outline-none"
              spellCheck={false}
              value={active?.content ?? ""}
              onChange={(e) => {
                if (active) updateFileContent(active.path, e.target.value, true);
              }}
            />
          </div>
        )}

        {pendingPatches.length > 0 && <PatchReviewPanel />}

        {statusPanelOpen && (
          <aside
            className="flex w-[240px] shrink-0 flex-col border-l border-[#1f1f1f] bg-[#111111]"
            data-testid="status-rail"
            aria-label="Status"
          >
            <div className="flex h-8 items-center gap-2 border-b border-[#1f1f1f] px-2 text-[11px]">
              <span className="rounded bg-[#222] px-1.5 py-0.5 text-white">Status</span>
              <span className="text-[#555]">Plan</span>
              <span className="text-[#555]">Filesystem</span>
            </div>
            <div className="flex-1 space-y-4 overflow-y-auto p-3 text-[11px]">
              <div>
                <div className="mb-1.5 text-[10px] font-semibold tracking-wider text-[#555]">
                  SESSION
                </div>
                <div className="space-y-1">
                  <KV k="State" v={busy ? "running" : "idle"} />
                  <KV k="Mode" v={agentMode} />
                  <KV k="Model" v={modelLabel?.split("/").pop() || "—"} />
                  <KV k="Tools this run" v={String(cmdCount)} />
                  <KV k="Pending patches" v={String(pendingPatches.length)} />
                  <KV k="Open files" v={String(openFiles.length)} />
                </div>
              </div>
              <div>
                <div className="mb-1.5 text-[10px] font-semibold tracking-wider text-[#555]">
                  WORKSPACE
                </div>
                <div className="space-y-1">
                  <KV k="Project" v={workspace?.name || "—"} />
                  <KV
                    k="Directory"
                    v={workspace?.rootPath?.split("/").slice(-2).join("/") || "—"}
                  />
                </div>
              </div>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-[#666]">{k}</span>
      <span className="truncate text-right text-[#bbb]">{v}</span>
    </div>
  );
}

function TabChip({
  label,
  active,
  dirty,
  onClick,
  onClose,
}: {
  label: string;
  active?: boolean;
  dirty?: boolean;
  onClick?: () => void;
  onClose?: () => void;
}) {
  return (
    <div
      className={cn(
        "group flex max-w-[160px] items-center gap-1 rounded-t-md border border-b-0 px-2 py-1 text-[11px]",
        active
          ? "border-[#2a2a2a] bg-[#0e0e0e] text-white"
          : "border-transparent text-[#777] hover:bg-[#1a1a1a] hover:text-[#ccc]",
      )}
    >
      <button type="button" onClick={onClick} className="truncate">
        {dirty ? "● " : ""}
        {label}
      </button>
      {onClose && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onClose();
          }}
          className="rounded p-0.5 opacity-0 hover:bg-[#333] group-hover:opacity-100"
          aria-label="Close tab"
        >
          <X className="h-3 w-3" />
        </button>
      )}
    </div>
  );
}

function StreamBlockView({ block }: { block: StreamBlock }) {
  if (block.type === "ran") {
    return (
      <details className="group rounded-md border border-[#222] bg-[#141414]">
        <summary className="flex cursor-pointer list-none items-center gap-2 px-2.5 py-1.5 text-[12px] text-[#999] hover:text-[#ccc]">
          <Terminal className="h-3 w-3 shrink-0" strokeWidth={1.5} />
          <span>{ranLabel(block.count, block.detail)}</span>
          <ChevronRight className="ml-auto h-3 w-3 opacity-50 transition group-open:rotate-90" />
        </summary>
        {block.output && (
          <pre className="max-h-40 overflow-auto border-t border-[#1f1f1f] px-2.5 py-2 font-mono text-[10px] text-[#777]">
            {block.output.slice(0, 2000)}
          </pre>
        )}
      </details>
    );
  }
  if (block.type === "thought") {
    return (
      <div className="space-y-1">
        <div className="flex items-center gap-1.5 text-[11px] font-medium text-[#888]">
          <Brain className="h-3 w-3" strokeWidth={1.5} />
          {STREAM_MARKERS.thought}
        </div>
        <p className="pl-4 text-[12.5px] leading-relaxed text-[#aaa]">{block.text}</p>
      </div>
    );
  }
  if (block.type === "status") {
    return (
      <p className="text-[12px] text-[#777]">
        <span className="font-medium text-[#888]">{STREAM_MARKERS.working}</span> {block.text}
      </p>
    );
  }
  if (block.type === "file") {
    return (
      <div className="rounded-md border border-[#222] bg-[#141414] px-2.5 py-2">
        <div className="flex items-center gap-1.5 text-[11px] text-[#8fbcbb]">
          <FilePenLine className="h-3 w-3" strokeWidth={1.5} />
          {fileWriteLabel(block.path, block.creating)}
        </div>
      </div>
    );
  }
  if (block.type === "patch") {
    return (
      <div className="rounded-md border border-[#222] bg-[#141414] px-2.5 py-2">
        <div className="flex items-center gap-1.5 text-[11px] text-[#c9a86c]">
          <FileDiff className="h-3 w-3" strokeWidth={1.5} />
          {STREAM_MARKERS.proposed} {block.path}
        </div>
      </div>
    );
  }
  if (block.type === "error") {
    return (
      <div className="rounded-md border border-[#5a2a2a] bg-[#1a1010] px-2.5 py-2 text-[12px] text-[#e07070]">
        {block.text}
      </div>
    );
  }
  return (
    <div className="text-[13px] leading-relaxed text-[#cfcfcf] whitespace-pre-wrap">{block.text}</div>
  );
}
