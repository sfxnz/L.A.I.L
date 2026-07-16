"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Boxes,
  Cable,
  ChevronDown,
  ChevronRight,
  FileCode2,
  Folder,
  FolderOpen,
  Gauge,
  LayoutDashboard,
  Plus,
  Plug,
  Search,
  Server,
  Settings2,
  Sparkles,
  SquareTerminal,
} from "lucide-react";
import { useEffect, useState } from "react";
import { api, type Session, type TreeNode } from "@/lib/api";
import { useLabStore } from "@/lib/store";
import { WORKSPACE_NAV } from "@/lib/ide-chrome";
import { cn } from "@/lib/utils";

const NAV_ICONS: Record<string, React.ComponentType<{ className?: string; strokeWidth?: number }>> = {
  Status: LayoutDashboard,
  Workbench: SquareTerminal,
  Models: Boxes,
  Configure: Settings2,
  Usage: Gauge,
  Integrations: Cable,
  Server: Server,
};

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const {
    workspace,
    setWorkspace,
    sessions,
    setSessions,
    session,
    setSession,
    openFile,
    modelLabel,
    setModelLabel,
  } = useLabStore();
  const [tree, setTree] = useState<TreeNode[]>([]);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [search, setSearch] = useState("");
  const [projects, setProjects] = useState<{ id: string; name: string }[]>([]);

  useEffect(() => {
    api
      .bootstrap()
      .then(async (b) => {
        setWorkspace(b.workspace);
        const list = await api.workspaces.list();
        setProjects(list.map((w) => ({ id: w.id, name: w.name })));
        return api.workspaces.tree(b.workspace.id);
      })
      .then(setTree)
      .catch(() => {});
    api.sessions.list().then(setSessions).catch(() => {});
    const tick = () => {
      api.health().then(() => setHealthy(true)).catch(() => setHealthy(false));
      api
        .labStatus()
        .then((s) => {
          setHealthy(true);
          const id = s.serve?.model_id || s.defaultModel;
          if (id && id !== "auto" && id !== "default") setModelLabel(id);
        })
        .catch(() => {});
    };
    tick();
    const t = setInterval(tick, 8000);
    return () => clearInterval(t);
  }, [setWorkspace, setSessions, setModelLabel]);

  useEffect(() => {
    if (!workspace) return;
    api.workspaces.tree(workspace.id).then(setTree).catch(() => {});
  }, [workspace, pathname]);

  async function openPath(rel: string) {
    if (!workspace) return;
    try {
      const f = await api.workspaces.readFile(workspace.id, rel);
      openFile({ path: f.path, content: f.content });
      if (pathname !== "/workbench") router.push("/workbench");
    } catch {
      /* binary or missing */
    }
  }

  const filteredSessions = sessions.filter(
    (s) => !search.trim() || s.title.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <div className="flex h-full min-h-0 bg-[#0c0c0c] text-[#e4e4e4]">
      <aside className="flex w-[232px] shrink-0 flex-col border-r border-[#1f1f1f] bg-[#111111]">
        <div className="flex h-10 items-center gap-1 border-b border-[#1f1f1f] px-2">
          <div className="flex flex-1 items-center gap-1.5 rounded-md bg-[#1a1a1a] px-2 py-1.5 text-[12px] text-[#666]">
            <Search className="h-3.5 w-3.5 shrink-0" strokeWidth={1.5} />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search"
              aria-label="Search"
              className="w-full bg-transparent text-[12px] text-[#ccc] outline-none placeholder:text-[#555]"
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-1.5 py-2">
          <div className="mb-1 px-2 text-[10px] font-medium uppercase tracking-wider text-[#555]">
            Workspace
          </div>
          <nav className="mb-3 flex flex-col gap-0.5" aria-label="Workspace">
            {WORKSPACE_NAV.map(({ href, label }) => {
              const Icon = NAV_ICONS[label] || LayoutDashboard;
              const active = pathname === href || pathname.startsWith(href + "/");
              return (
                <Link
                  key={href}
                  href={href}
                  className={cn(
                    "flex items-center gap-2 rounded-md px-2 py-[6px] text-[12.5px] transition-colors",
                    active
                      ? "bg-[#2a2a2a] text-white"
                      : "text-[#9a9a9a] hover:bg-[#1c1c1c] hover:text-[#ddd]",
                  )}
                >
                  <Icon className="h-[15px] w-[15px] shrink-0 opacity-80" strokeWidth={1.5} />
                  {label}
                </Link>
              );
            })}
          </nav>

          <div className="mb-1 px-2 text-[10px] font-medium uppercase tracking-wider text-[#555]">
            Pinned
          </div>
          <div className="mb-3 space-y-0.5" aria-label="Pinned">
            {sessions
              .filter((s) => s.pinned)
              .slice(0, 4)
              .map((s) => (
                <SessionRow
                  key={s.id}
                  s={s}
                  active={session?.id === s.id}
                  onClick={() => {
                    setSession(s);
                    router.push("/workbench");
                  }}
                />
              ))}
            {!sessions.some((s) => s.pinned) && sessions[0] && (
              <SessionRow
                s={sessions[0]}
                active={session?.id === sessions[0].id}
                onClick={() => {
                  setSession(sessions[0]);
                  router.push("/workbench");
                }}
              />
            )}
          </div>

          <div className="mb-1 flex items-center justify-between px-2">
            <span className="text-[10px] font-medium uppercase tracking-wider text-[#555]">
              Tasks
            </span>
            <button
              type="button"
              className="flex items-center gap-0.5 rounded px-1 py-0.5 text-[10px] text-[#888] hover:bg-[#1c1c1c] hover:text-white"
              title="New chat"
              aria-label="New chat"
              onClick={async () => {
                const s = await api.sessions.create("Composer", workspace?.id);
                setSession(s);
                setSessions(await api.sessions.list());
                router.push("/workbench");
              }}
            >
              <Plus className="h-3 w-3" strokeWidth={2} />
              New
            </button>
          </div>
          <div className="mb-3 space-y-0.5" aria-label="Tasks">
            {filteredSessions.slice(0, 10).map((s) => (
              <SessionRow
                key={s.id}
                s={s}
                active={session?.id === s.id}
                onClick={() => {
                  setSession(s);
                  router.push("/workbench");
                }}
              />
            ))}
            {!filteredSessions.length && (
              <div className="px-2 py-1 text-[11px] text-[#555]">No tasks yet</div>
            )}
          </div>

          <div className="mb-1 px-2 text-[10px] font-medium uppercase tracking-wider text-[#555]">
            Projects
          </div>
          <div className="space-y-0.5" aria-label="Projects">
            {projects.map((p) => (
              <button
                key={p.id}
                type="button"
                onClick={async () => {
                  const w = await api.workspaces.list().then((xs) => xs.find((x) => x.id === p.id));
                  if (w) {
                    setWorkspace(w);
                    setTree(await api.workspaces.tree(w.id));
                  }
                }}
                className={cn(
                  "flex w-full items-center gap-1.5 rounded-md px-2 py-[5px] text-left text-[12px]",
                  workspace?.id === p.id
                    ? "bg-[#222] text-white"
                    : "text-[#9a9a9a] hover:bg-[#1c1c1c]",
                )}
              >
                <Folder className="h-3.5 w-3.5 shrink-0 text-[#c09553]" strokeWidth={1.5} />
                <span className="truncate">{p.name}</span>
              </button>
            ))}
            {workspace && (
              <div className="mt-1 border-t border-[#1f1f1f] pt-1">
                <TreeView nodes={tree} depth={0} onOpen={openPath} />
              </div>
            )}
          </div>
        </div>

        <div className="border-t border-[#1f1f1f] px-2.5 py-2">
          <div className="flex items-center justify-between text-[10px] text-[#555]">
            <span className="flex items-center gap-1">
              <Plug className="h-3 w-3" />
              {healthy ? "connected" : "offline"}
            </span>
            <span className="max-w-[120px] truncate font-mono text-[#666]" title={modelLabel || ""}>
              {modelLabel?.split("/").pop() || "no model"}
            </span>
          </div>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col bg-[#0e0e0e]">{children}</div>
    </div>
  );
}

function SessionRow({
  s,
  active,
  onClick,
}: {
  s: Session;
  active?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex w-full items-center gap-1.5 rounded-md px-2 py-[5px] text-left text-[12px]",
        active ? "bg-[#2a2a2a] text-white" : "text-[#9a9a9a] hover:bg-[#1c1c1c] hover:text-[#ddd]",
      )}
    >
      <Sparkles className="h-3 w-3 shrink-0 opacity-50" strokeWidth={1.5} />
      <span className="truncate">{s.title || "Untitled"}</span>
    </button>
  );
}

function TreeView({
  nodes,
  depth,
  onOpen,
}: {
  nodes: TreeNode[];
  depth: number;
  onOpen: (path: string) => void;
}) {
  if (!nodes?.length) return null;
  return (
    <ul>
      {nodes.map((n) => (
        <TreeNodeRow key={n.path} node={n} depth={depth} onOpen={onOpen} />
      ))}
    </ul>
  );
}

function TreeNodeRow({
  node,
  depth,
  onOpen,
}: {
  node: TreeNode;
  depth: number;
  onOpen: (path: string) => void;
}) {
  const [open, setOpen] = useState(depth < 1);
  const isDir = node.type === "dir";
  return (
    <li>
      <button
        type="button"
        onClick={() => (isDir ? setOpen((v) => !v) : onOpen(node.path))}
        className="flex w-full items-center gap-1 rounded px-1 py-[2px] text-left text-[11.5px] text-[#8a8a8a] hover:bg-[#1c1c1c] hover:text-[#ccc]"
        style={{ paddingLeft: 4 + depth * 10 }}
      >
        {isDir ? (
          open ? (
            <>
              <ChevronDown className="h-3 w-3 shrink-0 opacity-50" />
              <FolderOpen className="h-3.5 w-3.5 shrink-0 text-[#c09553]" strokeWidth={1.5} />
            </>
          ) : (
            <>
              <ChevronRight className="h-3 w-3 shrink-0 opacity-50" />
              <Folder className="h-3.5 w-3.5 shrink-0 text-[#c09553]" strokeWidth={1.5} />
            </>
          )
        ) : (
          <>
            <span className="w-3" />
            <FileCode2 className="h-3.5 w-3.5 shrink-0 opacity-60" strokeWidth={1.5} />
          </>
        )}
        <span className="truncate">{node.name}</span>
      </button>
      {isDir && open && node.children && (
        <TreeView nodes={node.children} depth={depth + 1} onOpen={onOpen} />
      )}
    </li>
  );
}
