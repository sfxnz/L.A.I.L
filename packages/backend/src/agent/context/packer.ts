import { readdirSync, readFileSync, statSync, existsSync } from "fs";
import { join } from "path";
import type { ContextMention, EditorSnapshot } from "@lail/shared";
import { assertWorkspaceRelativePath } from "../tool-policy";
import { applyBudget, truncateBody } from "./budget";
import { loadIgnore, isIgnored, type IgnoreSet } from "./ignore";
import { parseMentions } from "./mentions";
import { ripgrepSearch } from "./search";
import { PRIORITY, type ContextChunk } from "./types";

const DEFAULT_MAX_FILE_CHARS = 200_000;
const DEFAULT_MAX_SEARCH_HITS = 30;
const FOLDER_LIST_LIMIT = 100;
const FOLDER_SMALL_FILE_CHARS = 8_000;

export type ContextPack = {
  chunks: ContextChunk[];
  truncated: boolean;
  droppedLabels: string[];
  systemExtra: string;
  contextMessage: { role: "system"; content: string } | null;
};

export async function buildContextPack(opts: {
  rootPath: string;
  snapshot: EditorSnapshot;
  budgetChars: number;
  maxFileChars?: number;
  maxSearchHits?: number;
  /** merge client mentions with parseMentions(message) */
  message?: string;
}): Promise<ContextPack> {
  const maxFileChars = opts.maxFileChars ?? DEFAULT_MAX_FILE_CHARS;
  const maxSearchHits = opts.maxSearchHits ?? DEFAULT_MAX_SEARCH_HITS;
  const rootPath = opts.rootPath;
  const ig = loadIgnore(rootPath);

  const mentions = dedupeMentions([
    ...(opts.snapshot.mentions ?? []),
    ...parseMentions(opts.message || ""),
  ]);

  const chunks: ContextChunk[] = [];
  /** Paths already packed as file mentions (skip re-adding as open tabs). */
  const packedFilePaths = new Set<string>();

  // 1. Selection (highest priority)
  const sel = opts.snapshot.selection;
  if (sel?.text) {
    const pathLabel = safeRelPath(sel.path) ?? sel.path ?? "selection";
    chunks.push({
      kind: "selection",
      path: pathLabel,
      label: `selection:${pathLabel}`,
      body: formatSelection(sel.path, sel.startLine, sel.endLine, sel.text),
      priority: PRIORITY.selection,
    });
  }

  // 2. Mentions
  for (const m of mentions) {
    if (m.type === "file") {
      const chunk = await packFileMention(rootPath, ig, m.path, maxFileChars);
      chunks.push(chunk);
      if (chunk.kind === "mention_file" && chunk.path) {
        packedFilePaths.add(chunk.path);
      }
    } else if (m.type === "folder") {
      chunks.push(packFolderMention(rootPath, ig, m.path, maxFileChars));
    } else if (m.type === "search") {
      chunks.push(await packSearchMention(rootPath, m.query, maxSearchHits));
    }
  }

  // 3. Open files (re-read disk; activePath → active_tab priority)
  const activePath = safeRelPath(opts.snapshot.activePath ?? undefined);
  for (const of of opts.snapshot.openFiles ?? []) {
    const rel = safeRelPath(of.path);
    if (!rel) {
      chunks.push({
        kind: "note",
        label: `open:${of.path}`,
        body: `Open file path invalid or escapes workspace: ${of.path}`,
        priority: PRIORITY.note,
      });
      continue;
    }
    if (packedFilePaths.has(rel)) continue;
    if (isIgnored(ig, rel)) continue;

    const body = readCapped(rootPath, rel, maxFileChars);
    if (body === null) {
      chunks.push({
        kind: "note",
        path: rel,
        label: `open-missing:${rel}`,
        body: `Open file not found on disk: ${rel}`,
        priority: PRIORITY.note,
      });
      continue;
    }

    const isActive = activePath === rel;
    chunks.push({
      kind: "open_tab",
      path: rel,
      label: isActive ? `active:${rel}` : `tab:${rel}`,
      body,
      priority: isActive ? PRIORITY.active_tab : PRIORITY.open_tab,
    });
  }

  // 4. Budget
  const budgeted = applyBudget(chunks, opts.budgetChars);

  // 5. Format context message
  const contextMessage = formatContextMessage(budgeted.chunks);

  // 6. systemExtra
  const systemExtra = buildSystemExtra(
    budgeted.chunks.length,
    budgeted.truncated,
    budgeted.droppedLabels,
  );

  return {
    chunks: budgeted.chunks,
    truncated: budgeted.truncated,
    droppedLabels: budgeted.droppedLabels,
    systemExtra,
    contextMessage,
  };
}

function dedupeMentions(list: ContextMention[]): ContextMention[] {
  const out: ContextMention[] = [];
  const seen = new Set<string>();
  for (const m of list) {
    const key =
      m.type === "search" ? `search:${m.query}` : `${m.type}:${m.path}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(m);
  }
  return out;
}

function safeRelPath(path: string | null | undefined): string | null {
  if (!path) return null;
  try {
    return assertWorkspaceRelativePath(path);
  } catch {
    return null;
  }
}

function formatSelection(
  path: string,
  startLine: number,
  endLine: number,
  text: string,
): string {
  return `// ${path}:${startLine}-${endLine}\n${text}`;
}

async function packFileMention(
  rootPath: string,
  ig: IgnoreSet,
  rawPath: string,
  maxFileChars: number,
): Promise<ContextChunk> {
  const rel = safeRelPath(rawPath);
  if (!rel) {
    return {
      kind: "note",
      label: `mention-invalid:${rawPath}`,
      body: `Mentioned file path invalid or escapes workspace: ${rawPath}`,
      priority: PRIORITY.note,
    };
  }
  if (isIgnored(ig, rel)) {
    return {
      kind: "note",
      path: rel,
      label: `mention-ignored:${rel}`,
      body: `Mentioned file is ignored by ignore rules: ${rel}`,
      priority: PRIORITY.note,
    };
  }
  const body = readCapped(rootPath, rel, maxFileChars);
  if (body === null) {
    return {
      kind: "note",
      path: rel,
      label: `mention-missing:${rel}`,
      body: `Mentioned file not found on disk: ${rel}`,
      priority: PRIORITY.note,
    };
  }
  return {
    kind: "mention_file",
    path: rel,
    label: `file:${rel}`,
    body,
    priority: PRIORITY.mention,
  };
}

function packFolderMention(
  rootPath: string,
  ig: IgnoreSet,
  rawPath: string,
  maxFileChars: number,
): ContextChunk {
  const rel = safeRelPath(rawPath);
  if (!rel) {
    return {
      kind: "note",
      label: `folder-invalid:${rawPath}`,
      body: `Mentioned folder path invalid or escapes workspace: ${rawPath}`,
      priority: PRIORITY.note,
    };
  }
  if (isIgnored(ig, rel)) {
    return {
      kind: "note",
      path: rel,
      label: `folder-ignored:${rel}`,
      body: `Mentioned folder is ignored: ${rel}`,
      priority: PRIORITY.note,
    };
  }

  const abs = join(rootPath, rel);
  if (!existsSync(abs)) {
    return {
      kind: "note",
      path: rel,
      label: `folder-missing:${rel}`,
      body: `Mentioned folder not found: ${rel}`,
      priority: PRIORITY.note,
    };
  }

  let st;
  try {
    st = statSync(abs);
  } catch {
    return {
      kind: "note",
      path: rel,
      label: `folder-missing:${rel}`,
      body: `Mentioned folder not readable: ${rel}`,
      priority: PRIORITY.note,
    };
  }
  if (!st.isDirectory()) {
    return {
      kind: "note",
      path: rel,
      label: `folder-not-dir:${rel}`,
      body: `Mentioned path is not a directory: ${rel}`,
      priority: PRIORITY.note,
    };
  }

  let entries: string[] = [];
  try {
    entries = readdirSync(abs).sort();
  } catch {
    return {
      kind: "note",
      path: rel,
      label: `folder-unreadable:${rel}`,
      body: `Could not list folder: ${rel}`,
      priority: PRIORITY.note,
    };
  }

  const lines: string[] = [`# folder ${rel}`];
  let listed = 0;
  for (const name of entries) {
    if (listed >= FOLDER_LIST_LIMIT) {
      lines.push(`… (${entries.length - listed} more entries omitted)`);
      break;
    }
    const childRel = `${rel}/${name}`.replace(/^\.\//, "");
    if (isIgnored(ig, childRel)) continue;
    const childAbs = join(rootPath, childRel);
    let kind = "file";
    try {
      if (statSync(childAbs).isDirectory()) kind = "dir";
    } catch {
      continue;
    }
    lines.push(`${kind === "dir" ? "d" : "f"}  ${childRel}`);
    listed++;

    // Include small non-ignored files when listing
    if (kind === "file") {
      const body = readCapped(rootPath, childRel, Math.min(FOLDER_SMALL_FILE_CHARS, maxFileChars));
      if (body !== null && body.length <= FOLDER_SMALL_FILE_CHARS) {
        lines.push("```");
        lines.push(body);
        lines.push("```");
      }
    }
  }

  return {
    kind: "mention_folder",
    path: rel,
    label: `folder:${rel}`,
    body: lines.join("\n"),
    priority: PRIORITY.mention,
  };
}

async function packSearchMention(
  rootPath: string,
  query: string,
  maxSearchHits: number,
): Promise<ContextChunk> {
  const r = await ripgrepSearch({ rootPath, query, maxHits: maxSearchHits });
  const header = r.ok
    ? `Search results for ${JSON.stringify(query)} (${r.hits} hits):`
    : `Search failed for ${JSON.stringify(query)}:`;
  return {
    kind: "mention_search",
    label: `search:${query}`,
    body: `${header}\n${r.output}`,
    priority: PRIORITY.mention,
  };
}

/** Read file and cap length; null if missing/unreadable. */
function readCapped(
  rootPath: string,
  rel: string,
  maxFileChars: number,
): string | null {
  try {
    const abs = join(rootPath, rel);
    const st = statSync(abs);
    if (!st.isFile()) return null;
    let text = readFileSync(abs, "utf8");
    if (text.length > maxFileChars) {
      text = truncateBody(text, maxFileChars);
    }
    return text;
  } catch {
    return null;
  }
}

function formatContextMessage(
  chunks: ContextChunk[],
): { role: "system"; content: string } | null {
  if (!chunks.length) return null;
  const parts: string[] = ["# Attached workspace context", ""];
  for (const c of chunks) {
    const title = c.path ? c.path : c.label;
    parts.push(`## ${title}`);
    parts.push("```");
    parts.push(c.body);
    parts.push("```");
    parts.push("");
  }
  return { role: "system", content: parts.join("\n").trimEnd() };
}

function buildSystemExtra(
  n: number,
  truncated: boolean,
  droppedLabels: string[],
): string {
  const lines = [
    "Context packing rules:",
    "- Prefer attached workspace context over guessing file contents.",
    "- Paths are workspace-relative; use tools for large edits.",
    `Attached ${n} context chunk${n === 1 ? "" : "s"}.`,
  ];
  if (truncated) {
    lines.push(
      `Context was truncated to budget${
        droppedLabels.length
          ? ` (dropped: ${droppedLabels.slice(0, 8).join(", ")}${
              droppedLabels.length > 8 ? ", …" : ""
            })`
          : ""
      }.`,
    );
  }
  return lines.join("\n");
}
