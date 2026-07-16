import type { ContextMention } from "./api";

/** Parse @file/@folder/@search/@code from composer text (same regex as backend). */
export function parseMentions(text: string): ContextMention[] {
  const out: ContextMention[] = [];
  const seen = new Set<string>();
  const add = (m: ContextMention) => {
    const key =
      m.type === "search" ? `search:${m.query}` : `${m.type}:${m.path}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(m);
  };

  // @file path  |  @folder path
  const fileFolder = /@(file|folder)\s+([^\s@]+)/gi;
  let m: RegExpExecArray | null;
  while ((m = fileFolder.exec(text))) {
    add({ type: m[1].toLowerCase() as "file" | "folder", path: m[2] });
  }

  // @search "query" or @search query | @code same
  const search =
    /@(search|code)\s+(?:"([^"]+)"|'([^']+)'|([^\s@]+))/gi;
  while ((m = search.exec(text))) {
    const q = (m[2] ?? m[3] ?? m[4] ?? "").trim();
    if (q) add({ type: "search", query: q });
  }
  return out;
}
