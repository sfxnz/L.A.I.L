import { readFileSync } from "fs";
import { join } from "path";

const DEFAULTS = ["node_modules/", ".git/", ".venv/", "dist/", "build/"];
const BINARY_EXT = new Set([
  ".png",
  ".jpg",
  ".jpeg",
  ".gif",
  ".woff",
  ".woff2",
  ".sqlite",
  ".wasm",
  ".pdf",
]);

export type IgnoreSet = { patterns: string[] };

function readIgnoreFile(rootPath: string, name: string): string[] {
  try {
    const text = readFileSync(join(rootPath, name), "utf8");
    return text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0 && !line.startsWith("#"));
  } catch {
    return [];
  }
}

export function loadIgnore(rootPath: string): IgnoreSet {
  const fromGit = readIgnoreFile(rootPath, ".gitignore");
  const fromLail = readIgnoreFile(rootPath, ".lailignore");
  return { patterns: [...DEFAULTS, ...fromGit, ...fromLail] };
}

function matchPattern(pat: string, p: string): boolean {
  // Directory pattern: trailing /
  if (pat.endsWith("/")) {
    const dir = pat.slice(0, -1);
    return p === dir || p.startsWith(dir + "/") || p.includes("/" + dir + "/");
  }
  // Glob: *.ext
  if (pat.startsWith("*.")) {
    const suffix = pat.slice(1); // .ext
    return p.endsWith(suffix) || p.includes(suffix + "/");
  }
  // Exact or prefix
  if (p === pat || p.startsWith(pat + "/") || p.endsWith("/" + pat)) return true;
  if (p.includes("/" + pat + "/") || p.includes("/" + pat)) return true;
  return false;
}

export function isIgnored(ig: IgnoreSet, relPath: string): boolean {
  const p = relPath.replace(/\\/g, "/").replace(/^\.\//, "");
  const ext = p.includes(".") ? p.slice(p.lastIndexOf(".")) : "";
  if (BINARY_EXT.has(ext.toLowerCase())) return true;
  for (const pat of ig.patterns) {
    if (matchPattern(pat, p)) return true;
  }
  return false;
}
