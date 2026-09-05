/**
 * Token guard: every `*-lab-<name>` colour utility used in apps/web/**\/*.tsx
 * must have a declared `--color-lab-<name>` in app/globals.css. An undeclared
 * utility compiles to nothing (this is how `bg-lab-surface` shipped invisible).
 */
import { describe, expect, test } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

const webRoot = join(import.meta.dir, "..");
const css = readFileSync(join(webRoot, "app/globals.css"), "utf8");

const declared = new Set(
  [...css.matchAll(/--color-lab-([a-z0-9-]+)\s*:/g)].map((m) => m[1]),
);

// Prefix families Tailwind maps onto the colour namespace, incl. sided borders
// (border-l-lab-accent) and opacity modifiers (text-lab-text-dim/80, which the
// pattern stops before).
const USE_RE =
  /(?:^|[\s"'`:!/([,])(?:bg|text|border(?:-[trblxyse])?|from|via|to|stroke|fill|ring|outline|shadow|decoration|accent|caret|divide|placeholder)-lab-([a-z0-9]+(?:-[a-z0-9]+)*)/g;

function tsxFiles(): string[] {
  const glob = new Bun.Glob("**/*.tsx");
  return [...glob.scanSync({ cwd: webRoot })].filter(
    (p) => !p.startsWith("node_modules/") && !p.startsWith(".next/"),
  );
}

describe("lab-* colour utilities resolve to declared tokens", () => {
  const files = tsxFiles();

  test("scans the app", () => {
    expect(files.length).toBeGreaterThan(10);
    expect(declared.has("surface")).toBe(true);
    expect(declared.has("target")).toBe(true);
  });

  test("every used lab-* colour is declared in globals.css", () => {
    const missing = new Map<string, Set<string>>();
    for (const rel of files) {
      const src = readFileSync(join(webRoot, rel), "utf8");
      for (const m of src.matchAll(USE_RE)) {
        const name = m[1];
        if (!declared.has(name)) {
          if (!missing.has(name)) missing.set(name, new Set());
          missing.get(name)!.add(rel);
        }
      }
    }
    const report = [...missing].map(([n, f]) => `--color-lab-${n} ← ${[...f].join(", ")}`);
    expect(report).toEqual([]);
  });

  test("both worlds declare every token the theme contract names", () => {
    const themeBlock = css.slice(css.indexOf("@theme {"), css.indexOf("}", css.indexOf("@theme {")));
    const contract = [...themeBlock.matchAll(/--color-lab-([a-z0-9-]+)\s*:/g)].map((m) => m[1]);
    const light = css.slice(css.indexOf('[data-theme="light"] {'));
    const dark = css.slice(css.indexOf('[data-theme="dark"] {'), css.indexOf('[data-theme="light"] {'));
    for (const name of contract) {
      expect(dark.includes(`--color-lab-${name}:`)).toBe(true);
      expect(light.includes(`--color-lab-${name}:`)).toBe(true);
    }
  });
});
