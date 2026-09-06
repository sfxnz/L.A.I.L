"use client";

import { Fragment } from "react";
import { isMacPlatform } from "@/lib/shortcuts";

/** Key caps for a shortcut. Chords ("g" then "b") get a "then" separator; ⌘ prints as Ctrl off-Mac. */
export function Keys({ keys, className }: { keys: readonly string[]; className?: string }) {
  const mac = isMacPlatform();
  const chord = keys.length > 1 && keys.every((k) => k.length === 1 && /[a-z0-9?]/i.test(k));
  return (
    <span className={className ?? "cmd-keys"} aria-label={keys.join(chord ? " then " : " ")}>
      {keys.map((k, i) => (
        <Fragment key={i}>
          {chord && i > 0 && <span className="cmd-then">then</span>}
          <kbd className="cmd-kbd">{!mac && k === "⌘" ? "Ctrl" : k}</kbd>
        </Fragment>
      ))}
    </span>
  );
}
