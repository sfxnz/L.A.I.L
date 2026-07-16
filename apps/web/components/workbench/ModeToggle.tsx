"use client";

import type { AgentMode } from "@/lib/api";
import { AGENT_MODES, AGENT_MODE_LABELS } from "@/lib/ide-chrome";
import { cn } from "@/lib/utils";

export function ModeToggle({
  value,
  onChange,
  disabled,
}: {
  value: AgentMode;
  onChange: (m: AgentMode) => void;
  disabled?: boolean;
}) {
  return (
    <div
      className="inline-flex items-center rounded-md border border-[#2a2a2a] bg-[#181818] p-0.5"
      role="group"
      aria-label="Agent mode"
    >
      {AGENT_MODES.map((mode) => {
        const active = value === mode;
        return (
          <button
            key={mode}
            type="button"
            disabled={disabled}
            aria-pressed={active}
            onClick={() => onChange(mode)}
            className={cn(
              "rounded px-2 py-0.5 text-[11px] font-medium transition",
              active
                ? "bg-[#2a2a2a] text-[#e8e8e8]"
                : "text-[#777] hover:text-[#ccc]",
              disabled && "cursor-not-allowed opacity-50",
            )}
          >
            {AGENT_MODE_LABELS[mode]}
          </button>
        );
      })}
    </div>
  );
}
