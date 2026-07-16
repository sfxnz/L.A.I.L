"use client";

import { useState } from "react";
import { ShieldAlert } from "lucide-react";
import { api } from "@/lib/api";
import { useLabStore } from "@/lib/store";
import { cn } from "@/lib/utils";

export function ShellApprovalBanner() {
  const shellApproval = useLabStore((s) => s.shellApproval);
  const setShellApproval = useLabStore((s) => s.setShellApproval);
  const [busy, setBusy] = useState(false);

  if (!shellApproval) return null;

  async function decide(decision: "allow" | "deny") {
    if (!shellApproval) return;
    setBusy(true);
    try {
      await api.shellApproval(
        shellApproval.runId,
        shellApproval.approvalId,
        decision,
      );
      setShellApproval(null);
    } catch (e) {
      console.error(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="mb-2 rounded-xl border border-[#5a4020] bg-[#1a1510] px-3 py-2.5"
      data-testid="shell-approval"
      role="alertdialog"
      aria-label="Shell command approval"
    >
      <div className="flex items-start gap-2">
        <ShieldAlert className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[#d4a017]" strokeWidth={1.5} />
        <div className="min-w-0 flex-1">
          <div className="text-[11px] font-medium text-[#e0c070]">
            Shell approval required
          </div>
          <pre className="mt-1 max-h-20 overflow-auto whitespace-pre-wrap break-all font-mono text-[11px] text-[#cfcfcf]">
            {shellApproval.command}
          </pre>
          <div className="mt-2 flex items-center gap-2">
            <button
              type="button"
              disabled={busy}
              onClick={() => void decide("allow")}
              className={cn(
                "rounded border border-[#2a4a2a] bg-[#1a2a1a] px-2.5 py-1 text-[11px] text-[#8fbc8f] hover:bg-[#223322]",
                busy && "opacity-50",
              )}
            >
              Allow
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => void decide("deny")}
              className={cn(
                "rounded border border-[#4a2a2a] bg-[#1a1010] px-2.5 py-1 text-[11px] text-[#e07070] hover:bg-[#2a1515]",
                busy && "opacity-50",
              )}
            >
              Deny
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
