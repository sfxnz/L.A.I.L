"use client";

import Link from "next/link";
import { Panel, btnClass } from "@/components/ui";

/** Composer / agent workbench retired from primary product — Hermes is the agent. */
export default function WorkbenchRetiredPage() {
  return (
    <div className="mx-auto max-w-lg space-y-4 py-12">
      <Panel className="p-8 text-center">
        <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-lab-muted">
          Retired
        </div>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight">Workbench → Hermes</h1>
        <p className="mt-3 text-[14px] leading-relaxed text-lab-muted">
          L.A.I.L is now a serve &amp; eval console. Agentic coding and chat run against the model
          endpoint from Hermes, not inside this IDE.
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          <Link href="/server" className={btnClass("primary", "md")}>
            Serve
          </Link>
          <Link href="/evals" className={btnClass("secondary", "md")}>
            Evals
          </Link>
          <a
            href="https://sfxnz.github.io/dgx-lab/"
            target="_blank"
            rel="noreferrer"
            className={btnClass("secondary", "md")}
          >
            Public site
          </a>
        </div>
      </Panel>
    </div>
  );
}
