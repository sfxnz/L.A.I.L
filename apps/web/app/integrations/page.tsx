"use client";

import Link from "next/link";
import { Panel } from "@/components/ui";

export default function IntegrationsPage() {
  return (
    <div className="space-y-4 p-4 md:p-5">
      <div className="page-header">
        <div>
          <h1 className="page-title">Integrations</h1>
          <p className="page-sub">Not shipped — this console is serve and evals</p>
        </div>
      </div>
      <Panel className="max-w-xl p-4">
        <p className="text-sm text-lab-muted">
          Slack, GitHub, and MCP connectors are not part of L.A.I.L. After you
          Start a model, wire{" "}
          <Link href="/connect" className="text-lab-accent underline">
            Hermes
          </Link>{" "}
          (or any OpenAI-compatible client) to the live <code>:8000</code>{" "}
          endpoint.
        </p>
      </Panel>
    </div>
  );
}
