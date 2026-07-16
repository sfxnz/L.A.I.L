"use client";

import { Cable, Github, MessageSquare, Workflow } from "lucide-react";
import { Badge, Panel } from "@/components/ui";

const items = [
  {
    name: "GitHub",
    desc: "Repo context, PR summaries, and issue triage from Workbench.",
    icon: Github,
  },
  {
    name: "Slack / chat",
    desc: "Push lab alerts and bench results to team channels.",
    icon: MessageSquare,
  },
  {
    name: "CI workflows",
    desc: "Trigger smoke / perf gates on model updates.",
    icon: Workflow,
  },
  {
    name: "Custom tools",
    desc: "Register MCP-style tools for Composer.",
    icon: Cable,
  },
];

export default function IntegrationsPage() {
  return (
    <div className="space-y-4 p-4 md:p-5">
      <div className="page-header">
        <div>
          <h1 className="page-title">Integrations</h1>
          <p className="page-sub">Future connectors — local-first by default</p>
        </div>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {items.map((it) => (
          <Panel key={it.name} className="p-4">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-2">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-zinc-800">
                  <it.icon className="h-4 w-4 text-lab-accent" />
                </div>
                <div className="font-medium">{it.name}</div>
              </div>
              <Badge tone="muted">Coming soon</Badge>
            </div>
            <p className="mt-3 text-sm text-lab-muted">{it.desc}</p>
          </Panel>
        ))}
      </div>
    </div>
  );
}
