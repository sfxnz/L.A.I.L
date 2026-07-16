"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { api, type UsageSummary } from "@/lib/api";
import { formatTokens } from "@/lib/utils";
import { Metric, Panel } from "@/components/ui";

const COLORS = ["#2dd4bf", "#a78bfa"];

export default function UsagePage() {
  const [u, setU] = useState<UsageSummary | null>(null);

  useEffect(() => {
    api.usage().then(setU).catch(console.error);
    const t = setInterval(() => api.usage().then(setU).catch(() => {}), 10000);
    return () => clearInterval(t);
  }, []);

  if (!u) return <div className="text-sm text-lab-muted">Loading usage…</div>;

  const mixData = [
    { name: "Prompt", value: u.mix.prompt || 0 },
    { name: "Completion", value: u.mix.completion || 0 },
  ];

  const maxHeat = Math.max(1, ...u.heatmap.map((h) => h.tokens));

  return (
    <div className="space-y-4 p-4 md:p-5">
      <div className="page-header">
        <div>
          <h1 className="page-title">Usage</h1>
          <p className="page-sub">Local token metering · proxy + Composer</p>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Metric
          label="Lifetime tokens"
          value={formatTokens(u.lifetimeTokens)}
          sub={`${u.lifetimeTokens.toLocaleString()} total`}
          accent
        />
        <Metric label="Prompt" value={formatTokens(u.lifetimePrompt)} />
        <Metric label="Completion" value={formatTokens(u.lifetimeCompletion)} />
        <Metric
          label="Models tracked"
          value={String(u.topModels.length)}
          sub="by token volume"
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel className="p-4">
          <h2 className="mb-3 text-sm font-semibold">Token activity</h2>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={u.daily}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis dataKey="date" tick={{ fill: "#71717a", fontSize: 10 }} />
                <YAxis tick={{ fill: "#71717a", fontSize: 10 }} />
                <Tooltip
                  contentStyle={{
                    background: "#18181b",
                    border: "1px solid #27272a",
                    borderRadius: 8,
                  }}
                />
                <Bar dataKey="prompt" stackId="a" fill="#2dd4bf" />
                <Bar dataKey="completion" stackId="a" fill="#a78bfa" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel className="p-4">
          <h2 className="mb-3 text-sm font-semibold">Token mix</h2>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={mixData} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80}>
                  {mixData.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "#18181b",
                    border: "1px solid #27272a",
                    borderRadius: 8,
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      <Panel className="p-4">
        <h2 className="mb-3 text-sm font-semibold">Activity heatmap</h2>
        <div className="flex flex-wrap gap-1">
          {u.heatmap.length === 0 && (
            <span className="text-sm text-lab-muted">No data yet — run Composer or proxy chat.</span>
          )}
          {u.heatmap.map((h) => {
            const intensity = h.tokens / maxHeat;
            return (
              <div
                key={h.date}
                title={`${h.date}: ${h.tokens} tokens`}
                className="h-4 w-4 rounded-sm"
                style={{
                  background: `rgba(45, 212, 191, ${0.15 + intensity * 0.85})`,
                }}
              />
            );
          })}
        </div>
      </Panel>

      <Panel className="p-4">
        <h2 className="mb-3 text-sm font-semibold">Most used models</h2>
        <div className="space-y-2">
          {u.topModels.map((m) => (
            <div
              key={m.model}
              className="flex items-center justify-between rounded-lg border border-lab-border/60 bg-black/20 px-3 py-2 text-sm"
            >
              <span className="truncate font-mono text-xs">{m.model}</span>
              <span className="text-lab-muted">
                {formatTokens(m.tokens)} · {m.calls} calls
              </span>
            </div>
          ))}
          {!u.topModels.length && (
            <div className="text-sm text-lab-muted">No model usage recorded yet.</div>
          )}
        </div>
      </Panel>
    </div>
  );
}
