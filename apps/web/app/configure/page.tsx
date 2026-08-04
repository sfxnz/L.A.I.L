"use client";

import { useEffect, useState } from "react";
import { api, type Settings } from "@/lib/api";
import { Btn, Field, Input, Panel, inputCls } from "@/components/ui";
import { cn } from "@/lib/utils";

export default function ConfigurePage() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    api.configure.get().then(setSettings).catch(console.error);
  }, []);

  if (!settings) {
    return <div className="text-sm text-lab-muted">Loading…</div>;
  }

  async function save() {
    const next = await api.configure.put(settings!);
    setSettings(next);
    setMsg("Saved");
    setTimeout(() => setMsg(null), 2000);
  }

  return (
    <div className="space-y-5">
      <div className="page-header">
        <div>
          <h1 className="page-title">Settings</h1>
          <p className="page-sub">vLLM / llama.cpp · default model · HF token · context budget</p>
        </div>
      </div>

      <Panel className="space-y-4 p-4">
        <Field label="Default backend">
          <select
            className={cn(inputCls)}
            value={settings.defaultBackend}
            onChange={(e) =>
              setSettings({ ...settings, defaultBackend: e.target.value })
            }
          >
            {Object.keys(settings.backends).map((k) => (
              <option key={k} value={k}>
                {settings.backends[k].label || k}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Default model (fallback only)">
          <Input
            value={settings.defaultModel}
            onChange={(e) => setSettings({ ...settings, defaultModel: e.target.value })}
            placeholder="auto — or leave and Workbench follows Server"
          />
          <p className="mt-1 text-[11px] text-lab-muted">
            Workbench always talks to whatever is live on the Server endpoint (
            <code className="text-lab-text">/v1/models</code>). This field is updated to match
            automatically; you only need it if nothing is served yet.
          </p>
        </Field>
        <Field label="Hugging Face token (optional)">
          <Input
            type="password"
            value={settings.hfToken || ""}
            onChange={(e) => setSettings({ ...settings, hfToken: e.target.value })}
            placeholder="hf_…"
          />
        </Field>
        <Field label="Context budget (chars)">
          <Input
            type="number"
            min={2000}
            step={1000}
            value={settings.contextBudgetChars ?? 32_000}
            onChange={(e) => {
              const n = Number(e.target.value);
              setSettings({
                ...settings,
                contextBudgetChars: Number.isFinite(n) ? n : 32_000,
              });
            }}
          />
          <p className="mt-1 text-[11px] text-lab-muted">
            Max characters packed from open tabs, @mentions, and search hits per agent run
            (default 32000, min 2000).
          </p>
        </Field>
      </Panel>

      <Panel className="space-y-4 p-4">
        <h2 className="text-sm font-semibold">Backend URLs</h2>
        {Object.entries(settings.backends).map(([key, b]) => (
          <div key={key} className="grid gap-2 sm:grid-cols-[120px_1fr_auto] sm:items-end">
            <Field label="Name">
              <div className="py-2 text-sm capitalize">{b.label || key}</div>
            </Field>
            <Field label="URL">
              <Input
                value={b.url}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    backends: {
                      ...settings.backends,
                      [key]: { ...b, url: e.target.value },
                    },
                  })
                }
              />
            </Field>
            <label className="flex items-center gap-2 pb-2 text-xs text-lab-muted">
              <input
                type="checkbox"
                checked={b.enabled}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    backends: {
                      ...settings.backends,
                      [key]: { ...b, enabled: e.target.checked },
                    },
                  })
                }
              />
              Enabled
            </label>
          </div>
        ))}
      </Panel>

      <div className="flex items-center gap-3">
        <Btn onClick={save}>Save configuration</Btn>
        {msg && <span className="text-sm text-lab-accent">{msg}</span>}
      </div>
    </div>
  );
}
