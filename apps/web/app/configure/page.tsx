"use client";

import { useEffect, useState } from "react";
import { api, type Settings } from "@/lib/api";
import { Btn, Field, Input, Panel } from "@/components/ui";

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
    <div className="mx-auto max-w-2xl space-y-4 p-4 md:p-5">
      <div className="page-header">
        <div>
          <h1 className="page-title">Settings</h1>
          <p className="page-sub">vLLM / llama.cpp · default model · HF token</p>
        </div>
      </div>

      <Panel className="space-y-4 p-4">
        <Field label="Default backend">
          <select
            className="w-full rounded-lg border border-lab-border bg-zinc-900 px-3 py-2 text-sm"
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
        <Field label="Default model">
          <Input
            value={settings.defaultModel}
            onChange={(e) => setSettings({ ...settings, defaultModel: e.target.value })}
          />
        </Field>
        <Field label="Hugging Face token (optional)">
          <Input
            type="password"
            value={settings.hfToken || ""}
            onChange={(e) => setSettings({ ...settings, hfToken: e.target.value })}
            placeholder="hf_…"
          />
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
