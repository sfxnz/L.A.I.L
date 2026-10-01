"use client";

import { useEffect, useState } from "react";
import { api, type Settings } from "@/lib/api";
import { isUnauthorizedError } from "@/lib/auth-token";
import {
  Btn,
  Callout,
  Field,
  Input,
  Panel,
  PageSkeleton,
  inputCls,
} from "@/components/ui";
import { cn } from "@/lib/utils";

export default function ConfigurePage() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [loadError, setLoadError] = useState<unknown>(null);

  useEffect(() => {
    api.configure
      .get()
      .then((s) => {
        setSettings(s);
        setLoadError(null);
      })
      .catch(setLoadError);
  }, []);

  if (loadError) {
    return (
      <div className="space-y-5 lab-fade-in">
        <div className="page-header">
          <div>
            <h1 className="page-title">Configure</h1>
            <p className="page-sub">vLLM / llama.cpp · default model · backend URLs</p>
          </div>
        </div>
        <Callout
          tone={isUnauthorizedError(loadError) ? "warn" : "danger"}
          title={
            isUnauthorizedError(loadError)
              ? "LAIL_TOKEN required"
              : "Couldn’t load settings"
          }
          action={
            <Btn
              variant="secondary"
              size="sm"
              onClick={() => {
                setLoadError(null);
                api.configure
                  .get()
                  .then(setSettings)
                  .catch(setLoadError);
              }}
            >
              Retry
            </Btn>
          }
        >
          {isUnauthorizedError(loadError)
            ? "The controller is up. Paste the token in the banner, then retry."
            : loadError instanceof Error
              ? loadError.message
              : String(loadError)}
        </Callout>
      </div>
    );
  }

  if (!settings) {
    return <PageSkeleton rows={2} />;
  }

  async function save() {
    setSaving(true);
    setErr(null);
    setMsg(null);
    try {
      const next = await api.configure.put(settings!);
      setSettings(next);
      setMsg("Saved");
      setTimeout(() => setMsg(null), 2500);
    } catch (e) {
      setErr(String((e as Error).message || e));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-5 lab-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Configure</h1>
          <p className="page-sub">vLLM / llama.cpp · default model · backend URLs</p>
        </div>
      </div>

      {err && (
        <Callout tone="danger" title="Save failed" onDismiss={() => setErr(null)}>
          {err}
        </Callout>
      )}
      {msg && (
        <Callout tone="ok" title="Configuration saved">
          Defaults apply on the next serve.
        </Callout>
      )}

      <Panel className="space-y-4 p-4">
        <Field label="Default backend" htmlFor="cfg-backend">
          <select
            id="cfg-backend"
            className={cn(inputCls)}
            value={settings.defaultBackend}
            onChange={(e) => setSettings({ ...settings, defaultBackend: e.target.value })}
          >
            {Object.keys(settings.backends).map((k) => (
              <option key={k} value={k}>
                {settings.backends[k].label || k}
              </option>
            ))}
          </select>
        </Field>
        <Field
          label="Default model (fallback only)"
          htmlFor="cfg-model"
          hint="Serve always talks to whatever is live on the endpoint (/v1/models). This field is a fallback only if nothing is served yet."
        >
          <Input
            id="cfg-model"
            value={settings.defaultModel}
            onChange={(e) => setSettings({ ...settings, defaultModel: e.target.value })}
            placeholder="auto — or the first live /v1/models id"
          />
        </Field>
        <p className="text-[12px] leading-relaxed text-lab-muted">
          Hugging Face token: serve-engine reads <code>HF_TOKEN</code> or{" "}
          <code>~/.cache/huggingface/token</code> (<code>hf auth login</code>) on the Spark host.
          It is never stored or shown here.
        </p>
      </Panel>

      <Panel className="space-y-4 p-4">
        <h2 className="text-sm font-semibold tracking-[-0.01em]">Backend URLs</h2>
        {Object.entries(settings.backends).map(([key, b]) => (
          <div key={key} className="grid gap-2 sm:grid-cols-[120px_1fr_auto] sm:items-end">
            <Field label="Name">
              <div className="py-2 text-sm capitalize text-lab-text-dim">{b.label || key}</div>
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
            <label className="flex cursor-pointer items-center gap-2 pb-2 text-xs text-lab-muted">
              <input
                type="checkbox"
                className="h-3.5 w-3.5 accent-lab-accent"
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
        <Btn onClick={() => void save()} loading={saving}>
          Save configuration
        </Btn>
        {msg && !err && (
          <span className="text-sm font-medium text-lab-ok" role="status">
            {msg}
          </span>
        )}
      </div>
    </div>
  );
}
