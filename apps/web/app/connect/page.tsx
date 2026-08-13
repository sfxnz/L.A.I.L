"use client";

import { useEffect, useMemo, useState } from "react";
import { api, type LabStatus } from "@/lib/api";
import { Badge, Btn, Metric, Panel } from "@/components/ui";

function hostFromBrowser(): string {
  if (typeof window === "undefined") return "127.0.0.1";
  return window.location.hostname || "127.0.0.1";
}

export default function ConnectPage() {
  const [status, setStatus] = useState<LabStatus | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const host = hostFromBrowser();

  useEffect(() => {
    api.labStatus().then(setStatus).catch(() => {});
    const t = setInterval(() => api.labStatus().then(setStatus).catch(() => {}), 8000);
    return () => clearInterval(t);
  }, []);

  const serve = status?.serve;
  const healthy = !!(serve && !serve.unreachable && serve.healthy);
  const model = serve?.model_id || "laguna";
  const port =
    (serve?.base_url || "").match(/:(\d+)/)?.[1] ||
    "8000";

  const snippets = useMemo(() => {
    const localBase = `http://127.0.0.1:${port}/v1`;
    const tsBase = `http://${host}:${port}/v1`;
    // Prefer loopback for Hermes on same box; Tailscale for Mac
    return {
      hermesLocal: localBase,
      hermesTailscale: tsBase,
      curl: `curl -s ${localBase}/models | jq .`,
      chat: `curl -s ${localBase}/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -d '{"model":"${model}","messages":[{"role":"user","content":"ping"}],"max_tokens":32}'`,
      env: `OPENAI_BASE_URL=${localBase}\nOPENAI_API_KEY=local\nOPENAI_MODEL=${model}`,
    };
  }, [host, model, port]);

  async function copy(label: string, text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(label);
      setTimeout(() => setCopied(null), 1500);
    } catch {
      /* */
    }
  }

  return (
    <div className="space-y-5">
      <div className="page-header">
        <div>
          <h1 className="page-title">Connect</h1>
          <p className="page-sub">Point Hermes (and other clients) at the live OpenAI-compatible endpoint</p>
        </div>
        <Badge tone={healthy ? "ok" : "danger"}>{healthy ? "endpoint ready" : "serve down"}</Badge>
      </div>

      <div className="bento">
        <div className="bento-span-4">
          <Metric
            label="Status"
            value={healthy ? "Healthy" : "Down"}
            sub={serve?.base_url || "—"}
            tone={healthy ? "ok" : "danger"}
          />
        </div>
        <div className="bento-span-4">
          <Metric
            label="Model id"
            value={model.split("/").pop() || model}
            sub="Hermes model name if pinned"
          />
        </div>
        <div className="bento-span-4">
          <Metric
            label="Headroom"
            value={
              serve?.hardware?.available_gib != null
                ? `${serve.hardware.available_gib} GiB`
                : "—"
            }
            sub={serve?.headroom ? `headroom: ${serve.headroom}` : "from free -h via serve-engine"}
            tone={
              serve?.headroom === "critical"
                ? "danger"
                : serve?.headroom === "tight"
                  ? "warn"
                  : undefined
            }
          />
        </div>
      </div>

      <Panel title="Hermes on this host (same machine)">
        <div className="space-y-3 p-4">
          <p className="text-[13px] text-lab-muted">
            When Hermes runs on Spark, bind to loopback so traffic never leaves the box.
          </p>
          <CodeBlock
            label="Base URL"
            text={snippets.hermesLocal}
            copied={copied === "local"}
            onCopy={() => copy("local", snippets.hermesLocal)}
          />
          <CodeBlock
            label="Env block"
            text={snippets.env}
            copied={copied === "env"}
            onCopy={() => copy("env", snippets.env)}
          />
        </div>
      </Panel>

      <Panel title="Hermes / clients on Mac (Tailscale)">
        <div className="space-y-3 p-4">
          <p className="text-[13px] text-lab-muted">
            Use the Spark Tailscale IP (or this page&apos;s host). Ensure vLLM is published beyond 127.0.0.1 if you need remote clients — L.A.I.L default bind is often localhost only for safety.
          </p>
          <CodeBlock
            label="Base URL (page host)"
            text={snippets.hermesTailscale}
            copied={copied === "ts"}
            onCopy={() => copy("ts", snippets.hermesTailscale)}
          />
          <div className="rounded-[12px] border border-[rgba(255,214,10,0.22)] bg-[rgba(255,214,10,0.08)] px-3.5 py-2.5 text-[12px] text-lab-text-dim">
            If curl from Mac fails but Spark localhost works, the docker publish is{" "}
            <code className="rounded bg-lab-hover px-1.5 py-0.5 font-mono text-[11px] text-lab-warn">127.0.0.1:8000</code> only. Either SSH tunnel or re-serve with LAN/Tailscale bind intentionally.
          </div>
        </div>
      </Panel>

      <Panel title="Quick probes">
        <div className="space-y-3 p-4">
          <CodeBlock
            label="List models"
            text={snippets.curl}
            copied={copied === "curl"}
            onCopy={() => copy("curl", snippets.curl)}
          />
          <CodeBlock
            label="Smoke chat"
            text={snippets.chat}
            copied={copied === "chat"}
            onCopy={() => copy("chat", snippets.chat)}
          />
        </div>
      </Panel>
    </div>
  );
}

function CodeBlock({
  label,
  text,
  onCopy,
  copied,
}: {
  label: string;
  text: string;
  onCopy: () => void;
  copied?: boolean;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <span className="text-[12px] font-medium text-lab-text-dim">{label}</span>
        <Btn variant="secondary" size="sm" onClick={onCopy}>
          {copied ? "Copied" : "Copy"}
        </Btn>
      </div>
      <pre className="overflow-x-auto rounded-[12px] border border-lab-border bg-lab-editor p-3.5 font-mono text-[12px] leading-relaxed text-lab-text-dim whitespace-pre-wrap">
        {text}
      </pre>
    </div>
  );
}
