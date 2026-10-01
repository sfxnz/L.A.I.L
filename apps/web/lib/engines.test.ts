import { describe, expect, test } from "bun:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { EndpointHero } from "../components/status/EndpointHero";
import { EngineSelect, EngineSummary, engineFields, engineHas } from "../components/serve/EngineSelect";
import type { LabStatus, ServeEngine } from "./api";
import { engineLabel } from "./engines";

type Serve = NonNullable<LabStatus["serve"]>;

const serve = (engine: string | null, baseUrl: string): Serve => ({
  healthy: true,
  base_url: baseUrl,
  model_id: "zai-org/GLM-5.3-Flash",
  engine: { name: engine, version: "0.5.20" },
});

const hero = (s: Serve) => renderToStaticMarkup(createElement(EndpointHero, { serve: s, endpoint: [], serverNow: null }));

describe("engine identity", () => {
  test("labels the detected engine", () => {
    expect(engineLabel("sglang")).toBe("SGLang");
    expect(engineLabel("llamacpp")).toBe("llama.cpp");
    expect(engineLabel("tensorfold")).toBe("TensorFold");
    expect(engineLabel(null)).toBeNull();
  });

  test("Served model names what answers and its endpoint, not the configured default backend", () => {
    const html = hero(serve("sglang", "http://127.0.0.1:30000"));
    expect(html).toContain("SGLang");
    expect(html).not.toContain(">vLLM<");
    expect(html).toContain("http://127.0.0.1:30000/v1");
    expect(html).toContain("v0.5.20");
  });

  test("spec-decode acceptance and TTFT sit next to the engine's other live readings", () => {
    const s = serve("vllm", "http://127.0.0.1:8000");
    s.metrics = { spec_accept_rate: 0.71, spec_tokens_per_step: 3.13, ttft_s: 0.182, itl_p50_s: 0.0413, itl_p95_s: 0.0697 };
    const live = hero(s);
    expect(live).toContain("ITL p50 · p95");
    expect(live).toContain('41<span class="text-lab-muted"> · 70</span> ms');
    expect(live).toContain("71%");
    expect(live).toContain("3.13/step");
    expect(live).toContain("182 ms");
    s.metrics = { spec_accept_rate: null, spec_accept_rate_lifetime: 0.752, ttft_s: null };
    expect(hero(s)).toContain("75% lifetime");
  });

  test("an engine nothing identified shows a state word, never a guessed vLLM", () => {
    const html = hero(serve(null, "http://127.0.0.1:8000"));
    expect(html).not.toContain(">vLLM<");
    // the engine slot holds the state word: one more "Awaiting" than with an identified engine
    const awaiting = (h: string) => h.split("Awaiting").length;
    expect(awaiting(html)).toBe(awaiting(hero(serve("vllm", "http://127.0.0.1:8000"))) + 1);
  });
});

describe("engine select", () => {
  const engines: ServeEngine[] = [
    { name: "vllm", label: "vLLM", default_port: 8000, default_image: "vllm/vllm-openai:v0.27.1", image_env: "", max_tp: null, fields: ["util", "moe_backend"], notes: "" },
    { name: "tensorfold", label: "TensorFold", default_port: 8090, default_image: "nvcr.io/nvidia/pytorch:26.07-py3", image_env: "", max_tp: 2, fields: ["max_model_len"], notes: "" },
  ];

  test("offers every engine and shows the selected one's image, port and TP cap", () => {
    const html = renderToStaticMarkup(createElement(EngineSelect, { engines, value: "tensorfold", onChange: () => {} }));
    expect(html).toContain("TensorFold");
    expect(html).toContain("vLLM");
    const summary = renderToStaticMarkup(createElement(EngineSummary, { engine: engines[1] }));
    expect(summary).toContain("port 8090");
    expect(summary).toContain("TP ≤ 2");
    expect(summary).toContain("nvcr.io/nvidia/pytorch:26.07-py3");
  });

  test("hides only the fields an engine cannot translate", () => {
    expect(engineHas(engines[1], "moe_backend")).toBe(false);
    expect(engineHas(engines[1], "max_model_len")).toBe(true);
    expect(engineHas(undefined, "moe_backend")).toBe(true); // engine list not loaded: show everything
  });

  test("Start sends only the fields the selected engine translates", () => {
    // A vLLM recommendation left TP=2 and a MoE backend in the form; llama.cpp-style
    // engines without those fields must never receive them.
    const form = { tensor_parallel_size: 2, moe_backend: "marlin", max_model_len: 32768 };
    expect(engineFields(engines[1], form)).toEqual({ max_model_len: 32768 });
    expect(engineFields(engines[0], form)).toEqual({ moe_backend: "marlin" });
    expect(engineFields(undefined, form)).toEqual(form);
  });
});
