import { afterEach, describe, expect, test } from "bun:test";
import { getSettings, putSettings } from "./settings";

describe("settings", () => {
  const before = getSettings();
  afterEach(() => {
    putSettings(before);
  });

  test("every engine backend survives a save; SGLang and TensorFold are not stripped", () => {
    expect(Object.keys(getSettings().backends)).toEqual(["vllm", "sglang", "llamacpp", "tensorfold"]);
    // TensorFold's upstream default :8080 is llama.cpp's; L.A.I.L serves it on :8090.
    expect(getSettings().backends.tensorfold.url).toBe("http://127.0.0.1:8090");
    expect(getSettings().backends.sglang.url).toBe("http://127.0.0.1:30000");
    const saved = putSettings({
      defaultBackend: "sglang",
      backends: { ...getSettings().backends, sglang: { url: "http://127.0.0.1:30001", enabled: true, label: "SGLang" } },
    });
    expect(saved.defaultBackend).toBe("sglang");
    expect(getSettings().backends.sglang.url).toBe("http://127.0.0.1:30001");
  });

  test("legacy backends are still dropped and an unknown default falls back", () => {
    const saved = putSettings({
      defaultBackend: "ollama" as never,
      backends: { ...getSettings().backends, ollama: { url: "http://x", enabled: true, label: "x" } } as never,
    });
    expect(saved.defaultBackend).toBe("vllm");
    expect(Object.keys(saved.backends)).not.toContain("ollama");
  });
});
