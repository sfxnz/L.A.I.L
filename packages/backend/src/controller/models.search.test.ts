import { describe, expect, test } from "bun:test";
import { hfSearchQuery } from "./models";

describe("hfSearchQuery", () => {
  test("empty or missing query is safetensors, never gguf", () => {
    expect(hfSearchQuery("")).toBe("safetensors");
    expect(hfSearchQuery("   ")).toBe("safetensors");
    expect(hfSearchQuery(undefined)).toBe("safetensors");
    expect(hfSearchQuery(null)).toBe("safetensors");
    expect(hfSearchQuery("")).not.toBe("gguf");
  });

  test("keeps a real query", () => {
    expect(hfSearchQuery("Qwen3.6-27B")).toBe("Qwen3.6-27B");
    expect(hfSearchQuery("  nvfp4  ")).toBe("nvfp4");
  });
});
