import { describe, expect, test } from "bun:test";
import { approvalHub } from "./approvals";

describe("approvalHub", () => {
  test("resolves allow", async () => {
    const p = approvalHub.wait("a1", 5_000);
    const ok = approvalHub.decide("a1", "allow");
    expect(ok).toBe(true);
    expect(await p).toBe("allow");
  });

  test("timeout denies (50ms)", async () => {
    const decision = await approvalHub.wait("a-timeout", 50);
    expect(decision).toBe("deny");
    // late decide after timeout should fail
    expect(approvalHub.decide("a-timeout", "allow")).toBe(false);
  });
});
