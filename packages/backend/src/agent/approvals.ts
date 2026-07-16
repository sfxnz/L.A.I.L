type Decision = "allow" | "deny";
const pending = new Map<
  string,
  { resolve: (d: Decision) => void; timer: ReturnType<typeof setTimeout> }
>();

export const approvalHub = {
  wait(approvalId: string, timeoutMs = 120_000): Promise<Decision> {
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        pending.delete(approvalId);
        resolve("deny");
      }, timeoutMs);
      pending.set(approvalId, {
        resolve: (d) => {
          clearTimeout(timer);
          pending.delete(approvalId);
          resolve(d);
        },
        timer,
      });
    });
  },
  decide(approvalId: string, decision: Decision): boolean {
    const p = pending.get(approvalId);
    if (!p) return false;
    p.resolve(decision);
    return true;
  },
};
