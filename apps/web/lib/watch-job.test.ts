/**
 * watchJob: a transient job-API failure is never "failed", log replays after a
 * reconnect are not duplicated, and the stream is not re-polled forever.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { api, watchJob } from "./api";

type Listener = (e: { data: string }) => void;

class FakeEventSource {
  static all: FakeEventSource[] = [];
  listeners = new Map<string, Listener[]>();
  onerror: (() => void) | null = null;
  onopen: (() => void) | null = null;
  closed = false;
  constructor(public url: string) {
    FakeEventSource.all.push(this);
  }
  addEventListener(type: string, fn: Listener) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), fn]);
  }
  emit(type: string, data: string) {
    for (const fn of this.listeners.get(type) ?? []) fn({ data });
  }
  close() {
    this.closed = true;
  }
}

const realJob = api.job;
const realES = globalThis.EventSource;
const tick = (ms = 0) => new Promise((r) => setTimeout(r, ms));

beforeEach(() => {
  FakeEventSource.all = [];
  (globalThis as unknown as { EventSource: unknown }).EventSource = FakeEventSource;
});
afterEach(() => {
  api.job = realJob;
  (globalThis as unknown as { EventSource: unknown }).EventSource = realES;
});

describe("watchJob", () => {
  test("after 3 stream errors it asks the job API once; a transient failure is 'reconnecting', not failed", async () => {
    let calls = 0;
    api.job = (() => {
      calls++;
      return Promise.reject(new Error("fetch failed"));
    }) as typeof api.job;
    const statuses: string[] = [];
    let result: unknown = "unset";
    const stop = watchJob("j1", () => {}, (s) => statuses.push(`${s.status}:${s.message}`), (r) => (result = r));
    const es = FakeEventSource.all[0];
    es.emit("status", JSON.stringify({ status: "running", progress: 0.3, message: "loading" }));
    es.onerror?.();
    es.onerror?.();
    expect(calls).toBe(0); // EventSource retries on its own first
    es.onerror?.();
    await tick();
    expect(es.closed).toBe(true); // no more EventSource retries + polls in parallel
    expect(calls).toBe(1);
    expect(statuses.at(-1)).toBe("running:reconnecting…");
    expect(result).toBe("unset");
    // further errors on the closed stream do not poll again
    es.onerror?.();
    await tick();
    expect(calls).toBe(1);
    stop();
  });

  test("when the job API answers again with a live job, the stream resumes without duplicating the log", async () => {
    api.job = (() => Promise.resolve({ job_id: "j1", status: "running", progress: 0.5, message: "loading" })) as unknown as typeof api.job;
    const logs: string[] = [];
    const stop = watchJob("j1", (c) => logs.push(c), () => {});
    const first = FakeEventSource.all[0];
    first.emit("log", "line 1\nline 2\n");
    for (let k = 0; k < 3; k++) first.onerror?.();
    await tick();
    const second = FakeEventSource.all[1];
    expect(second).toBeDefined();
    second.onopen?.();
    // the server replays the log from byte 0 on every connection
    second.emit("log", "line 1\nline 2\nline 3\n");
    expect(logs.join("")).toBe("line 1\nline 2\nline 3\n");
    stop();
  });

  test("a terminal job row finishes the watch; a 404 is the only failure it declares", async () => {
    api.job = (() => Promise.resolve({ job_id: "j1", status: "completed", progress: 1, message: "done", result: { ok: 1 } })) as unknown as typeof api.job;
    let result: unknown = null;
    watchJob("j1", () => {}, () => {}, (r) => (result = r));
    for (let k = 0; k < 3; k++) FakeEventSource.all[0].onerror?.();
    await tick();
    expect((result as { status: string }).status).toBe("completed");

    api.job = (() => Promise.reject(new Error('{"detail":"job not found"}'))) as typeof api.job;
    const statuses: string[] = [];
    watchJob("gone", () => {}, (s) => statuses.push(s.status), () => {});
    for (let k = 0; k < 3; k++) FakeEventSource.all[1].onerror?.();
    await tick();
    expect(statuses).toEqual(["failed"]);
  });
});
