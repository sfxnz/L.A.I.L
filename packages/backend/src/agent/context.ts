import { listMessages } from "../controller/sessions";

export {
  buildContextPack,
  type ContextPack,
} from "./context/packer";
export type { ContextChunk } from "./context/types";

/**
 * Poison-filtered recent chat history for agent runs.
 * (Former thin Phase A `buildContext` history path.)
 */
export async function loadHistory(sessionId: string): Promise<
  Array<{ role: string; content: string }>
> {
  return listMessages(sessionId)
    .filter((m) => m.role === "user" || m.role === "assistant")
    .filter((m) => {
      if (m.role === "assistant" && /^Error:\s*LLM error/i.test(m.content)) return false;
      if (m.role === "assistant" && /model `default` does not exist/i.test(m.content)) return false;
      return true;
    })
    .slice(-20)
    .map((m) => ({ role: m.role, content: m.content }));
}

/** @deprecated Use loadHistory; kept for any old imports. */
export const buildContext = loadHistory;
