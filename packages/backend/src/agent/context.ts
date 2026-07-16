import { listMessages } from "../controller/sessions";

export async function buildContext(sessionId: string): Promise<
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
