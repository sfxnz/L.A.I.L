"use client";

type Handler = (event: Record<string, unknown>, channel: string) => void;

let socket: WebSocket | null = null;
const handlers = new Set<Handler>();

function wsUrl() {
  if (typeof window === "undefined") return "";
  const env = process.env.NEXT_PUBLIC_LAIL_WS;
  if (env) return env;
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  // Dev: Next on 3000, API on 8787
  if (window.location.port === "3000") {
    return `${proto}//${window.location.hostname}:8787/ws`;
  }
  return `${proto}//${window.location.host}/ws`;
}

export function connectWs() {
  if (typeof window === "undefined") return;
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return socket;
  }
  const url = wsUrl();
  socket = new WebSocket(url);
  socket.onmessage = (ev) => {
    try {
      const msg = JSON.parse(String(ev.data)) as {
        channel: string;
        event: Record<string, unknown>;
      };
      for (const h of handlers) h(msg.event, msg.channel);
    } catch {
      /* */
    }
  };
  socket.onclose = () => {
    socket = null;
    setTimeout(connectWs, 2000);
  };
  return socket;
}

export function onWsEvent(handler: Handler) {
  handlers.add(handler);
  connectWs();
  return () => {
    handlers.delete(handler);
  };
}

export function wsSubscribe(channel: string) {
  connectWs();
  const send = () => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ subscribe: channel }));
    }
  };
  if (socket?.readyState === WebSocket.OPEN) send();
  else socket?.addEventListener("open", send, { once: true });
}
