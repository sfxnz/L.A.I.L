"use client";

type Handler = (event: Record<string, unknown>, channel: string) => void;

let socket: WebSocket | null = null;
const handlers = new Set<Handler>();

function wsUrl() {
  if (typeof window === "undefined") return "";
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const pageHost = window.location.hostname;
  const env = process.env.NEXT_PUBLIC_LAIL_WS?.trim();

  // Prefer same host as the page (Tailscale / LAN / hostname). Only honor
  // NEXT_PUBLIC_LAIL_WS when it targets the host the user is actually browsing,
  // otherwise Mac→Spark via 100.x breaks: env was baked as ws://127.0.0.1:8787
  // and the browser connects to the *laptop* loopback, not the lab.
  if (env) {
    try {
      const u = new URL(env);
      const local =
        pageHost === "localhost" ||
        pageHost === "127.0.0.1" ||
        pageHost === "::1";
      if (u.hostname === pageHost || (local && (u.hostname === "127.0.0.1" || u.hostname === "localhost"))) {
        return env;
      }
    } catch {
      /* fall through */
    }
  }

  // Dev: Next on 3000, controller WS on 8787
  if (window.location.port === "3000") {
    return `${proto}//${pageHost}:8787/ws`;
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
