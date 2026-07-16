type Client = {
  id: string;
  send: (data: string) => void;
  channels: Set<string>;
};

const clients = new Map<string, Client>();

export const wsHub = {
  add(id: string, send: (data: string) => void) {
    clients.set(id, { id, send, channels: new Set(["*"]) });
  },
  remove(id: string) {
    clients.delete(id);
  },
  subscribe(id: string, channel: string) {
    clients.get(id)?.channels.add(channel);
  },
  publish(channel: string, event: unknown) {
    const payload = JSON.stringify({ channel, event });
    for (const c of clients.values()) {
      if (c.channels.has("*") || c.channels.has(channel)) {
        try {
          c.send(payload);
        } catch {
          /* drop */
        }
      }
    }
  },
};
