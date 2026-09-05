/**
 * Incremental server-sent-events parser. Feed decoded text as it arrives (any chunking,
 * including partial lines and a `\r\n` split across reads) and get back the `data`
 * payload of every completed event. Multi-line `data:` fields are joined with `\n`;
 * `event:`/`id:`/`retry:` fields and `:` comments are ignored. `[DONE]` is returned
 * verbatim so the caller decides what it means.
 */
export class SseParser {
  private buf = "";
  private data: string[] = [];

  feed(chunk: string): string[] {
    this.buf += chunk;
    const out: string[] = [];
    for (;;) {
      const m = /\r\n|\n|\r/.exec(this.buf);
      if (!m) break;
      // A trailing "\r" may be the first half of "\r\n" — wait for the next chunk.
      if (m[0] === "\r" && m.index === this.buf.length - 1) break;
      const line = this.buf.slice(0, m.index);
      this.buf = this.buf.slice(m.index + m[0].length);
      this.line(line, out);
    }
    return out;
  }

  /** End of stream: flush a final event that had no terminating blank line. */
  end(): string[] {
    const out: string[] = [];
    if (this.buf) {
      this.line(this.buf.replace(/\r$/, ""), out);
      this.buf = "";
    }
    this.dispatch(out);
    return out;
  }

  private line(line: string, out: string[]) {
    if (line === "") {
      this.dispatch(out);
      return;
    }
    if (line.startsWith(":")) return;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "data") this.data.push(value);
  }

  private dispatch(out: string[]) {
    if (this.data.length) out.push(this.data.join("\n"));
    this.data = [];
  }
}
