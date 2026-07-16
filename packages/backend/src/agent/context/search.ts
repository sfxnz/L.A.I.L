export async function ripgrepSearch(opts: {
  rootPath: string;
  query: string;
  maxHits?: number;
}): Promise<{ ok: boolean; output: string; hits: number }> {
  const max = opts.maxHits ?? 30;
  const query = String(opts.query || "").trim();
  if (!query) {
    return { ok: true, output: "(no matches)", hits: 0 };
  }
  try {
    const proc = Bun.spawn(
      [
        "rg",
        "-n",
        "--max-count",
        "5",
        "--max-filesize",
        "200K",
        "-m",
        String(max),
        query,
        ".",
      ],
      { cwd: opts.rootPath, stdout: "pipe", stderr: "pipe" },
    );
    const out = await new Response(proc.stdout).text();
    await proc.exited;
    if (proc.exitCode === 1 && !out.trim()) {
      return { ok: true, output: "(no matches)", hits: 0 };
    }
    if (proc.exitCode !== 0 && proc.exitCode !== 1) {
      return { ok: false, output: "ripgrep failed or unavailable", hits: 0 };
    }
    const lines = out.split("\n").filter(Boolean).slice(0, max);
    return {
      ok: true,
      output: lines.join("\n") || "(no matches)",
      hits: lines.length,
    };
  } catch {
    return { ok: false, output: "ripgrep unavailable", hits: 0 };
  }
}
