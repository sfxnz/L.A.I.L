/**
 * `bun test` preload: every run gets a throwaway data dir, so no test can read or
 * write the operator's data/lail.sqlite or data/lab-runs. Set before any test file
 * imports `config` (which resolves these paths once, at import time).
 */
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

const dir = mkdtempSync(join(tmpdir(), "lail-test-"));
process.env.LAIL_DATA_DIR = dir;
process.env.LAIL_DB_PATH = join(dir, "lail.sqlite");
process.on("exit", () => rmSync(dir, { recursive: true, force: true }));
