let meta = #{
    name: "open-source-prep",
    description: "Get L.A.I.L ready for anyone to clone, detect their cluster, and serve any model with reliable auto-config — loop until green",
    when_to_use: "Preparing L.A.I.L for public open source, or when a stranger must serve a new model (including Qwen3.8-27B) on 1–N nodes without the author's spark1/spark2 lab.",
    phases: [
        #{ title: "Inventory", detail: "parallel audits: OSS hygiene, cluster, autoconfig, first-run" },
        #{ title: "Implement", detail: "sequential workstreams then one integration pass" },
        #{ title: "Verify", detail: "pytest, typecheck, default-cluster, corpus, docs gates" },
        #{ title: "Harden", detail: "fix residual failures until green or rounds exhausted" },
        #{ title: "Report", detail: "write a status report to scratch" },
    ],
};

// args.repo      — absolute path to this L.A.I.L worktree (required)
// args.rounds    — max harden rounds after first verify (default 4)
// args.skip_live — skip Hub-live recommend (default true for OSS CI path)
// args.license   — SPDX id written if LICENSE is missing (default Apache-2.0)

let repo = if args == () { () } else { args.repo };
if repo == () {
    pause("verification", "Pass args.repo — absolute path to the L.A.I.L worktree.");
}

let max_rounds = 4;
if args != () && args.rounds != () {
    max_rounds = args.rounds;
    if max_rounds < 1 { max_rounds = 1; }
}

let skip_live = true;
if args != () && args.skip_live == false {
    skip_live = false;
}

let license_name = "Apache-2.0";
if args != () && args.license != () {
    license_name = args.license;
}

let path_export = "export PATH=\"$HOME/.bun/bin:/home/sfxnz/.bun/bin:$PATH\"";
let venv_cmd = "if [ -f \"" + repo + "/.venv/bin/activate\" ]; then . \"" + repo + "/.venv/bin/activate\"; elif [ -f /home/sfxnz/projects/ai-lab/local-ai-lab/.venv/bin/activate ]; then . /home/sfxnz/projects/ai-lab/local-ai-lab/.venv/bin/activate; fi";
let se_cd = "cd \"" + repo + "/packages/serve-engine\" && " + path_export + " && " + venv_cmd + " && PYTHONPATH=. ";
let root_cd = "cd \"" + repo + "\" && " + path_export + " && " + venv_cmd + " ";

let findings_schema = #{
    "type": "object",
    "required": ["area", "p0", "p1"],
    "properties": #{
        "area": #{ "type": "string" },
        "p0": #{
            "type": "array",
            "maxItems": 10,
            "items": #{
                "type": "object",
                "required": ["id", "title", "files", "acceptance"],
                "properties": #{
                    "id": #{ "type": "string" },
                    "title": #{ "type": "string" },
                    "files": #{ "type": "string" },
                    "acceptance": #{ "type": "string" },
                },
            },
        },
        "p1": #{
            "type": "array",
            "maxItems": 8,
            "items": #{
                "type": "object",
                "required": ["id", "title", "files", "acceptance"],
                "properties": #{
                    "id": #{ "type": "string" },
                    "title": #{ "type": "string" },
                    "files": #{ "type": "string" },
                    "acceptance": #{ "type": "string" },
                },
            },
        },
        "already_ok": #{ "type": "array", "items": #{ "type": "string" } },
    },
};

let impl_schema = #{
    "type": "object",
    "required": ["stream", "done", "summary"],
    "properties": #{
        "stream": #{ "type": "string" },
        "done": #{ "type": "boolean" },
        "summary": #{ "type": "string" },
        "tests_added": #{ "type": "string" },
        "files_changed": #{ "type": "array", "items": #{ "type": "string" } },
        "blocked_on": #{ "type": "string" },
    },
};

let verify_schema = #{
    "type": "object",
    "required": ["pytest_pass", "gates_ok", "failures", "notes"],
    "properties": #{
        "pytest_pass": #{ "type": "boolean" },
        "typecheck_pass": #{ "type": "boolean" },
        "gates_ok": #{ "type": "boolean" },
        "live_ok": #{ "type": "boolean" },
        "failures": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 24 },
        "notes": #{ "type": "string" },
        "pytest_tail": #{ "type": "string" },
    },
};

let fix_schema = #{
    "type": "object",
    "required": ["fixed", "summary"],
    "properties": #{
        "fixed": #{ "type": "boolean" },
        "summary": #{ "type": "string" },
        "remaining": #{ "type": "array", "items": #{ "type": "string" } },
    },
};

// ═══════════════════════════════════════════════════════════════════════════
phase("Inventory");
// ═══════════════════════════════════════════════════════════════════════════

let common_inv = "";
common_inv += "Workspace (absolute): " + repo + "\n";
common_inv += "Use read_file, grep, list_dir. Do not answer from memory. Do not edit.\n";
common_inv += "A stranger must clone this repo tomorrow and serve Qwen3.8-27B (or any model) ";
common_inv += "with correct cluster accounting and best-known vLLM settings.\n";
common_inv += "Emit at most 10 P0 and 8 P1 items with id, title, files, acceptance.\n";
common_inv += "already_ok is items that are genuinely done — do not re-list them as work.\n";

let inv_jobs = [];

let p_oss = "";
p_oss += "Audit OPEN SOURCE HYGIENE for L.A.I.L.\n" + common_inv;
p_oss += "Read: README.md LICENSE (likely missing) package.json .gitignore .env.example ";
p_oss += "CHECKPOINT.md AGENTS.md docker-compose.yml Dockerfile next.config.ts ";
p_oss += "scripts/ CONTRIBUTING SECURITY CODE_OF_CONDUCT .github/\n";
p_oss += "Hunt: sfxnz, /home/sfxnz, spark1, spark2, private IPs, tailscale IPs, tokens, ";
p_oss += "package.json private:true, missing LICENSE, personal CHECKPOINT, research scratch dirs.\n";
p_oss += "area must be the string oss.\n";
inv_jobs.push(#{
    prompt: p_oss,
    label: "audit:oss",
    capability_mode: "read-only",
    agent_type: "explore",
    output_schema: findings_schema,
});

let p_cl = "";
p_cl += "Audit CLUSTER ACCOUNTING for L.A.I.L.\n" + common_inv;
p_cl += "Read: packages/serve-engine/app/services/cluster.py ";
p_cl += "packages/serve-engine/app/services/autoconfig.py (_cluster_topology, plan_placement, _apply_topology, _gpu_arch_env) ";
p_cl += "packages/serve-engine/app/services/serve.py (multi-node start/stop) ";
p_cl += "apps/web/components/ClusterPanel.tsx .env.example\n";
p_cl += "P0 if default config is spark1+spark2 with lab IPs. ";
p_cl += "P0 if unknown GPU SKU gets GB10 compile flags. ";
p_cl += "P0 if unprobed RAM defaults to 121.7 GiB for a laptop. ";
p_cl += "P0 if there is no test that a fresh clone is a single local node.\n";
p_cl += "area must be the string cluster.\n";
inv_jobs.push(#{
    prompt: p_cl,
    label: "audit:cluster",
    capability_mode: "read-only",
    agent_type: "explore",
    output_schema: findings_schema,
});

let p_ac = "";
p_ac += "Audit AUTO-CONFIG RELIABILITY so ANY model (including unreleased Qwen3.8-27B) serves well.\n" + common_inv;
p_ac += "Read: packages/serve-engine/app/services/autoconfig.py recommend() and helpers, ";
p_ac += "data/serve_overlays.json, packages/serve-engine/tests/test_autoconfig.py, ";
p_ac += "packages/serve-engine/tests/test_corpus.py, tests/corpus/*.\n";
p_ac += "A new Qwen3.8-27B must work from live card + config.json with ZERO code change ";
p_ac += "if the card has a vllm recipe, or from config.json + first-boot defaults if not. ";
p_ac += "Overlays are only for families whose card recipe is wrong on Spark.\n";
p_ac += "P0: $VARS in argv, flashinfer_b12x on mixed NVFP4, serve_blocked false positives, ";
p_ac += "Start gates diverging from recommend, no generic unknown-dense-Qwen test.\n";
p_ac += "area must be the string autoconfig.\n";
inv_jobs.push(#{
    prompt: p_ac,
    label: "audit:autoconfig",
    capability_mode: "read-only",
    agent_type: "explore",
    output_schema: findings_schema,
});

let p_ux = "";
p_ux += "Audit FIRST-RUN UX: clone → bun/python install → bun run dev → Serve auto-config → Start.\n" + common_inv;
p_ux += "Read: README.md scripts/dev.ts docker-compose.yml apps/web/app/server/page.tsx ";
p_ux += "apps/web/app/page.tsx apps/web/app/connect/page.tsx packages/backend/src/config.ts .env.example\n";
p_ux += "P0: README contradictions (Status vs Workbench as /), personal hostnames as required steps, ";
p_ux += "no CI, unpinned Python in docs, UI copy that says spark1.\n";
p_ux += "area must be the string first-run.\n";
inv_jobs.push(#{
    prompt: p_ux,
    label: "audit:first-run",
    capability_mode: "read-only",
    agent_type: "explore",
    output_schema: findings_schema,
});

let inv_results = parallel(inv_jobs);
let inv_blob = json_encode(inv_results);
log("Inventory panel returned " + inv_results.len().to_string() + " slots");

// Script owns the workstreams. Inventory informs prompts; it cannot expand forever.
let streams = [
    #{
        id: "cluster-generic",
        title: "Generic cluster: local-first, no author lab baked in",
        detail: "packages/serve-engine/app/services/cluster.py must NOT default to spark1/spark2 or 10.20.20.x / 10.100.8.x / tailscale IPs. Default topology is a single local node (id=local, local=true, vllm_url=http://127.0.0.1:8000). Optional remotes come only from LAIL_CLUSTER_JSON or a gitignored data/cluster.json. Unreachable remotes are offline and must not count as placement nodes. collect_cluster() on a fresh clone with no env: 1 local node, fabric note = no multi-node fabric, healthy based on the local node only. Keep dual-Spark as an EXAMPLE in .env.example comments, not as code defaults. Tests in packages/serve-engine/tests/test_cluster.py (new): (1) no env → 1 local node, no spark2 ssh; (2) LAIL_CLUSTER_JSON two nodes, remote ssh fail → remote offline, topology nodes_available=1; (3) two nodes both online → 2. Do not break serving_worker detection. Update tests that imported the old default only if they called collect_cluster without a monkeypatch.",
    },
    #{
        id: "autoconfig-any-hardware",
        title: "Placement and auto-config work on any probed machine",
        detail: "packages/serve-engine/app/services/autoconfig.py: (1) _gpu_arch_env must NOT emit CUTE_DSL_ARCH/sm_121a (or any GB10 pin) when the probed sku is unknown or not in _SKU_ARCH — only set arch env when the sku matches. (2) When topology is unavailable or ram_gib is missing, probe local hardware via metadata.collect_hardware() instead of assuming 121.7 GiB GB10 UMA. (3) Add an offline recommend test for a brand-new dense Qwen-like id Qwen/Qwen3.8-27B using a fixture card+config modeled on nvidia/Qwen3.6-27B (no family overlay required): no $VARS, no flashinfer_b12x unless checkpoint allows, tensor_parallel_size 1 on single-node topo, serve_blocked false when weights fit one node. (4) Existing corpus + Nemotron/DSpark/marlin tests must stay green. (5) Start gates in serve.py must still refuse when recommend.serve_blocked. Tests first, then minimal code.",
    },
    #{
        id: "oss-hygiene",
        title: "License, privacy, CI, no personal lab baked into product files",
        detail: "Add LICENSE (" + license_name + "). Set package.json private to false (keep name lail). Rewrite README License from 'Private lab software' to the SPDX license. Add CONTRIBUTING.md and SECURITY.md (short). Add .github/workflows/ci.yml: bun test (web+backend), bun run typecheck, python3 -m pytest packages/serve-engine/tests -q. Gitignore .research-scratch/ .research-tmp/ .vllm-research-tmp/. Remove hardcoded spark1 / spark1.home / 100.115.190.113 from apps/web/next.config.ts defaults (keep LAIL_DEV_ORIGINS). Do not delete CHECKPOINT.md but add a one-line banner that it is maintainer session notes, not user docs. Do not commit .env, sqlite, or run logs. Do not invent a new product name. Keep AGENTS.md coding rules.",
    },
    #{
        id: "docs-first-run",
        title: "Stranger-usable first-run docs and copy",
        detail: "Rewrite README.md Quick start so a new user on Linux+NVIDIA or Apple Silicon can clone, copy .env.example, install bun + python deps from pyproject.toml, bun install, bun run dev, open Status, Serve → Auto-configure → Start. Fix the Status vs Workbench / redirect contradiction (code currently redirects to Status per product surface table — document what the code does). Replace required sfxnz@spark1 SSH as the only path with a generic host example. .env.example: document LAIL_CLUSTER_JSON as optional; show a 2-node EXAMPLE that is clearly an example. Strip 'Install on spark1' product copy in apps/web/app/server/page.tsx to generic host wording. Keep Hermes connect snippets. Do not add a marketing site.",
    },
];

// ═══════════════════════════════════════════════════════════════════════════
phase("Implement");
// ═══════════════════════════════════════════════════════════════════════════

let stream_summaries = [];
let si = 0;
while si < streams.len() {
    let s = streams[si];
    log("Implementing stream: " + s.id);
    let p = "";
    p += "You implement ONE L.A.I.L open-source readiness workstream.\n";
    p += "Workspace: " + repo + "\n";
    p += "Stream id: " + s.id + "\n";
    p += "Title: " + s.title + "\n";
    p += "Requirements:\n" + s.detail + "\n\n";
    p += "Coding rules:\n";
    p += "- Think first. Surgical: every changed line traces to this stream.\n";
    p += "- Simplicity: minimum code. No speculative abstractions.\n";
    p += "- Tests first when behavior changes. Run them. Show output.\n";
    p += "- Verify after edits: " + se_cd + "python -m pytest tests/ -q --tb=short\n";
    p += "  and if you touch TS: " + root_cd + "bun run typecheck\n\n";
    p += "Read the current files on disk before editing — prior streams may have changed them.\n";
    p += "Do NOT revert Nemotron 20GiB / DSpark / marlin / corpus work already on this branch.\n";
    p += "Do NOT commit or push. Do NOT touch data/*.sqlite or run logs.\n";
    p += "Do NOT hardcode the author's spark1/spark2 lab as a product default.\n";
    p += "Return done=true only if acceptance is met and tests you ran passed.\n";
    p += "Inventory JSON (guidance only): " + inv_blob + "\n";

    let r = agent(p, #{
        label: "impl:" + s.id,
        capability_mode: "all",
        agent_type: "general-purpose",
        output_schema: impl_schema,
    });

    if r != () && r.success == true && r.output != () {
        let done_s = "false";
        if r.output.done == true { done_s = "true"; }
        let sum = "";
        if r.output.summary != () { sum = r.output.summary; }
        let line = s.id + ": done=" + done_s + " — " + sum;
        stream_summaries.push(line);
        log(line);
    } else {
        stream_summaries.push(s.id + ": FAILED or empty");
        log(s.id + " implement failed");
    }
    si += 1;
}

let integ = "";
integ += "You are the integration coordinator for L.A.I.L open-source prep.\n";
integ += "Workspace: " + repo + "\n";
integ += "Stream results:\n";
let sj = 0;
while sj < stream_summaries.len() {
    integ += "- " + stream_summaries[sj] + "\n";
    sj += 1;
}
integ += "\n1. Re-read cluster.py, autoconfig.py, README.md, LICENSE, next.config.ts, .env.example\n";
integ += "2. Finish any stream marked done=false or FAILED\n";
integ += "3. Confirm default cluster is local-only (grep cluster.py for spark1 must not be a default node)\n";
integ += "4. Run: " + se_cd + "python -m pytest tests/ -q --tb=line\n";
integ += "5. Run: " + root_cd + "bun run typecheck\n";
integ += "6. Fix until those are green. Do not push. Preserve Nemotron/DSpark/marlin.\n";
integ += "stream field must be the string integration.\n";

let merge = agent(integ, #{
    label: "integration-pass",
    capability_mode: "all",
    agent_type: "general-purpose",
    output_schema: impl_schema,
});
if merge != () && merge.success == true && merge.output != () && merge.output.summary != () {
    log("Integration: " + merge.output.summary);
} else {
    log("Integration agent failed — verify phase will expose gaps");
}

// ═══════════════════════════════════════════════════════════════════════════
phase("Verify");
// ═══════════════════════════════════════════════════════════════════════════

let verify_prompt = "";
verify_prompt += "Verify L.A.I.L open-source readiness in " + repo + "\n\n";
verify_prompt += "Run real commands. Do not guess pass/fail.\n";
verify_prompt += "1. " + se_cd + "python -m pytest tests/ -q --tb=short\n";
verify_prompt += "   pytest_pass = exit code 0.\n";
verify_prompt += "2. " + root_cd + "bun run typecheck\n";
verify_prompt += "   typecheck_pass = exit code 0. If bun missing, record a failure.\n";
verify_prompt += "3. Gates (gates_ok=true only if ALL hold; list each miss in failures):\n";
verify_prompt += "   - LICENSE file exists at repo root\n";
verify_prompt += "   - README does not say Private lab software\n";
verify_prompt += "   - packages/serve-engine/app/services/cluster.py default nodes do not include spark1 or spark2\n";
verify_prompt += "   - test file packages/serve-engine/tests/test_cluster.py exists\n";
verify_prompt += "   - .github/workflows/ci.yml exists\n";
verify_prompt += "   - next.config.ts default allowedDevOrigins does not include 100.115.190.113\n";
verify_prompt += "4. ";
if skip_live == true {
    verify_prompt += "skip_live=true — set live_ok=true only if corpus cases exist and pytest_pass.\n";
} else {
    verify_prompt += "Live recommend workflow_max for nvidia/Qwen3.6-27B-NVFP4 and ";
    verify_prompt += "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4. ";
    verify_prompt += "live_ok=false if serve_blocked wrongly, $ in extra_flags, or Lightning missing DSpark/marlin.\n";
}
verify_prompt += "5. Return accurate booleans and concrete failure strings.\n";

let ver = agent(verify_prompt, #{
    label: "verify-suite",
    capability_mode: "execute",
    agent_type: "general-purpose",
    output_schema: verify_schema,
});

let pytest_pass = false;
let typecheck_pass = false;
let gates_ok = false;
let live_ok = false;
let failures = [];
if ver != () && ver.success == true && ver.output != () {
    if ver.output.pytest_pass == true { pytest_pass = true; }
    if ver.output.typecheck_pass == true { typecheck_pass = true; }
    if ver.output.gates_ok == true { gates_ok = true; }
    if ver.output.live_ok == true { live_ok = true; }
    if ver.output.failures != () { failures = ver.output.failures; }
    log("Verify: pytest=" + pytest_pass.to_string() + " typecheck=" + typecheck_pass.to_string() + " gates=" + gates_ok.to_string() + " live=" + live_ok.to_string());
} else {
    failures.push("verify agent failed");
    log("Verify agent failed");
}

// ═══════════════════════════════════════════════════════════════════════════
phase("Harden");
// ═══════════════════════════════════════════════════════════════════════════

let round = 0;
while round < max_rounds {
    let all_green = false;
    if pytest_pass == true {
        if gates_ok == true {
            if live_ok == true {
                all_green = true;
            }
        }
    }
    if all_green == true {
        log("All green — skipping further harden rounds");
        break;
    }
    round += 1;
    log("Harden round " + round.to_string() + "/" + max_rounds.to_string());

    let fail_blob = json_encode(failures);
    let fix_prompt = "";
    fix_prompt += "Fix remaining L.A.I.L open-source readiness failures in " + repo + "\n";
    fix_prompt += "Round " + round.to_string() + " of " + max_rounds.to_string() + "\n";
    fix_prompt += "pytest_pass=" + pytest_pass.to_string() + " typecheck_pass=" + typecheck_pass.to_string();
    fix_prompt += " gates_ok=" + gates_ok.to_string() + " live_ok=" + live_ok.to_string() + "\n";
    fix_prompt += "Failures JSON: " + fail_blob + "\n\n";
    fix_prompt += "Surgical fixes only. Preserve Nemotron/DSpark/marlin/corpus. No spark1/spark2 defaults.\n";
    fix_prompt += "Re-run: " + se_cd + "python -m pytest tests/ -q --tb=short\n";
    fix_prompt += "Re-run: " + root_cd + "bun run typecheck\n";
    fix_prompt += "Re-check LICENSE, cluster.py defaults, test_cluster.py, ci.yml, next.config.ts, README license line.\n";

    let fix = agent(fix_prompt, #{
        label: "harden-" + round.to_string(),
        capability_mode: "all",
        agent_type: "general-purpose",
        output_schema: fix_schema,
    });

    let re_prompt = "";
    re_prompt += "Re-verify after harden round " + round.to_string() + " in " + repo + "\n";
    re_prompt += "Run: " + se_cd + "python -m pytest tests/ -q --tb=line\n";
    re_prompt += "Run: " + root_cd + "bun run typecheck\n";
    re_prompt += "Gates: LICENSE exists; README not Private lab; cluster.py default has no spark1/spark2; ";
    re_prompt += "test_cluster.py exists; .github/workflows/ci.yml exists; next.config.ts has no 100.115.190.113 default.\n";
    if skip_live == true {
        re_prompt += "skip_live: live_ok=true only if pytest_pass.\n";
    } else {
        re_prompt += "Live Lightning 30B workflow_max: blocked=false, weights 15-30, no $ in extras.\n";
    }
    re_prompt += "Return accurate pytest_pass/typecheck_pass/gates_ok/live_ok/failures.\n";

    let re = agent(re_prompt, #{
        label: "reverify-" + round.to_string(),
        capability_mode: "execute",
        agent_type: "general-purpose",
        output_schema: verify_schema,
    });

    if re != () && re.success == true && re.output != () {
        pytest_pass = false;
        typecheck_pass = false;
        gates_ok = false;
        live_ok = false;
        if re.output.pytest_pass == true { pytest_pass = true; }
        if re.output.typecheck_pass == true { typecheck_pass = true; }
        if re.output.gates_ok == true { gates_ok = true; }
        if re.output.live_ok == true { live_ok = true; }
        failures = [];
        if re.output.failures != () { failures = re.output.failures; }
        log("Reverify: pytest=" + pytest_pass.to_string() + " gates=" + gates_ok.to_string() + " live=" + live_ok.to_string());
    } else {
        failures.push("reverify failed round " + round.to_string());
    }

    if fix != () && fix.success == true && fix.output != () && fix.output.summary != () {
        log("Harden summary: " + fix.output.summary);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
phase("Report");
// ═══════════════════════════════════════════════════════════════════════════

let report = "";
report += "# L.A.I.L open-source prep report\n\n";
report += "Repo: " + repo + "\n";
report += "license: " + license_name + "\n";
report += "pytest_pass: " + pytest_pass.to_string() + "\n";
report += "typecheck_pass: " + typecheck_pass.to_string() + "\n";
report += "gates_ok: " + gates_ok.to_string() + "\n";
report += "live_ok: " + live_ok.to_string() + "\n";
report += "harden_rounds: " + round.to_string() + "\n\n";
report += "## Stream summaries\n";
let sk = 0;
while sk < stream_summaries.len() {
    report += "- " + stream_summaries[sk] + "\n";
    sk += 1;
}
report += "\n## Failures remaining\n";
if failures.len() == 0 {
    report += "(none)\n";
} else {
    let fi = 0;
    while fi < failures.len() {
        report += "- " + failures[fi] + "\n";
        fi += 1;
    }
}
report += "\n## Definition of done\n";
report += "- Anyone can clone and run without the author's hostnames or IPs\n";
report += "- Default cluster is this machine only; remotes are opt-in JSON\n";
report += "- Auto-config places from live card + probed hardware for any model\n";
report += "- Qwen3.8-27B-class dense checkpoints work without a new overlay\n";
report += "- CI + LICENSE + CONTRIBUTING exist\n";
report += "- pytest and typecheck green\n";

let report_path = write_scratch_file("open-source-prep-report.md", report);
log("Report written: " + report_path);

let final_green = false;
if pytest_pass == true {
    if gates_ok == true {
        if live_ok == true {
            final_green = true;
        }
    }
}
if final_green == true {
    complete(#{
        status: "green",
        pytest_pass: true,
        gates_ok: true,
        live_ok: true,
        report: report_path,
        summary: "L.A.I.L open-source prep tracks green",
    });
}

complete(#{
    status: "partial",
    pytest_pass: pytest_pass,
    typecheck_pass: typecheck_pass,
    gates_ok: gates_ok,
    live_ok: live_ok,
    failures: failures,
    report: report_path,
    summary: "Open-source prep incomplete — see report",
});
