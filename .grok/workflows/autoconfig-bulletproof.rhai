let meta = #{
    name: "autoconfig-bulletproof",
    description: "Implement remaining L.A.I.L auto-config bulletproof work: corpus, images, cookbooks, overlays, parsers — loop until tests green",
    when_to_use: "After Nemotron placement fixes; when auto-config must handle new models end-to-end (card, image, quant, placement, corpus).",
    phases: [
        #{ title: "Inventory", detail: "map remaining gaps vs bulletproof definition of done" },
        #{ title: "Implement", detail: "parallel workstreams with isolated edits then merge" },
        #{ title: "Verify", detail: "pytest + live recommend matrix + corpus expansion" },
        #{ title: "Harden", detail: "fix residual failures until green or budget exhausted" },
        #{ title: "Report", detail: "write status report to scratch" },
    ],
};

// ── args ────────────────────────────────────────────────────────────────────
// args.repo       — absolute path to the lail worktree (required)
// args.rounds     — max harden rounds after first implement+verify (default 3)
// args.skip_live  — if true, skip Hub-live recommend matrix (default false)

let repo = if args == () { () } else { args.repo };
if repo == () {
    pause("verification", "Pass args.repo — absolute path to the auto-config-bug (or lail) worktree.");
}

let max_rounds = 3;
if args != () && args.rounds != () {
    // Coerce JSON number safely for dry-run / live.
    max_rounds = args.rounds;
    if max_rounds < 1 {
        max_rounds = 1;
    }
}
let skip_live = false;
if args != () && args.skip_live == true {
    skip_live = true;
}

let path_export = "export PATH=\"/home/sfxnz/.bun/bin:$PATH\"";
let venv = "source /home/sfxnz/projects/ai-lab/local-ai-lab/.venv/bin/activate";
let se_cd = "cd " + repo + "/packages/serve-engine && " + path_export + " && " + venv + " && PYTHONPATH=. ";

// ── schemas ─────────────────────────────────────────────────────────────────
let gaps_schema = #{
    "type": "object",
    "required": ["p0", "p1", "definition_of_done"],
    "properties": #{
        "definition_of_done": #{ "type": "string" },
        "p0": #{
            "type": "array",
            "maxItems": 12,
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
            "maxItems": 12,
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
        "already_done": #{ "type": "array", "items": #{ "type": "string" } },
    },
};

let impl_schema = #{
    "type": "object",
    "required": ["stream", "done", "summary", "tests_added"],
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
    "required": ["pytest_pass", "live_ok", "failures", "notes"],
    "properties": #{
        "pytest_pass": #{ "type": "boolean" },
        "live_ok": #{ "type": "boolean" },
        "failures": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 20 },
        "notes": #{ "type": "string" },
        "pytest_output_tail": #{ "type": "string" },
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

let inv_prompt = "";
inv_prompt += "You are inventorying remaining L.A.I.L auto-config bulletproof work.\n";
inv_prompt += "Workspace (absolute): " + repo + "\n\n";
inv_prompt += "ALREADY LANDED (do NOT re-implement):\n";
inv_prompt += "- Weight floors no longer override Hub blob sums; compact *B-A*B MoEs skip 400GiB expert floor\n";
inv_prompt += "- _expand_card_exports before scrub; DSpark draft+method restore for Nemotron Lightning\n";
inv_prompt += "- config.json ModelOpt sibling overrides Ampere prose modelopt_fp4\n";
inv_prompt += "- placement util_computed only when weights need more than mode default\n";
inv_prompt += "- serve.py keeps Nemotron marlin via analyze_config({}, model)\n";
inv_prompt += "- check_serve_loadability shared for Lab Safe util + absolute fit\n";
inv_prompt += "- capability min stock image for modelopt_mixed/marlin/new parsers\n";
inv_prompt += "- Nemotron corpus expects serve_blocked=false, weights~20, modelopt_mixed\n\n";
inv_prompt += "DEFINITION OF DONE (bulletproof):\n";
inv_prompt += "For major open-weight families that fit 1–2× GB10: correct image, quant/moe/KV, parsers,\n";
inv_prompt += "placement from real weights, Lab Safe never Start-illegal, final argv free of $VARS,\n";
inv_prompt += "offline corpus case per family, Start mirrors recommend gates.\n\n";
inv_prompt += "Read these with tools (read_file/grep/list_dir) before answering:\n";
inv_prompt += "- packages/serve-engine/app/services/autoconfig.py (recommend, overlays, image, fetch)\n";
inv_prompt += "- packages/serve-engine/app/services/serve.py (Start gates, entrypoint, marlin)\n";
inv_prompt += "- packages/serve-engine/tests/test_corpus.py and tests/corpus/*\n";
inv_prompt += "- data/serve_overlays.json\n";
inv_prompt += "- .research-scratch/findings/19-autoconfig-gaps.md if present\n\n";
inv_prompt += "Emit p0 (must fix for bulletproof) and p1 (high leverage) items with id, title, files, acceptance.\n";
inv_prompt += "Prioritize these known remaining gaps if still open in code:\n";
inv_prompt += "1. Expand offline corpus: Unsloth 35B, dense Qwen NVFP4, MiniMax M2, Magistral, optional Nano-9B\n";
inv_prompt += "2. Propagate card/docker image into ServeCandidate + earlier resolve\n";
inv_prompt += "3. Optional GitHub cookbook fetch when HF card has no vllm serve recipes\n";
inv_prompt += "4. Overlay merge: no duplicate tokenizer/config flags (Magistral)\n";
inv_prompt += "5. Nano-9B-v2 tool parser (nemotron_json vs qwen3_coder) from card scoring\n";
inv_prompt += "6. Capture script: hub blobs / weights_gib into corpus fixtures\n";
inv_prompt += "7. Single-node Anemll HF_HOME mount parity with multi-node\n";
inv_prompt += "Do not invent work already done. Use read-only tools.\n";

let inv = agent(inv_prompt, #{
    label: "inventory-gaps",
    capability_mode: "read-only",
    agent_type: "explore",
    output_schema: gaps_schema,
});

if inv == () || inv.success != true || inv.output == () {
    pause("infra", "Inventory agent failed — resume to retry or inspect /workflows.");
}

let p0_n = 0;
let p1_n = 0;
if inv.output.p0 != () { p0_n = inv.output.p0.len(); }
if inv.output.p1 != () { p1_n = inv.output.p1.len(); }
log("Inventory: P0=" + p0_n.to_string() + " P1=" + p1_n.to_string());

// Fixed workstream list — script owns scope; inventory informs prompts but cannot expand forever.
let streams = [
    #{
        id: "corpus",
        title: "Expand offline corpus + capture weights",
        detail: "Add corpus cases for unsloth/Qwen3.6-35B-A3B-NVFP4, nvidia/Qwen3.6-27B-NVFP4, Magistral (or mistral overlay), MiniMax M2 if public. Extend _capture.py for hub_api/weights_gib. Assert serve_blocked + weights bounds where applicable. Update test_corpus.py expects.",
    },
    #{
        id: "image-card",
        title: "Card/docker image into candidates + capability resolve",
        detail: "Parse docker run image into ServeCandidate.config.image; expand _CARD_IMAGE_RE for nvcr/vllm tags; apply candidate image before safety/marlin gates; never replace Anemll; never downgrade lab default without card/capability raise.",
    },
    #{
        id: "cookbook",
        title: "GitHub cookbook fetch when card recipe-poor",
        detail: "If extract_serve_candidates empty/weak, scan README for public github raw/cookbook URLs, fetch with timeout/size cap, re-extract candidates, add sources. Never override family overlay. Tests with fixture readme linking a local fixture body.",
    },
    #{
        id: "overlay-parsers",
        title: "Overlay merge hygiene + Nano-9B parsers",
        detail: "Deduplicate overlay extra_flags/docker_env when merging. Fix MiniMax overlay label if FP8 vs NVFP4. Improve Nemotron Nano-9B-v2: prefer card tool-call parser (nemotron_json) when present in higher-scoring or tool-focused recipes; unit tests.",
    },
    #{
        id: "serve-parity",
        title: "serve.py Anemll single-node HF mount + TP unify",
        detail: "Align single-node Anemll HF cache mount/env with multi-node (/cache/huggingface + HF_HOME). Unify TP used by fits gate and launch. Quote worker docker pull image. Tests for entrypoint + mount when possible without docker.",
    },
];

// ═══════════════════════════════════════════════════════════════════════════
phase("Implement");
// ═══════════════════════════════════════════════════════════════════════════
// Sequential in the parent workspace: every stream edits autoconfig.py / tests;
// parallel isolation worktrees would not auto-merge and would thrash the same files.

let stream_summaries = [];
for s in streams {
    log("Implementing stream: " + s.id);
    let p = "";
    p += "You implement ONE auto-config workstream in the L.A.I.L monorepo.\n";
    p += "Workspace: " + repo + "\n";
    p += "Stream id: " + s.id + "\n";
    p += "Title: " + s.title + "\n";
    p += "Requirements:\n" + s.detail + "\n\n";
    p += "Coding rules (AGENTS.md):\n";
    p += "- Surgical: only change what this stream needs; no drive-by refactors\n";
    p += "- Simplicity: minimum code that solves the problem\n";
    p += "- TDD: write/adjust failing tests first when behavior changes\n";
    p += "- Verify: run `" + se_cd + "python -m pytest tests/test_autoconfig.py tests/test_corpus.py -q`\n";
    p += "  after your edits and fix failures you introduced.\n\n";
    p += "Read packages/serve-engine/app/services/autoconfig.py and serve.py and existing tests first.\n";
    p += "Do NOT revert Nemotron weight/DSpark/marlin fixes already on main.\n";
    p += "Do NOT commit or push. Do NOT touch data/*.sqlite or run logs.\n";
    p += "Prior streams may already have edited the same files — read current disk state and build on it.\n";
    p += "If blocked (missing HF network for capture), leave a fixture-based offline path.\n";
    p += "Return done=true only if the stream's acceptance is met and your tests pass.\n";
    p += "Inventory context (guidance only): " + json_encode(inv.output) + "\n";

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
}

// Integration pass: ensure all streams compose and tests are green once.
let integ_prompt = "";
integ_prompt += "You are the integration coordinator for autoconfig bulletproof streams.\n";
integ_prompt += "Workspace: " + repo + "\n";
integ_prompt += "Stream results:\n";
for line in stream_summaries {
    integ_prompt += "- " + line + "\n";
}
integ_prompt += "\n1. Re-read autoconfig.py / serve.py / tests/corpus for incomplete streams\n";
integ_prompt += "2. Finish any stream marked done=false or FAILED\n";
integ_prompt += "3. Run: " + se_cd + "python -m pytest tests/ -q --tb=line\n";
integ_prompt += "4. Fix until green; do not push; optional one clean commit if green\n";
integ_prompt += "5. Preserve Nemotron 20GiB / DSpark / marlin fixes\n";

let merge = agent(integ_prompt, #{
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
verify_prompt += "Verify auto-config bulletproof progress in " + repo + "\n\n";
verify_prompt += "1. Run: " + se_cd + "python -m pytest tests/ -q --tb=short\n";
verify_prompt += "2. Confirm corpus cases under tests/corpus/ (list dirs + case.json expects for serve_blocked/weights where relevant)\n";
if skip_live == true {
    verify_prompt += "3. skip_live=true — do not hit Hub; mark live_ok=true only if offline corpus covers Nemotron+others\n";
} else {
    verify_prompt += "3. Live recommend matrix (workflow_max) for:\n";
    verify_prompt += "   - nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 (must: weights~20, blocked=false, marlin, DSpark draft, modelopt_mixed)\n";
    verify_prompt += "   - unsloth/Qwen3.6-35B-A3B-NVFP4 (no flashinfer_b12x, compressed-tensors)\n";
    verify_prompt += "   - nvidia/Qwen3.6-27B-NVFP4\n";
    verify_prompt += "   - deepseek-ai/DeepSeek-V4-Flash (Anemll overlay TP=2 if 2 nodes)\n";
    verify_prompt += "   Report weights_gib, serve_blocked, quant, moe, image, $ in extra_flags\n";
}
verify_prompt += "4. Set pytest_pass from real pytest exit code. List concrete failures.\n";
verify_prompt += "5. live_ok=false if any critical model is wrongly blocked or has $VARS or missing DSpark on Lightning.\n";

let ver = agent(verify_prompt, #{
    label: "verify-suite",
    capability_mode: "execute",
    agent_type: "general-purpose",
    output_schema: verify_schema,
});

let pytest_pass = false;
let live_ok = false;
let failures = [];
if ver != () && ver.success == true && ver.output != () {
    if ver.output.pytest_pass == true { pytest_pass = true; }
    if ver.output.live_ok == true { live_ok = true; }
    if ver.output.failures != () {
        failures = ver.output.failures;
    }
    log("Verify: pytest_pass=" + pytest_pass.to_string() + " live_ok=" + live_ok.to_string());
} else {
    failures.push("verify agent failed");
    log("Verify agent failed");
}

// ═══════════════════════════════════════════════════════════════════════════
phase("Harden");
// ═══════════════════════════════════════════════════════════════════════════

let round = 0;
while round < max_rounds {
    // Explicit bool compares — dry-run canned outputs can leave () on schema fields.
    let both_green = false;
    if pytest_pass == true {
        if live_ok == true {
            both_green = true;
        }
    }
    if both_green == true {
        log("All green — skipping further harden rounds");
        break;
    }
    round += 1;
    log("Harden round " + round.to_string() + "/" + max_rounds.to_string());

    let fail_blob = json_encode(failures);
    let fix_prompt = "";
    fix_prompt += "Fix remaining auto-config failures in " + repo + "\n";
    fix_prompt += "Round " + round.to_string() + " of " + max_rounds.to_string() + "\n";
    fix_prompt += "pytest_pass=" + pytest_pass.to_string() + " live_ok=" + live_ok.to_string() + "\n";
    fix_prompt += "Failures JSON: " + fail_blob + "\n\n";
    fix_prompt += "Rules: surgical fixes only; run pytest after; preserve Nemotron 20GiB/DSpark/marlin fixes.\n";
    fix_prompt += "Commands: " + se_cd + "python -m pytest tests/ -q --tb=short\n";
    if skip_live != true {
        fix_prompt += "Re-check live Lightning recommend after fixes.\n";
    }

    let fix = agent(fix_prompt, #{
        label: "harden-" + round.to_string(),
        capability_mode: "all",
        agent_type: "general-purpose",
        output_schema: fix_schema,
    });

    // Re-verify
    let re_prompt = "";
    re_prompt += "Re-verify after harden round " + round.to_string() + " in " + repo + "\n";
    re_prompt += "Run: " + se_cd + "python -m pytest tests/ -q --tb=line\n";
    if skip_live == true {
        re_prompt += "skip_live: set live_ok from corpus Nemotron case only.\n";
    } else {
        re_prompt += "Live: recommend Lightning 30B workflow_max — blocked=false, weights 15-30, marlin, DSpark model present, no $ in extras.\n";
    }
    re_prompt += "Return accurate pytest_pass/live_ok/failures.\n";

    let re = agent(re_prompt, #{
        label: "reverify-" + round.to_string(),
        capability_mode: "execute",
        agent_type: "general-purpose",
        output_schema: verify_schema,
    });

    if re != () && re.success == true && re.output != () {
        pytest_pass = false;
        live_ok = false;
        if re.output.pytest_pass == true { pytest_pass = true; }
        if re.output.live_ok == true { live_ok = true; }
        failures = [];
        if re.output.failures != () {
            failures = re.output.failures;
        }
        log("Reverify: pytest=" + pytest_pass.to_string() + " live=" + live_ok.to_string());
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
report += "# Auto-config bulletproof workflow report\n\n";
report += "Repo: " + repo + "\n";
report += "pytest_pass: " + pytest_pass.to_string() + "\n";
report += "live_ok: " + live_ok.to_string() + "\n";
report += "harden_rounds: " + round.to_string() + "\n\n";
report += "## Stream summaries\n";
for line in stream_summaries {
    report += "- " + line + "\n";
}
report += "\n## Failures remaining\n";
if failures.len() == 0 {
    report += "(none)\n";
} else {
    for f in failures {
        report += "- " + f + "\n";
    }
}
report += "\n## Inventory P0 titles\n";
if inv.output.p0 != () {
    for g in inv.output.p0 {
        report += "- [" + g.id + "] " + g.title + "\n";
    }
}
report += "\n## Next steps if not green\n";
report += "- Re-run: /workflow autoconfig-bulletproof with higher agent_budget\n";
report += "- Or set args.rounds higher and resume with raised budget\n";
report += "- Manual: " + se_cd + "python -m pytest tests/ -q\n";

let report_path = write_scratch_file("autoconfig-bulletproof-report.md", report);
log("Report written: " + report_path);

let final_green = false;
if pytest_pass == true {
    if live_ok == true {
        final_green = true;
    }
}
if final_green == true {
    complete(#{
        status: "green",
        pytest_pass: true,
        live_ok: true,
        report: report_path,
        summary: "Auto-config bulletproof tracks green",
    });
}

complete(#{
    status: "partial",
    pytest_pass: pytest_pass,
    live_ok: live_ok,
    failures: failures,
    report: report_path,
    summary: "Workflow finished with remaining gaps — see report",
});
