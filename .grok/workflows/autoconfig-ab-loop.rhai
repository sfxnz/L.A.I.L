let meta = #{
    name: "autoconfig-ab-loop",
    description: "Keep A/B looping Spark auto-config against live NVIDIA/Unsloth/vLLM/GitHub recipes until every panel wins",
    when_to_use: "After a partial autoconfig-perfect run; do not stop while any vendor recipe still wins.",
    phases: [
        #{ title: "AB", detail: "blind A/B vs official recipes" },
        #{ title: "Fix", detail: "builder on the single biggest remaining gap" },
        #{ title: "Report", detail: "status; green only if every A/B passed" },
    ],
};

// args.repo    — worktree (required)
// args.rounds  — max fix+A/B cycles (default 8)

let repo = if args == () { () } else { args.repo };
if repo == () {
    pause("verification", "Pass args.repo — absolute path to feat/autoconfig-any-model-spark.");
}
let max_rounds = 8;
if args != () && args.rounds != () {
    max_rounds = args.rounds;
    if max_rounds < 1 { max_rounds = 1; }
}

let se_cd = "cd " + repo + "/packages/serve-engine && export PATH=\"/home/sfxnz/.bun/bin:$PATH\" && source /home/sfxnz/projects/ai-lab/local-ai-lab/.venv/bin/activate && PYTHONPATH=. ";

let critic_schema = #{
    "type": "object",
    "required": ["passed", "winner", "biggest_gap", "evidence"],
    "properties": #{
        "passed": #{ "type": "boolean" },
        "winner": #{ "type": "string" },
        "biggest_gap": #{ "type": "string" },
        "evidence": #{ "type": "string" },
    },
};
let impl_schema = #{
    "type": "object",
    "required": ["stream", "done", "summary"],
    "properties": #{
        "stream": #{ "type": "string" },
        "done": #{ "type": "boolean" },
        "summary": #{ "type": "string" },
    },
};
let verify_schema = #{
    "type": "object",
    "required": ["pytest_pass", "failures"],
    "properties": #{
        "pytest_pass": #{ "type": "boolean" },
        "failures": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 20 },
    },
};

let models = [
    #{
        id: "unsloth-35b",
        model: "unsloth/Qwen3.6-35B-A3B-NVFP4",
        ref: "Unsloth Spark (unsloth.ai/docs/basics/nvfp4): CUTE_DSL_ARCH=sm_121a AND --moe-backend flashinfer_b12x. compressed-tensors. NOT qwen3_xml (that is NVIDIA Agent-Ready). no $.",
    },
    #{
        id: "nvidia-35b-playbook",
        model: "nvidia/Qwen3.6-35B-A3B-NVFP4",
        ref: "NVIDIA dgx-spark-playbooks Agent Ready: --moe-backend marlin, --tool-call-parser qwen3_xml, --attention-backend flashinfer, MTP 3 with moe_backend triton. no $.",
    },
    #{
        id: "nvidia-27b",
        model: "nvidia/Qwen3.6-27B-NVFP4",
        ref: "Dense ModelOpt: quant modelopt_mixed from config.json, kv fp8, qwen3 parsers, no flashinfer_b12x.",
    },
    #{
        id: "gemma4-31b",
        model: "google/gemma-4-31B-it",
        ref: "NVIDIA playbook + recipes.vllm.ai/Google/gemma-4-31B-it: image gemma4-cu130, gemma4 parsers, NO flashinfer_b12x (dense, not 26B-A4B MoE). Prefer official 31B flags (chat-template / async / mm-cap) over Unsloth 26B-A4B Spark line.",
    },
    #{
        id: "unsloth-38",
        model: "unsloth/Qwen3.8-27B-NVFP4",
        ref: "VL stays on, fp8 KV, qwen3 parsers, no language-model-only, no YaRN 1M.",
    },
    #{
        id: "nemotron",
        model: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        ref: "weights ~20GiB, serve_blocked=false, modelopt_mixed, DSpark draft resolved or stripped, no $.",
    },
    #{
        id: "dsv4",
        model: "deepseek-ai/DeepSeek-V4-Flash",
        ref: "Anemll image, moe flashinfer_b12x, kv nvfp4_ds_mla, TP=2 if 2 nodes else blocked.",
    },
    #{
        id: "llamacpp-gguf",
        model: "unsloth/Qwen3.6-27B-GGUF",
        ref: "recommend_llamacpp: -ngl 99 --flash-attn on --no-mmap --jinja, one Spark quant not sum of all GGUF siblings, no $.",
    },
];

// ═══════════════════════════════════════════════════════════════════════════
phase("AB");
// ═══════════════════════════════════════════════════════════════════════════

let ab_jobs = [];
let abi0 = 0;
while abi0 < models.len() {
    let m0 = models[abi0];
    let p0 = "";
    p0 += "Harsh independent critic. Fresh context. Fail closed. Do not implement.\n";
    p0 += "Workspace: " + repo + "\n";
    p0 += "Model: " + m0.model + "\n";
    p0 += "REFERENCE (inspect real vendor text with tools): " + m0.ref + "\n\n";
    p0 += "Use tools. Call recommend() or recommend_llamacpp() and quote real flags.\n";
    p0 += "winner=ours only if we match or beat the reference. passed=true only if ours/tie and no $.\n";
    p0 += "Single biggest_gap if we lose.\n";
    p0 += "Commands: " + se_cd + "\n";
    ab_jobs.push(#{
        prompt: p0,
        label: "ab:" + m0.id,
        capability_mode: "execute",
        agent_type: "general-purpose",
        output_schema: critic_schema,
    });
    abi0 += 1;
}
let ab_results = parallel(ab_jobs);
let losses = [];
let log_lines = [];
let ai = 0;
while ai < ab_results.len() {
    let r = ab_results[ai];
    let mid = models[ai].id;
    if r != () && r.success == true && r.output != () {
        let win = "";
        if r.output.winner != () { win = r.output.winner; }
        log("A/B " + mid + " winner=" + win);
        let line = mid + " winner=" + win;
        if r.output.passed != true {
            let g = "";
            if r.output.biggest_gap != () { g = r.output.biggest_gap; }
            losses.push(mid + ": " + g);
            line += " GAP " + g;
        }
        log_lines.push(line);
    } else {
        losses.push(mid + ": critic failed");
        log_lines.push(mid + " critic-failed");
    }
    ai += 1;
}

// ═══════════════════════════════════════════════════════════════════════════
phase("Fix");
// ═══════════════════════════════════════════════════════════════════════════

let round = 0;
while round < max_rounds {
    if losses.len() == 0 {
        log("All A/B passed — no fix rounds");
        break;
    }
    round += 1;
    log("Fix round " + round.to_string() + " losses=" + losses.len().to_string());
    let fp = "";
    fp += "You fix the SINGLE biggest remaining auto-config A/B loss in " + repo + "\n";
    fp += "Round " + round.to_string() + " of " + max_rounds.to_string() + "\n";
    fp += "Losses (fix the first one first; do not boil the ocean):\n";
    for g in losses { fp += "- " + g + "\n"; }
    fp += "\nRules: surgical; TDD; no drive-by; preserve Unsloth b12x on >=0.27 MoE, DSpark/Anemll, $VAR scrub.\n";
    fp += "Do not commit. Run: " + se_cd + "python -m pytest tests/test_autoconfig.py tests/test_corpus.py -q --tb=short\n";
    let fx = agent(fp, #{
        label: "fix-" + round.to_string(),
        capability_mode: "all",
        agent_type: "general-purpose",
        output_schema: impl_schema,
    });
    if fx != () && fx.success == true && fx.output != () && fx.output.summary != () {
        log("Fix: " + fx.output.summary);
    }
    let vp = "";
    vp += "Run " + se_cd + "python -m pytest tests/ -q --tb=line\n";
    vp += "pytest_pass from real exit code.\n";
    let ver = agent(vp, #{
        label: "pytest-" + round.to_string(),
        capability_mode: "execute",
        agent_type: "general-purpose",
        output_schema: verify_schema,
    });
    if ver != () && ver.success == true && ver.output != () && ver.output.pytest_pass != true {
        log("pytest red — next fix must restore green");
    }
    let re_jobs = [];
    let rj = 0;
    while rj < models.len() {
        let m1 = models[rj];
        let p1 = "";
        p1 += "Harsh independent critic. Fresh context. Fail closed. Do not implement.\n";
        p1 += "Workspace: " + repo + "\n";
        p1 += "Model: " + m1.model + "\n";
        p1 += "REFERENCE: " + m1.ref + "\n";
        p1 += "Use tools. Call recommend() or recommend_llamacpp() and quote real flags.\n";
        p1 += "winner=ours only if we match or beat the reference. passed=true only if ours/tie and no $.\n";
        p1 += "Commands: " + se_cd + "\n";
        re_jobs.push(#{
            prompt: p1,
            label: "ab" + round.to_string() + ":" + m1.id,
            capability_mode: "execute",
            agent_type: "general-purpose",
            output_schema: critic_schema,
        });
        rj += 1;
    }
    ab_results = parallel(re_jobs);
    losses = [];
    ai = 0;
    while ai < ab_results.len() {
        let r = ab_results[ai];
        let mid = models[ai].id;
        if r != () && r.success == true && r.output != () {
            let win = "";
            if r.output.winner != () { win = r.output.winner; }
            log("A/B " + mid + " winner=" + win);
            if r.output.passed != true {
                let g = "";
                if r.output.biggest_gap != () { g = r.output.biggest_gap; }
                losses.push(mid + ": " + g);
            }
        } else {
            losses.push(mid + ": critic failed");
        }
        ai += 1;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
phase("Report");
// ═══════════════════════════════════════════════════════════════════════════

let report = "";
report += "# Auto-config A/B loop\n\n";
report += "Repo: " + repo + "\n";
report += "fix_rounds: " + round.to_string() + "\n";
report += "losses_remaining: " + losses.len().to_string() + "\n\n";
report += "## Last A/B\n";
for line in log_lines { report += "- " + line + "\n"; }
report += "\n## Remaining losses\n";
if losses.len() == 0 { report += "(none)\n"; }
for g in losses { report += "- " + g + "\n"; }
let report_path = write_scratch_file("autoconfig-ab-loop-report.md", report);
log("Report: " + report_path);

if losses.len() == 0 {
    complete(#{
        status: "green",
        report: report_path,
        summary: "Every vendor A/B passed",
    });
}
complete(#{
    status: "partial",
    losses: losses,
    report: report_path,
    summary: "A/B loop finished with remaining vendor losses — raise rounds/budget and re-run",
});
