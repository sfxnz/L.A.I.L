let meta = #{
    name: "autoconfig-perfect",
    description: "Builder/critic loop: any-model Spark auto-config vs live NVIDIA/Unsloth/vLLM/llama.cpp recipes",
    when_to_use: "When auto-config must beat vendor Spark recipes (vLLM + llama.cpp) for any local model.",
    phases: [
        #{ title: "Inventory", detail: "confirm remaining gaps vs live vendor recipes" },
        #{ title: "Implement", detail: "sequential streams; critic after each; loop until pass" },
        #{ title: "BlindAB", detail: "fresh critics compare generated configs to reference recipes" },
        #{ title: "Harden", detail: "fix residual critic gaps until green or rounds exhausted" },
        #{ title: "Report", detail: "write status report" },
    ],
};

// args.repo   — absolute path to the worktree (required)
// args.rounds — max critic retries per stream (default 3)
// args.skip_live — skip Hub-live recommend (default false)

let repo = if args == () { () } else { args.repo };
if repo == () {
    pause("verification", "Pass args.repo — absolute path to the feat/autoconfig-any-model-spark worktree.");
}

let max_rounds = 3;
if args != () && args.rounds != () {
    max_rounds = args.rounds;
    if max_rounds < 1 { max_rounds = 1; }
}
let skip_live = false;
if args != () && args.skip_live == true { skip_live = true; }

let se_cd = "cd " + repo + "/packages/serve-engine && export PATH=\"/home/sfxnz/.bun/bin:$PATH\" && source /home/sfxnz/projects/ai-lab/local-ai-lab/.venv/bin/activate && PYTHONPATH=. ";

let gaps_schema = #{
    "type": "object",
    "required": ["p0", "already_done"],
    "properties": #{
        "already_done": #{ "type": "array", "items": #{ "type": "string" } },
        "p0": #{
            "type": "array",
            "maxItems": 10,
            "items": #{
                "type": "object",
                "required": ["id", "title", "acceptance"],
                "properties": #{
                    "id": #{ "type": "string" },
                    "title": #{ "type": "string" },
                    "acceptance": #{ "type": "string" },
                },
            },
        },
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
    },
};

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

let verify_schema = #{
    "type": "object",
    "required": ["pytest_pass", "live_ok", "failures"],
    "properties": #{
        "pytest_pass": #{ "type": "boolean" },
        "live_ok": #{ "type": "boolean" },
        "failures": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 20 },
        "notes": #{ "type": "string" },
    },
};

// ═══════════════════════════════════════════════════════════════════════════
phase("Inventory");
// ═══════════════════════════════════════════════════════════════════════════

let inv_p = "";
inv_p += "Read-only inventory of L.A.I.L auto-config remaining gaps.\n";
inv_p += "Workspace: " + repo + "\n";
inv_p += "Use read_file/grep/list_dir. Do not edit.\n\n";
inv_p += "KNOWN 2026-08-15 QUALITY BAR (already researched — verify in code):\n";
inv_p += "1. Unsloth NVFP4 Spark (unsloth.ai/docs/basics/nvfp4, Aug 14 2026):\n";
inv_p += "   export CUTE_DSL_ARCH=sm_121a\n";
inv_p += "   vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --moe-backend flashinfer_b12x\n";
inv_p += "   Current engine STRIPS flashinfer_b12x for mixed compressed-tensors AND first-boot\n";
inv_p += "   allowed-set omits flashinfer_b12x. Lab image is vllm/vllm-openai:v0.27.1.\n";
inv_p += "2. NVIDIA dgx-spark-playbooks nvidia/vllm README (live):\n";
inv_p += "   Agent-ready nvidia/Qwen3.6-35B-A3B-NVFP4 uses --moe-backend marlin,\n";
inv_p += "   --tool-call-parser qwen3_xml, --attention-backend flashinfer, MTP 3.\n";
inv_p += "   Gemma 4 needs image vllm/vllm-openai:gemma4-cu130.\n";
inv_p += "   DiffusionGemma needs vllm/vllm-openai:gemma + special flags.\n";
inv_p += "3. family_doc_slugs only qwen3.x/nemotron/minimax/deepseek/nvfp4.\n";
inv_p += "   recipes.vllm.ai/{slug}.md is wrong; real pages are /Org/Model.\n";
inv_p += "4. llama.cpp recommend does not exist. Spark refs: -ngl 99/--no-mmap/--flash-attn on/--jinja.\n";
inv_p += "5. Corpus forbids flashinfer_b12x on Unsloth 35B — that expect is now WRONG vs Unsloth.\n\n";
inv_p += "Read autoconfig.py, serve.py flashinfer guard, data/serve_overlays.json, corpus case.json files.\n";
inv_p += "Emit p0 remaining work (do not invent already-done items). already_done = landed capabilities.\n";

let inv = agent(inv_p, #{
    label: "inventory",
    capability_mode: "read-only",
    agent_type: "explore",
    output_schema: gaps_schema,
});
if inv == () || inv.success != true || inv.output == () {
    pause("infra", "Inventory agent failed.");
}
log("Inventory complete");

// Script owns scope. Inventory informs prompts only.
let streams = [
    #{
        id: "spark-moe",
        title: "Image-aware Spark MoE backend (Unsloth b12x + NVIDIA marlin)",
        detail: "ROOT CAUSE: _flashinfer_b12x_unsafe_for_checkpoint is image-blind and _apply_first_boot_defaults allowed-set omits flashinfer_b12x, so Unsloth Spark recipes lose on v0.27.1. serve.py also drops b12x for unsloth/compressed-tensors. FIX: image-aware policy — on stock image >= 0.27 keep flashinfer_b12x when card/vendor/Unsloth Spark recipe sets it (still strip on older images). Add flashinfer_b12x to first-boot allowed when safe for this checkpoint x image. Align serve.py guard with the same function. Update Unsloth 35B corpus to EXPECT moe_backend=flashinfer_b12x and CUTE_DSL_ARCH=sm_121a; keep strip tests for images < 0.27. NVIDIA ModelOpt Qwen3.6-35B-A3B may keep marlin on >=0.27 (playbook). Do not break DSpark overlay. TDD then implement.",
        bar: "Blind A/B: Unsloth official Spark command must match ours on moe_backend + CUTE_DSL_ARCH. Ours must not crash-flag on v0.27.1.",
    },
    #{
        id: "recipe-urls",
        title: "Derived recipe URLs for any family + recipes.vllm.ai org/name",
        detail: "Expand family_doc_slugs for gemma/gemma-4, glm, kimi, llama, phi, granite, gpt-oss, mistral without breaking qwen3.8/3.6/3.5 specificity. discover_recipe_urls must add https://recipes.vllm.ai/{org}/{name} from the HF id (not {slug}.md). Keep Unsloth /docs/models/{slug} and NVIDIA playbooks. Allowlist already has recipes.vllm.ai. Tests for slugs + URL derivation. Never override family overlays.",
        bar: "google/gemma-4-31B-it and unsloth/gemma-4-31B-it-NVFP4 must derive Unsloth + recipes.vllm.ai URLs. Qwen slugs still specific.",
    },
    #{
        id: "playbook-images",
        title: "NVIDIA playbook image pins (Gemma 4 / DiffusionGemma)",
        detail: "When detected family is gemma4 (or id has gemma-4/gemma4), pin image vllm/vllm-openai:gemma4-cu130 unless overlay Anemll. DiffusionGemma / diffusiongemma -> vllm/vllm-openai:gemma + do not steal Anemll. Never downgrade a newer stock pin. Overlay still wins. Unit + corpus/fixture tests. Optional overlay entries in data/serve_overlays.json for parsers (gemma4).",
        bar: "recommend(google/gemma-4-31B-it) image contains gemma4-cu130; DSv4 still Anemll.",
    },
    #{
        id: "llamacpp",
        title: "llama.cpp Spark recommend path",
        detail: "Add recommend_llamacpp(model) (or recommend(..., backend='llamacpp')) in autoconfig.py. Detect GGUF from id/path/tags. Spark defaults from ggml DGX Spark benches: ngl=99, flash_attn=on, no_mmap=true, jinja=true, ubatch=2048, ctx sized from UMA like vLLM envelope. Return structured config + argv string. Wire GET /serve/recommend?backend=llamacpp (default vllm). Tests with a tiny GGUF-id fixture. Do not start llama.cpp. Do not invent Ollama.",
        bar: "GGUF model returns -ngl 99, --no-mmap, flash-attn on, no $. Safetensors vLLM path unchanged.",
    },
    #{
        id: "corpus-families",
        title: "Corpus + overlays for playbook families that fit 1-2 Sparks",
        detail: "Add offline corpus and/or overlays for: nvidia/Qwen3.6-35B-A3B-NVFP4 (playbook: marlin + qwen3_xml harvest if card has it), google/gemma-4-31B-it or nvidia NVFP4 sibling (image + gemma4 parsers), openai/gpt-oss-20b (mxfp4, no crash flags). Capture via tests/corpus/_capture.py if Hub reachable else fixture-based cases. Update test_corpus expects. Do not hardcode recipes in Python — overlays JSON or card/vendor parse.",
        bar: "New corpus cases green; serve_blocked false for models that fit 1 Spark; no $ / --model in argv.",
    },
];

// ═══════════════════════════════════════════════════════════════════════════
phase("Implement");
// ═══════════════════════════════════════════════════════════════════════════

let stream_summaries = [];
let critic_log = [];

fn builder_prompt(repo, s, se_cd, gap, prev_gap) {
    let p = "";
    p += "You implement ONE auto-config stream in the L.A.I.L monorepo.\n";
    p += "Workspace: " + repo + "\n";
    p += "Stream: " + s.id + " — " + s.title + "\n";
    p += "Requirements:\n" + s.detail + "\n";
    p += "Quality bar: " + s.bar + "\n";
    if prev_gap != "" {
        p += "\nPREVIOUS CRITIC biggest_gap (must fix this, nothing else):\n" + prev_gap + "\n";
    }
    p += "\nRules (AGENTS.md):\n";
    p += "- Surgical. TDD: failing test first. No drive-by refactors.\n";
    p += "- Serve flags from placement engine / overlays JSON — do not hardcode model recipes in Python.\n";
    p += "- Do NOT revert Nemotron 20GiB / DSpark / Anemll / $VAR scrub / serve_blocked gates.\n";
    p += "- Do NOT commit, push, or touch data/*.sqlite.\n";
    p += "- Read current files first (prior streams may have edited them).\n";
    p += "- After edits run: " + se_cd + "python -m pytest tests/test_autoconfig.py tests/test_corpus.py -q --tb=short\n";
    p += "  Fix failures you introduced.\n";
    p += "Return done=true only if acceptance met and tests you ran passed.\n";
    if gap != () {
        p += "Inventory JSON (guidance): " + json_encode(gap) + "\n";
    }
    return p;
}

fn critic_prompt(repo, s, se_cd, skip_live) {
    let p = "";
    p += "You are a HARSCH independent critic. Fresh context. Fail closed.\n";
    p += "Workspace: " + repo + "\n";
    p += "Stream: " + s.id + " — " + s.title + "\n";
    p += "Quality bar: " + s.bar + "\n\n";
    p += "You MUST inspect REAL output with tools:\n";
    p += "1. Read the code the builder changed (autoconfig.py / serve.py / overlays / corpus).\n";
    p += "2. Run: " + se_cd + "python -m pytest tests/test_autoconfig.py tests/test_corpus.py -q --tb=line\n";
    if skip_live != true {
        p += "3. If Hub reachable, run a tiny PYTHONPATH=. python snippet calling recommend() for the stream's models.\n";
    } else {
        p += "3. skip_live — use offline corpus / fixtures only.\n";
    }
    p += "4. Blind A/B: compare generated flags to the vendor reference in the quality bar.\n";
    p += "   winner=ours only if we match or beat the reference on the judged flags.\n";
    p += "   winner=reference if the vendor recipe is better (missing flag, wrong image, stripped Spark moe).\n";
    p += "   passed=true ONLY if winner is ours or tie AND pytest is green AND no $ in extras.\n";
    p += "5. Identify the SINGLE biggest remaining gap. Evidence must quote real command/config.\n";
    p += "Do not implement. Do not give benefit of the doubt.\n";
    return p;
}

let si = 0;
while si < streams.len() {
    let s = streams[si];
    log("Stream " + s.id);
    let prev_gap = "";
    let passed = false;
    let rnum = 0;
    while rnum < max_rounds {
        rnum += 1;
        log(s.id + " builder round " + rnum.to_string());
        let bp = builder_prompt(repo, s, se_cd, inv.output, prev_gap);
        let br = agent(bp, #{
            label: "build:" + s.id + "-" + rnum.to_string(),
            capability_mode: "all",
            agent_type: "general-purpose",
            output_schema: impl_schema,
        });
        let bsum = "builder failed";
        if br != () && br.success == true && br.output != () && br.output.summary != () {
            bsum = br.output.summary;
        }
        log(s.id + " builder: " + bsum);

        let cp = critic_prompt(repo, s, se_cd, skip_live);
        let cr = agent(cp, #{
            label: "critic:" + s.id + "-" + rnum.to_string(),
            capability_mode: "execute",
            agent_type: "general-purpose",
            output_schema: critic_schema,
        });
        if cr != () && cr.success == true && cr.output != () {
            let win = "";
            if cr.output.winner != () { win = cr.output.winner; }
            let gap = "";
            if cr.output.biggest_gap != () { gap = cr.output.biggest_gap; }
            critic_log.push(s.id + " r" + rnum.to_string() + " winner=" + win + " gap=" + gap);
            log(s.id + " critic winner=" + win);
            if cr.output.passed == true {
                passed = true;
                break;
            }
            prev_gap = gap;
        } else {
            critic_log.push(s.id + " r" + rnum.to_string() + " critic-failed");
            prev_gap = "critic returned no usable verdict — re-run tests and fix obvious gaps";
        }
    }
    let done_s = "false";
    if passed == true { done_s = "true"; }
    stream_summaries.push(s.id + ": critic_pass=" + done_s);
    si += 1;
}

let integ = "";
integ += "Integration pass after all autoconfig streams in " + repo + "\n";
integ += "Stream results:\n";
for line in stream_summaries {
    integ += "- " + line + "\n";
}
integ += "Finish any critic_pass=false streams. Run: " + se_cd + "python -m pytest tests/ -q --tb=line\n";
integ += "Fix until green. Preserve DSpark/Nemotron/Anemll. Do not commit.\n";
let merge = agent(integ, #{
    label: "integration",
    capability_mode: "all",
    agent_type: "general-purpose",
    output_schema: impl_schema,
});
if merge != () && merge.success == true && merge.output != () && merge.output.summary != () {
    log("Integration: " + merge.output.summary);
}

// ═══════════════════════════════════════════════════════════════════════════
phase("BlindAB");
// ═══════════════════════════════════════════════════════════════════════════

let ab_models = [
    #{
        id: "unsloth-35b",
        model: "unsloth/Qwen3.6-35B-A3B-NVFP4",
        ref: "Unsloth Spark: CUTE_DSL_ARCH=sm_121a and --moe-backend flashinfer_b12x. compressed-tensors. parsers qwen3/qwen3_coder. no $.",
    },
    #{
        id: "nvidia-27b",
        model: "nvidia/Qwen3.6-27B-NVFP4",
        ref: "recipes/card: --quantization modelopt or modelopt_mixed from config.json, kv fp8, qwen3 parsers, no flashinfer_b12x (dense).",
    },
    #{
        id: "nemotron-lightning",
        model: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        ref: "weights ~20GiB, serve_blocked=false, modelopt_mixed, DSpark draft resolved or stripped, marlin allowed, no $.",
    },
    #{
        id: "dsv4",
        model: "deepseek-ai/DeepSeek-V4-Flash",
        ref: "Anemll image, moe flashinfer_b12x, kv nvfp4_ds_mla, TP=2 if 2 nodes else serve_blocked.",
    },
    #{
        id: "unsloth-38",
        model: "unsloth/Qwen3.8-27B-NVFP4",
        ref: "VL stays on (no language-model-only), fp8 KV, qwen3 parsers, hardware-sized util/max-len.",
    },
];

let ab_jobs = [];
let abi = 0;
while abi < ab_models.len() {
    let m = ab_models[abi];
    let ap = "";
    ap += "Harsh blind A/B critic. Workspace: " + repo + "\n";
    ap += "Model: " + m.model + "\n";
    ap += "REFERENCE (do not peek at our code first): " + m.ref + "\n\n";
    ap += "1. Run offline recommend if corpus exists under tests/corpus/; else if skip_live=";
    if skip_live == true { ap += "true"; } else { ap += "false"; }
    ap += " and Hub works, call recommend().\n";
    ap += "Use: " + se_cd + "python - <<'PY'\n";
    ap += "from app.services import autoconfig as ac\n";
    ap += "# prefer corpus fixtures via monkeypatch pattern in tests/test_corpus.py if present\n";
    ap += "print('inspect corpus or recommend')\n";
    ap += "PY\n";
    ap += "2. Quote the generated moe/quant/image/parsers/extra_flags/util/serve_blocked.\n";
    ap += "3. winner=ours only if we match or beat the reference on judged flags.\n";
    ap += "   passed=true only if winner is ours or tie and no $ / --model in extras.\n";
    ap += "4. Single biggest_gap if we lose.\n";
    ab_jobs.push(#{
        prompt: ap,
        label: "ab:" + m.id,
        capability_mode: "execute",
        agent_type: "general-purpose",
        output_schema: critic_schema,
    });
    abi += 1;
}
let ab_results = parallel(ab_jobs);
let ab_fail = [];
let ab_i = 0;
while ab_i < ab_results.len() {
    let r = ab_results[ab_i];
    let mid = ab_models[ab_i].id;
    if r != () && r.success == true && r.output != () {
        let win = "";
        if r.output.winner != () { win = r.output.winner; }
        log("A/B " + mid + " winner=" + win);
        if r.output.passed != true {
            let g = "";
            if r.output.biggest_gap != () { g = r.output.biggest_gap; }
            ab_fail.push(mid + ": " + g);
        }
    } else {
        ab_fail.push(mid + ": critic failed");
    }
    ab_i += 1;
}

// ═══════════════════════════════════════════════════════════════════════════
phase("Harden");
// ═══════════════════════════════════════════════════════════════════════════

let pytest_pass = false;
let live_ok = false;
let failures = [];
for f in ab_fail { failures.push(f); }

let ver_p = "";
ver_p += "Verify auto-config in " + repo + "\n";
ver_p += "1. " + se_cd + "python -m pytest tests/ -q --tb=short\n";
if skip_live == true {
    ver_p += "2. skip_live: live_ok=true only if corpus covers Nemotron+Unsloth 35B+DSv4+Qwen 27B\n";
} else {
    ver_p += "2. Live recommend Lightning 30B + Unsloth 35B: blocked=false, no $, Unsloth keeps flashinfer_b12x on v0.27.1\n";
}
ver_p += "3. pytest_pass from real exit code. List failures.\n";
let ver = agent(ver_p, #{
    label: "verify",
    capability_mode: "execute",
    agent_type: "general-purpose",
    output_schema: verify_schema,
});
if ver != () && ver.success == true && ver.output != () {
    if ver.output.pytest_pass == true { pytest_pass = true; }
    if ver.output.live_ok == true { live_ok = true; }
    if ver.output.failures != () { failures = ver.output.failures; }
}

let hround = 0;
while hround < max_rounds {
    let green = false;
    if pytest_pass == true {
        if live_ok == true {
            if ab_fail.len() == 0 { green = true; }
        }
    }
    if green == true {
        log("All green — stop harden");
        break;
    }
    hround += 1;
    log("Harden " + hround.to_string());
    let fp = "";
    fp += "Fix remaining auto-config gaps in " + repo + "\n";
    fp += "Failures: " + json_encode(failures) + "\n";
    fp += "A/B losses: " + json_encode(ab_fail) + "\n";
    fp += "Surgical. Run pytest. Preserve DSpark/Nemotron. Do not commit.\n";
    fp += se_cd + "python -m pytest tests/ -q --tb=short\n";
    let fx = agent(fp, #{
        label: "harden-" + hround.to_string(),
        capability_mode: "all",
        agent_type: "general-purpose",
        output_schema: impl_schema,
    });
    if fx != () && fx.success == true && fx.output != () && fx.output.summary != () {
        log("Harden: " + fx.output.summary);
    }
    let rp = "";
    rp += "Re-verify after harden in " + repo + "\n";
    rp += se_cd + "python -m pytest tests/ -q --tb=line\n";
    if skip_live == true {
        rp += "skip_live: live_ok from corpus Nemotron+Unsloth35B.\n";
    } else {
        rp += "Live Lightning + Unsloth 35B: blocked=false; Unsloth moe=flashinfer_b12x; no $.\n";
    }
    let re = agent(rp, #{
        label: "reverify-" + hround.to_string(),
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
        if re.output.failures != () { failures = re.output.failures; }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
phase("Report");
// ═══════════════════════════════════════════════════════════════════════════

let report = "";
report += "# Auto-config perfect workflow\n\n";
report += "Repo: " + repo + "\n";
report += "pytest_pass: " + pytest_pass.to_string() + "\n";
report += "live_ok: " + live_ok.to_string() + "\n";
report += "harden_rounds: " + hround.to_string() + "\n\n";
report += "## Streams\n";
for line in stream_summaries { report += "- " + line + "\n"; }
report += "\n## Critic log\n";
for line in critic_log { report += "- " + line + "\n"; }
report += "\n## A/B losses\n";
if ab_fail.len() == 0 { report += "(none)\n"; }
for f in ab_fail { report += "- " + f + "\n"; }
report += "\n## Failures\n";
if failures.len() == 0 { report += "(none)\n"; }
for f in failures { report += "- " + f + "\n"; }
let report_path = write_scratch_file("autoconfig-perfect-report.md", report);
log("Report: " + report_path);

let final_green = false;
if pytest_pass == true {
    if live_ok == true {
        if ab_fail.len() == 0 { final_green = true; }
    }
}
if final_green == true {
    complete(#{
        status: "green",
        pytest_pass: true,
        live_ok: true,
        report: report_path,
        summary: "Auto-config beat or tied vendor Spark recipes; tests green",
    });
}
complete(#{
    status: "partial",
    pytest_pass: pytest_pass,
    live_ok: live_ok,
    failures: failures,
    ab_fail: ab_fail,
    report: report_path,
    summary: "Workflow finished with remaining gaps",
});
