"""Tests for HF model-card auto-config (shipped recommend path).

Live tests hit huggingface.co when network is available.
Parser/safety tests use fixtures but call the real shipped functions.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from app.services import autoconfig as ac

FIX = Path(__file__).resolve().parent / "fixtures"
UNSLOTH = "unsloth/Qwen3.6-35B-A3B-NVFP4"
NVIDIA = "nvidia/Qwen3.6-27B-NVFP4"
QWEN_FP8 = "Qwen/Qwen3.6-27B-FP8"

# Skip live network tests when offline (CI/sandbox without hub access)
def _hub_reachable() -> bool:
    body, err = ac._http_get(
        f"https://huggingface.co/{NVIDIA}/raw/main/README.md",
        timeout=15.0,
    )
    return bool(body and len(body) > 100 and not err)


requires_hub = pytest.mark.skipif(
    not _hub_reachable(),
    reason="huggingface.co not reachable",
)


# ─── Parser / scoring (fixtures → real extract_serve_candidates) ─────────────


def test_extract_serve_candidates_from_fixture_card():
    readme = (FIX / "card_mixed_moe_nvfp4.md").read_text()
    cands = ac.extract_serve_candidates(readme)
    assert len(cands) >= 3
    # At least one recipe mentions flashinfer_b12x and one is bare serve
    raws = " ".join(c.raw for c in cands)
    assert "flashinfer_b12x" in raws
    assert "vllm serve" in raws.lower()
    # Each candidate has a score from real scorer
    assert all(isinstance(c.score, (int, float)) for c in cands)


def test_mixed_checkpoint_penalizes_flashinfer_in_scoring():
    readme = (FIX / "card_mixed_moe_nvfp4.md").read_text()
    cfg = json.loads((FIX / "config_mixed_compressed_tensors.json").read_text())
    detected = ac.analyze_config(cfg, "example/Mixed-MoE-NVFP4")
    assert detected["is_mixed_nvfp4_fp8"] is True
    assert detected["quant_flag"] == "compressed-tensors"

    cands = ac.extract_serve_candidates(readme, detected=detected)
    # Recipes that mentioned flashinfer on the card (raw), even if moe_backend was cleared
    flash_raw = [
        c
        for c in cands
        if "flashinfer_b12x" in (c.raw or "")
        or any("flashinfer" in (r or "").lower() for r in (c.reasons or []))
    ]
    non_flash = [
        c
        for c in cands
        if "flashinfer_b12x" not in (c.raw or "")
        and (c.config.get("moe_backend") or "") != "flashinfer_b12x"
    ]
    assert flash_raw, "fixture must include flashinfer recipe text"
    assert non_flash, "fixture must include non-flashinfer recipe"
    # Best overall must not keep flashinfer_b12x in config (cleared for safety)
    best = cands[0]
    assert (best.config.get("moe_backend") or "") != "flashinfer_b12x"
    # Flashinfer card recipes must carry a penalty / clearance reason
    assert any(
        any(
            "PENALTY" in (x or "") or "FP8 MoE" in (x or "") or "cleared moe_backend" in (x or "")
            for x in (c.reasons or [])
        )
        for c in flash_raw
    )


def test_checkpoint_safety_strips_flashinfer_b12x():
    cfg_json = json.loads((FIX / "config_mixed_compressed_tensors.json").read_text())
    detected = ac.analyze_config(cfg_json, "example/Mixed-MoE-NVFP4")
    serve_cfg = {
        "model": "example/Mixed-MoE-NVFP4",
        "quantization": "compressed-tensors",
        "moe_backend": "flashinfer_b12x",
        "docker_env": ["CUTE_DSL_ARCH=sm_121a"],
        "kv_cache_dtype": "",
        "max_num_seqs": None,
    }
    warnings: list[str] = []
    rationale: list[str] = []
    ac._apply_checkpoint_safety(serve_cfg, detected, warnings, rationale)
    assert serve_cfg["moe_backend"] == ""
    assert any("flashinfer_b12x" in w for w in warnings)
    assert any("SAFETY" in r for r in rationale)
    # Still keeps Spark env
    assert any(e.startswith("CUTE_DSL_ARCH=") for e in serve_cfg["docker_env"])


def test_modelopt_card_recipe_sets_quantization():
    readme = (FIX / "card_modelopt_dense.md").read_text()
    cands = ac.extract_serve_candidates(readme)
    assert cands
    best = cands[0]
    assert best.config.get("quantization") == "modelopt"
    assert best.config.get("reasoning_parser") == "qwen3"



def test_analyze_config_modelopt_mixed_precision():
    """ModelOpt MIXED_PRECISION must not collapse to umbrella modelopt."""
    cfg = {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
            "quantized_layers": {
                "a": {"quant_algo": "FP8"},
                "b": {"quant_algo": "W4A16_NVFP4"},
            },
        },
    }
    d = ac.analyze_config(cfg, "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
    assert d["quant_flag"] == "modelopt_mixed"
    assert d["quant_algo"] == "MIXED_PRECISION"
    assert d["is_mixed_nvfp4_fp8"] is True
    assert d["suggested_kv_cache_dtype"] == "fp8"
    assert ac._flashinfer_b12x_unsafe_for_checkpoint(d) is True
    # Nemotron still keeps marlin despite mixed ModelOpt.
    assert ac._marlin_unsafe_for_checkpoint(d) is False


def test_analyze_config_modelopt_nvfp4_algo():
    cfg = {
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
        }
    }
    d = ac.analyze_config(cfg, "nvidia/Example-27B-NVFP4")
    assert d["quant_flag"] == "modelopt_fp4"
    assert d["has_nvfp4"] is True
    assert d["is_mixed_nvfp4_fp8"] is False



def test_merge_hf_quant_config_into_empty():
    cfg = {"architectures": ["FooForCausalLM"]}
    qc = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    out = ac._merge_hf_quant_config(cfg, qc)
    assert out["quantization_config"]["quant_method"] == "modelopt"
    d = ac.analyze_config(out, "nvidia/Example-NVFP4")
    assert d["quant_flag"] in ("modelopt_fp4", "modelopt")


def test_merge_hf_quant_does_not_clobber_existing_method():
    cfg = {"quantization_config": {"quant_method": "compressed-tensors"}}
    out = ac._merge_hf_quant_config(cfg, {"quant_method": "modelopt", "quant_algo": "FP8"})
    assert out["quantization_config"]["quant_method"] == "compressed-tensors"


def test_merge_hf_quant_normalizes_modelopt_sidecar():
    """Real ModelOpt sidecars nest algo under quantization + producer.name."""
    cfg = {"architectures": ["NemotronHForCausalLM"]}
    sidecar = {
        "producer": {"name": "modelopt", "version": "0.34.1"},
        "quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": None},
    }
    out = ac._merge_hf_quant_config(cfg, sidecar)
    assert out["quantization_config"]["quant_method"] == "modelopt"
    assert out["quantization_config"]["quant_algo"] == "NVFP4"
    d = ac.analyze_config(out, "nvidia/Example-NVFP4")
    assert d["quant_flag"] == "modelopt_fp4"


def test_load_local_fallback_merges_hf_quant_sidecar(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["FooForCausalLM"], "model_type": "foo"}),
        encoding="utf-8",
    )
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {"name": "modelopt"},
                "quantization": {"quant_algo": "NVFP4"},
            }
        ),
        encoding="utf-8",
    )
    local = ac.load_local_fallback(str(tmp_path))
    qc = (local.get("config") or {}).get("quantization_config") or {}
    assert qc.get("quant_method") == "modelopt"
    assert qc.get("quant_algo") == "NVFP4"


def test_quant_flags_compatible_modelopt_siblings():
    assert ac._quant_flags_compatible("modelopt_mixed", "modelopt_fp4")
    assert ac._quant_flags_compatible("modelopt", "modelopt")
    assert not ac._quant_flags_compatible("modelopt", "compressed-tensors")


def test_card_prose_prefers_modelopt_fp4_over_umbrella():
    hints = ac._card_prose_hints("run with --quantization modelopt_fp4 on Spark")
    assert hints["quantization"] == "modelopt_fp4"


def test_analyze_config_detects_mixed_formats():
    cfg = json.loads((FIX / "config_mixed_compressed_tensors.json").read_text())
    d = ac.analyze_config(cfg, "org/Model-A3B-NVFP4")
    assert d["is_moe"] is True
    assert d["has_nvfp4"] is True
    assert d["has_fp8"] is True
    assert d["is_mixed_nvfp4_fp8"] is True
    assert d["quant_flag"] == "compressed-tensors"


@pytest.mark.parametrize(
    "model_id,arch,want",
    [
        ("MiniMaxAI/MiniMax-M2-NVFP4", ["MiniMaxM2ForCausalLM"], "minimax_m2"),
        ("MiniMaxAI/MiniMax-M3", ["MiniMaxM3ForCausalLM"], "minimax_m3"),
        ("mistralai/Magistral-Small-2507", ["MistralForCausalLM"], "mistral"),
        ("deepseek-ai/DeepSeek-R1", ["DeepseekV3ForCausalLM"], "deepseek_r1"),
        ("deepseek-ai/DeepSeek-V3.2", ["DeepseekV3ForCausalLM"], "deepseek_v3"),
        ("deepseek-ai/DeepSeek-V4-Flash", ["DeepseekV4ForCausalLM"], "deepseek_v4"),
        ("THUDM/glm-4-9b-chat", ["ChatGLMModel"], "glm"),
        ("moonshotai/Kimi-K2-Instruct", ["KimiK2ForCausalLM"], "kimi"),
        ("meta-llama/Llama-3.3-70B-Instruct", ["LlamaForCausalLM"], "llama"),
        ("meta-llama/Llama-4-Scout-17B-16E-Instruct", ["Llama4ForCausalLM"], "llama4"),
        ("google/gemma-2-9b-it", ["Gemma2ForCausalLM"], "gemma"),
        ("google/gemma-4-9b-it", ["Gemma4ForCausalLM"], "gemma4"),
        ("microsoft/Phi-4-mini-instruct", ["Phi3ForCausalLM"], "phi"),
        ("ibm-granite/granite-3.3-8b-instruct", ["GraniteForCausalLM"], "granite"),
    ],
)

def test_analyze_config_family_detection(model_id, arch, want):
    d = ac.analyze_config({"architectures": arch, "model_type": arch[0].lower()}, model_id)
    assert d["family"] == want


def test_qwen25_skips_qwen3_parsers():
    cfg = ac._empty_config("Qwen/Qwen2.5-7B-Instruct")
    rationale: list[str] = []
    ac._fill_from_config_detection(
        cfg,
        {"family": "qwen", "quant_flag": "", "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"},
        rationale,
    )
    assert cfg.get("reasoning_parser") in ("", None)
    assert cfg.get("tool_call_parser") in ("", None)
    assert any("Qwen2.5" in r or "Qwen2" in r for r in rationale)


def test_qwen3_still_gets_parsers():
    cfg = ac._empty_config("Qwen/Qwen3-8B")
    rationale: list[str] = []
    ac._fill_from_config_detection(
        cfg,
        {"family": "qwen", "quant_flag": "", "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
        rationale,
    )
    assert cfg["reasoning_parser"] == "qwen3"
    assert cfg["tool_call_parser"] == "qwen3_coder"


def test_strip_spark_unsafe_flags():
    cfg = {
        "extra_flags": "--enable-expert-parallel --data-parallel-size 8 --max-num-batched-tokens 8192",
        "moe_backend": "humming",
        "docker_env": ["VLLM_USE_DEEP_GEMM_MEGA_MOE=1", "CUTE_DSL_ARCH=sm_121a"],
    }
    warnings: list[str] = []
    rationale: list[str] = []
    ac._strip_spark_unsafe_flags(cfg, warnings, rationale)
    ex = cfg["extra_flags"]
    assert "--enable-expert-parallel" not in ex
    assert "--data-parallel-size" not in ex
    assert "--max-num-batched-tokens" in ex
    assert cfg["moe_backend"] == ""
    assert not any(e.startswith("VLLM_USE_DEEP_GEMM_MEGA_MOE=") for e in cfg["docker_env"])
    assert any(e.startswith("CUTE_DSL_ARCH=") for e in cfg["docker_env"])


def test_fill_from_config_minimax_and_mistral_parsers():
    rationale: list[str] = []
    m2 = ac._empty_config("MiniMaxAI/MiniMax-M2")
    ac._fill_from_config_detection(
        m2, {"family": "minimax_m2", "quant_flag": "", "architectures": [], "model_type": ""}, rationale
    )
    assert m2["reasoning_parser"] == "minimax_m2"
    assert m2["tool_call_parser"] == "minimax_m2"
    assert m2["enable_auto_tool_choice"] is True

    m3 = ac._empty_config("MiniMaxAI/MiniMax-M3")
    ac._fill_from_config_detection(
        m3, {"family": "minimax_m3", "quant_flag": "", "architectures": [], "model_type": ""}, rationale
    )
    assert m3["reasoning_parser"] == "minimax_m3"
    assert "--block-size 128" in (m3.get("extra_flags") or "")

    mis = ac._empty_config("mistralai/Magistral-Small")
    ac._fill_from_config_detection(
        mis,
        {"family": "mistral", "quant_flag": "", "architectures": ["MagistralForCausalLM"], "model_type": "magistral"},
        rationale,
    )
    assert mis["tool_call_parser"] == "mistral"
    assert mis["reasoning_parser"] == "mistral"
    assert mis["load_format"] == "mistral"
    assert "--tokenizer-mode mistral" in (mis.get("extra_flags") or "")

    llama = ac._empty_config("meta-llama/Llama-3.3-70B-Instruct")
    ac._fill_from_config_detection(
        llama, {"family": "llama", "quant_flag": "", "architectures": [], "model_type": ""}, rationale
    )
    assert llama["tool_call_parser"] == "llama3_json"

    llama4 = ac._empty_config("meta-llama/Llama-4-Scout")
    ac._fill_from_config_detection(
        llama4, {"family": "llama4", "quant_flag": "", "architectures": [], "model_type": ""}, rationale
    )
    assert llama4["tool_call_parser"] == "llama4_pythonic"

    gemma4 = ac._empty_config("google/gemma-4-9b")
    ac._fill_from_config_detection(
        gemma4, {"family": "gemma4", "quant_flag": "", "architectures": [], "model_type": ""}, rationale
    )
    assert gemma4["reasoning_parser"] == "gemma4"
    assert gemma4["tool_call_parser"] == "gemma4"

    phi = ac._empty_config("microsoft/Phi-4-mini-instruct")
    ac._fill_from_config_detection(
        phi,
        {"family": "phi", "quant_flag": "", "architectures": ["Phi3ForCausalLM"], "model_type": "phi3"},
        rationale,
    )
    assert phi["tool_call_parser"] == "phi4_mini_json"

    gran = ac._empty_config("ibm-granite/granite-4.0-h-small")
    ac._fill_from_config_detection(
        gran,
        {"family": "granite", "quant_flag": "", "architectures": ["GraniteMoeHybridForCausalLM"], "model_type": "granitemoehybrid"},
        rationale,
    )
    assert gran["tool_call_parser"] == "granite4"


# ─── Live hub recommend (real entry point) ───────────────────────────────────


@requires_hub
def test_live_recommend_unsloth_35b_mixed_moe():
    r = ac.recommend(UNSLOTH, mode="lab_safe", fetch_remote=True)
    assert r["from_website"] is True
    assert r.get("card_url", "").startswith("https://huggingface.co/")
    c = r["config"]
    assert c.get("quantization") == "compressed-tensors"
    # Must not force flashinfer_b12x on mixed FP8 MoE
    assert (c.get("moe_backend") or "") == ""
    assert c.get("reasoning_parser") == "qwen3"
    assert any(e.startswith("CUTE_DSL_ARCH=") for e in (c.get("docker_env") or []))
    # Live sources present
    kinds = {s.get("kind") for s in r.get("sources") or []}
    assert "huggingface" in kinds or "hf_card_recipe" in kinds
    assert len(r.get("card_recipes") or []) >= 1
    # Criterion 3: clear warning + rationale when card recommends flashinfer but we avoid it
    warn_blob = " ".join(r.get("warnings") or [])
    rat_blob = " ".join(r.get("rationale") or [])
    assert "flashinfer_b12x" in warn_blob, f"expected flashinfer warning, got {r.get('warnings')}"
    assert "flashinfer_b12x" in rat_blob or "SAFETY" in rat_blob
    # Penalized recipes must expose reasons (for UI)
    flash_recipes = [
        cr
        for cr in (r.get("card_recipes") or [])
        if "flashinfer_b12x" in (cr.get("raw") or "")
        or (cr.get("config") or {}).get("moe_backend") == "flashinfer_b12x"
    ]
    assert flash_recipes, "card should still list flashinfer recipe among candidates"
    assert any(
        any("PENALTY" in (x or "") or "FP8 MoE" in (x or "") for x in (cr.get("reasons") or []))
        for cr in flash_recipes
    )

@requires_hub
def test_live_recommend_nvidia_27b_modelopt():
    r = ac.recommend(NVIDIA, mode="lab_safe", fetch_remote=True)
    assert r["from_website"] is True
    c = r["config"]
    assert c.get("quantization") == "modelopt"
    assert c.get("reasoning_parser") == "qwen3"
    # Card includes max-model-len 262144
    assert c.get("max_model_len") in (262144, 65536) or c.get("max_model_len") is not None


@requires_hub
def test_live_recommend_qwen_fp8():
    r = ac.recommend(QWEN_FP8, mode="lab_safe", fetch_remote=True)
    assert r["from_website"] is True
    c = r["config"]
    assert c.get("quantization") == "fp8"
    assert c.get("reasoning_parser") == "qwen3"


@requires_hub
def test_fetch_hf_card_returns_readme_and_config():
    remote = ac.fetch_hf_card(NVIDIA, timeout=20.0)
    assert remote.get("readme") and "vllm" in remote["readme"].lower()
    assert isinstance(remote.get("config"), dict)
    assert remote.get("fetched"), "must record fetched URLs"
    assert all("huggingface.co" in u for u in remote["fetched"])


def test_offline_does_not_claim_live_website(monkeypatch):
    """When fetch fails, from_website must be False (no silent live success)."""

    def _fail_fetch(model_id: str, timeout: float = 20.0):
        return {
            "model_id": model_id,
            "readme": None,
            "config": None,
            "api": None,
            "card_url": f"https://huggingface.co/{model_id}",
            "errors": ["forced offline"],
            "fetched": [],
        }

    monkeypatch.setattr(ac, "fetch_hf_card", _fail_fetch)
    # Still may use local cache for unsloth if present
    r = ac.recommend(UNSLOTH, mode="lab_safe", fetch_remote=True)
    assert r["from_website"] is False


def test_http_get_retries_anonymous_on_401(monkeypatch):
    """Bad Bearer must not poison public card fetches."""
    calls: list[str | None] = []

    class FakeResp:
        def __init__(self, body: bytes):
            self._body = body
            self.headers = {"Content-Type": "text/plain; charset=utf-8"}

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=20.0):
        # Request may store auth in header_items / unredirected_hdrs
        auth = None
        try:
            auth = req.get_header("Authorization")
        except Exception:
            auth = None
        if not auth:
            auth = getattr(req, "headers", {}).get("Authorization")
        calls.append(auth)
        if auth:
            raise HTTPError(req.full_url, 401, "Unauthorized", hdrs=None, fp=None)  # type: ignore[arg-type]
        return FakeResp(b"# hello card\n\nvllm serve org/model\n")

    from urllib.error import HTTPError

    monkeypatch.setattr(ac, "_HF_TOKEN_USABLE", None)
    monkeypatch.setattr(ac, "_hf_token", lambda: "hf_bad_token")
    # autoconfig imports urlopen into its module namespace
    monkeypatch.setattr(ac, "urlopen", fake_urlopen)

    body, err = ac._http_get("https://huggingface.co/org/model/raw/main/README.md")
    assert err is None, err
    assert body and "vllm serve" in body
    assert any(c for c in calls if c)  # tried with token
    assert any(c is None for c in calls)  # then anonymous
    assert ac._HF_TOKEN_USABLE is False


def test_mtp_speculative_config_not_duplicated_in_extra():
    args = [
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":2}',
        "--trust-remote-code",
    ]
    cfg = ac._args_to_config(args, [])
    assert cfg.get("mtp") is True
    assert cfg.get("mtp_num_tokens") == 2
    assert "speculative-config" not in (cfg.get("extra_flags") or "")


def test_strip_flag_from_extra():
    s = ac._strip_flag_from_extra(
        '--foo 1 --speculative-config \'{"method":"mtp"}\' --bar',
        "--speculative-config",
    )
    assert "speculative-config" not in s
    assert "--foo" in s and "--bar" in s


def test_mixed_checkpoint_prefers_spark_salvage_over_bare():
    """After flashinfer penalty + salvage, DGX Spark recipe should beat bare serve."""
    readme = (FIX / "card_mixed_moe_nvfp4.md").read_text()
    cfg = json.loads((FIX / "config_mixed_compressed_tensors.json").read_text())
    detected = ac.analyze_config(cfg, "example/Mixed-MoE-NVFP4")
    cands = ac.extract_serve_candidates(readme, detected=detected)
    best = cands[0]
    # Selected recipe should carry Spark signal (CUTE or DGX section), not bare-only
    blob = (best.raw + " " + (best.section or "")).lower()
    assert "spark" in blob or "cute" in blob or best.config.get("docker_env")
    assert (best.config.get("moe_backend") or "") != "flashinfer_b12x" or any(
        "PENALTY" in r for r in best.reasons
    )


def test_checkpoint_safety_strips_marlin_on_moe():
    detected = {
        "is_moe": True,
        "has_nvfp4": True,
        "quant_flag": "modelopt",
        "quant_method": "modelopt",
        "family": "qwen",
    }
    serve_cfg = {
        "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "quantization": "modelopt",
        "moe_backend": "marlin",
        "mtp": True,
        "mtp_num_tokens": 3,
        "extra_flags": '--attention-backend flashinfer --speculative-config \'{"method":"mtp"}\'',
        "kv_cache_dtype": "fp8",
        "max_num_seqs": 4,
        "docker_env": [],
    }
    warnings: list[str] = []
    rationale: list[str] = []
    ac._apply_checkpoint_safety(serve_cfg, detected, warnings, rationale)
    assert serve_cfg["moe_backend"] == ""
    assert any("marlin" in w for w in warnings)
    ac._apply_first_boot_defaults(
        serve_cfg, mode="workflow_max", detected=detected, warnings=warnings, rationale=rationale
    )
    assert serve_cfg["mtp"] is False
    assert "speculative-config" not in (serve_cfg.get("extra_flags") or "")


def test_marlin_kept_for_nemotron_family():
    detected = {
        "is_moe": True,
        "has_nvfp4": True,
        "quant_flag": "modelopt",
        "quant_method": "modelopt",
        "family": "nemotron",
    }
    assert ac._marlin_unsafe_for_checkpoint(detected) is False
    cfg = {"moe_backend": "marlin", "mtp": False, "extra_flags": "", "docker_env": []}
    warnings: list[str] = []
    rationale: list[str] = []
    ac._apply_checkpoint_safety(cfg, detected, warnings, rationale)
    assert cfg["moe_backend"] == "marlin"
    ac._apply_first_boot_defaults(
        cfg, mode="workflow_max", detected=detected, warnings=warnings, rationale=rationale
    )
    assert cfg["moe_backend"] == "marlin"


def test_scrub_unexpanded_shell_vars():
    warnings: list[str] = []
    out = ac._scrub_unexpanded_shell_vars(
        "--mamba-backend flashinfer --speculative_config.model $DSPARK_CKPT --speculative_config.method dspark",
        warnings,
    )
    assert "$" not in out
    assert "--speculative_config.method dspark" in out
    assert "--mamba-backend flashinfer" in out
    assert any("DSPARK" in w or "$" in w for w in warnings)


def test_resolve_stock_image_raises_never_downgrades():
    rat: list[str] = []
    assert (
        ac._resolve_stock_image("vllm/vllm-openai:v0.27.0", "vllm/vllm-openai:v0.27.1", rat)
        == "vllm/vllm-openai:v0.27.1"
    )
    rat.clear()
    assert (
        ac._resolve_stock_image("vllm/vllm-openai:v0.27.1", "vllm/vllm-openai:v0.26.0", rat)
        == "vllm/vllm-openai:v0.27.1"
    )
    assert any("downgrade" in r.lower() or "older" in r.lower() for r in rat)
    # Anemll overlay image never replaced by stock card pin
    assert (
        ac._resolve_stock_image(
            "ghcr.io/anemll/dspark-vllm-gx10:0.1.1", "vllm/vllm-openai:v0.27.1", []
        )
        == "ghcr.io/anemll/dspark-vllm-gx10:0.1.1"
    )


def test_recommend_requires_model():
    with pytest.raises(ValueError, match="model"):
        ac.recommend("", fetch_remote=False)


# ─── UI / API wiring (static structure of shipped sources) ───────────────────


def test_api_route_wires_recommend():
    routes = Path(__file__).resolve().parents[1] / "app" / "api" / "routes.py"
    text = routes.read_text()
    assert '@router.get("/serve/recommend")' in text
    assert "autoconfig.recommend" in text


def test_serve_ui_wires_auto_configure():
    root = Path(__file__).resolve().parents[3]  # local-ai-lab monorepo root
    serve_tsx = root / "apps" / "web" / "app" / "server" / "page.tsx"
    api_ts = root / "apps" / "web" / "lib" / "api.ts"
    assert serve_tsx.is_file() and api_ts.is_file(), f"missing UI: {serve_tsx} / {api_ts}"
    st = serve_tsx.read_text()
    at = api_ts.read_text()
    assert "recommendServe" in at
    assert "/serve/recommend" in at
    assert "recommendServe" in st
    assert "applyConfig" in st
    assert "from_website" in st
    assert "Auto-configure" in st
    # Recipe scoring reasons (e.g. PENALTY for flashinfer) must render in UI
    assert "cr.reasons" in st or "reasons" in st
    assert "card_recipes" in st
    assert "warnings" in st
    # Topology-aware auto-config surfaces cluster plan + TP in the form
    assert "rec.topology" in st
    assert "tensor_parallel_size" in st
    # Job log panel must remain present (Job dock streams serve logs)
    assert "Job" in st and "LogView" in st


def test_fixture_card_mixed_moe_surfaces_flashinfer_warning_via_recommend_path(monkeypatch):
    """End-to-end recommend with fixture card+config (no network): warning must fire."""
    readme = (FIX / "card_mixed_moe_nvfp4.md").read_text()
    cfg = json.loads((FIX / "config_mixed_compressed_tensors.json").read_text())

    def fake_fetch(model_id: str, timeout: float = 20.0):
        return {
            "model_id": model_id,
            "readme": readme,
            "config": cfg,
            "api": {"tags": ["compressed-tensors", "moe"]},
            "card_url": f"https://huggingface.co/{model_id}",
            "errors": [],
            "fetched": [
                f"https://huggingface.co/{model_id}/raw/main/README.md",
                f"https://huggingface.co/{model_id}/raw/main/config.json",
            ],
        }

    monkeypatch.setattr(ac, "fetch_hf_card", fake_fetch)
    monkeypatch.setattr(
        ac,
        "load_local_fallback",
        lambda model_id: {"config": None, "readme": None, "notes": []},
    )
    r = ac.recommend("example/Mixed-MoE-NVFP4", mode="lab_safe", fetch_remote=True)
    assert r["from_website"] is True
    assert (r["config"].get("moe_backend") or "") == ""
    assert any("flashinfer_b12x" in w for w in (r.get("warnings") or []))
    assert any("flashinfer" in x.lower() or "SAFETY" in x for x in (r.get("rationale") or []))


# ─── Topology-aware auto-config + model-family overlay ───────────────────────

DSV4 = "deepseek-ai/DeepSeek-V4-Flash-0731"


def _two_spark_topo():
    nodes = [
        {"id": "spark1", "qsfp_ip": "10.100.8.1", "qsfp_if": "enp1s0f1np1", "ram_gib": 121.7, "local": True, "ssh_host": "spark1"},
        {"id": "spark2", "qsfp_ip": "10.100.8.2", "qsfp_if": "enp1s0f1np1", "ram_gib": 121.7, "local": False, "ssh_host": "spark2"},
    ]
    return {"nodes": 2, "node_list": nodes, "head": nodes[0], "workers": [nodes[1]], "fabric_ok": True, "available": True}


def _one_spark_topo():
    t = _two_spark_topo()
    t.update({"nodes": 1, "workers": [], "fabric_ok": False})
    return t


def test_family_overlay_matches_deepseek_v4():
    ov = ac._family_overlay(DSV4, {})
    assert ov is not None
    assert ov["family_key"] == "deepseek_v4_dspark"
    c = ov["config"]
    assert c["image"].startswith("ghcr.io/anemll/dspark-vllm-gx10")
    assert c["kv_cache_dtype"] == "nvfp4_ds_mla"
    assert c["moe_backend"] == "flashinfer_b12x"
    assert c["tool_call_parser"] == "deepseek_v4"
    # DSpark rides --speculative-config (method=dspark), not the structured mtp flag
    assert c["mtp"] is False
    assert "dspark" in c["extra_flags"]


def test_family_overlay_ignores_normal_models():
    assert ac._family_overlay(NVIDIA, {}) is None
    assert ac._family_overlay("meta-llama/Llama-4-Scout", {}) is None


def test_topology_two_sparks_sets_tp2_and_fabric():
    cfg = ac._empty_config(DSV4)
    warnings, rationale = [], []
    ac._apply_topology(cfg, overlay=ac._family_overlay(DSV4, {}), topology=_two_spark_topo(), weights_gib=155.4, mode="workflow_max", warnings=warnings, rationale=rationale)
    assert cfg["tensor_parallel_size"] == 2
    assert "--nnodes 2" in cfg["extra_flags"]
    assert "--master-addr 10.100.8.1" in cfg["extra_flags"]
    env = cfg["docker_env"]
    assert any(e == "VLLM_HOST_IP=10.100.8.1" for e in env)
    assert any(e == "WORKER_VLLM_HOST_IP=10.100.8.2" for e in env)
    assert any(e.startswith("NCCL_SOCKET_IFNAME=enp1s0f1np1") for e in env)
    assert not warnings, f"fabric ok → no warnings, got {warnings}"


def test_topology_one_spark_strips_multinode_and_dp():
    cfg = ac._empty_config(DSV4)
    cfg["extra_flags"] = "--data-parallel-size 4 --tensor-parallel-size 2"
    warnings, rationale = [], []
    ac._apply_topology(cfg, overlay=ac._family_overlay(DSV4, {}), topology=_one_spark_topo(), weights_gib=21.0, mode="lab_safe", warnings=warnings, rationale=rationale)
    assert cfg.get("tensor_parallel_size") is None
    assert "--data-parallel-size" not in cfg["extra_flags"]
    assert "--nnodes" not in cfg["extra_flags"]
    assert not any(e.startswith("VLLM_HOST_IP=") for e in cfg["docker_env"])
    assert any("single-node" in w or "1 Spark" in w for w in warnings)


def test_recommend_dsv4_two_sparks_end_to_end(monkeypatch):
    """Full recommend: overlay + topology produce Mia's 2-node DSv4 recipe."""
    monkeypatch.setattr(ac, "_cluster_topology", _two_spark_topo)
    r = ac.recommend(DSV4, mode="workflow_max", fetch_remote=False)
    c = r["config"]
    assert r["topology"]["nodes"] == 2
    assert r["topology"]["overlay"] == "deepseek_v4_dspark"
    assert c["image"].startswith("ghcr.io/anemll/dspark-vllm-gx10")
    assert c["kv_cache_dtype"] == "nvfp4_ds_mla"
    assert c["moe_backend"] == "flashinfer_b12x"
    assert c["tensor_parallel_size"] == 2
    assert c["util"] == 0.80
    assert c["max_model_len"] == 1048576  # 1M preserved (not clamped by envelope)
    ex = c["extra_flags"]
    assert "--nnodes 2" in ex and "--speculative-config" in ex and "dspark" in ex
    assert "--data-parallel-size" not in ex
    assert "--enable-expert-parallel" not in ex  # card garbage dropped with card path


def test_serve_build_args_passes_tp_through():
    from app.services import serve

    args = serve._build_vllm_args(
        util=0.8,
        max_model_len=1048576,
        port=8000,
        kv_cache_dtype="nvfp4_ds_mla",
        moe_backend="flashinfer_b12x",
        extra_flags="--nnodes 2 --master-addr 10.100.8.1 --tensor-parallel-size 9",
        tensor_parallel_size=2,
    )
    joined = " ".join(args)
    # TP honored from the structured field; the duplicate extra TP is stripped (envelope owns it)
    assert joined.count("--tensor-parallel-size") == 1
    i = args.index("--tensor-parallel-size")
    assert args[i + 1] == "2"
    assert "--nnodes 2" in joined and "--master-addr 10.100.8.1" in joined


# ─── Placement engine + multi-node launcher ──────────────────────────────────


def _topo(n: int):
    nodes = [
        {"id": f"spark{i+1}", "qsfp_ip": f"10.100.8.{i+1}", "qsfp_if": "enp1s0f1np1", "ram_gib": 121.7, "local": i == 0, "ssh_host": f"spark{i+1}"}
        for i in range(n)
    ]
    return {
        "nodes": n,
        "node_list": nodes,
        "head": nodes[0],
        "workers": nodes[1:],
        "fabric_ok": True,
        "available": True,
    }


def test_placement_small_model_single_node():
    p = ac.plan_placement(21.0, _topo(2), mode="workflow_max", overlay=None)
    assert p["nodes_needed"] == 1
    assert p["tensor_parallel_size"] == 1
    assert p["util_computed"] == 0.4  # (21+15)/121.7 = 0.30 -> clamped to 0.40 floor
    assert p["fits"] is True


def test_placement_dsv4_needs_two_nodes_computed_util():
    p = ac.plan_placement(155.4, _topo(2), mode="workflow_max", overlay=None)
    assert p["nodes_needed"] == 2
    assert p["tensor_parallel_size"] == 2
    assert p["per_node_weights_gib"] == 77.7
    assert p["util_computed"] == 0.76  # (77.7+15)/121.7
    assert p["fits"] is True


def test_placement_too_big_warns_no_fit():
    p = ac.plan_placement(400.0, _topo(2), mode="workflow_max", overlay=None)
    assert p["nodes_needed"] == 2  # capped at available
    assert p["fits"] is False  # 200 GiB/node won't fit


def test_weight_floor_blocks_deepseek_r1_without_blobs(monkeypatch):
    """P0.4: known oversized families must not claim fit when Hub blobs are empty."""
    monkeypatch.setattr(ac, "_http_get", lambda *a, **k: (None, "offline"))
    w = ac.estimate_weights_gib("deepseek-ai/DeepSeek-R1", None)
    assert w is not None and w >= 700
    p = ac.plan_placement(w, _topo(2), mode="workflow_max", overlay=None)
    assert p["fits"] is False


def test_weight_floor_spares_deepseek_v4_flash():
    """V4-Flash has a Spark overlay; floor must not force-block it."""
    assert ac._weight_floor_gib("deepseek-ai/DeepSeek-V4-Flash") is None
    assert ac._weight_floor_gib("deepseek-ai/DeepSeek-V4-Pro") == 900.0


def test_moe_config_estimate_not_near_zero(monkeypatch):
    monkeypatch.setattr(ac, "_http_get", lambda *a, **k: (None, "offline"))
    cfg = {
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "n_routed_experts": 32,  # below floor threshold; forces MoE-aware formula
        "moe_intermediate_size": 2048,
        "quantization_config": {"config_groups": {"g0": {"weights": {"num_bits": 8}}}},
    }
    w = ac.estimate_weights_gib("org/Custom-MoE-Offline", cfg)
    dense_only = round(12 * 61 * 7168 * 7168 * 1.0 / (1024**3), 1)
    assert w is not None and w > dense_only * 1.5


def test_placement_future_4node_minimal_and_full():
    # DSv4 on 4 nodes still uses only the 2 it needs (no waste)
    p = ac.plan_placement(155.4, _topo(4), mode="workflow_max", overlay=None)
    assert p["nodes_needed"] == 2
    # a much larger model spreads to all 4
    p2 = ac.plan_placement(420.0, _topo(4), mode="workflow_max", overlay=None)
    assert p2["nodes_needed"] == 4
    assert p2["tensor_parallel_size"] == 4


def test_multi_node_launch_head_worker_split():
    from app.services import serve

    launch = serve.build_multi_node_launch(
        image="img", model="m",
        vllm_args=["--host", "0.0.0.0", "--port", "8000", "--tensor-parallel-size", "2", "--kv-cache-dtype", "nvfp4_ds_mla", "--nnodes", "2"],
        env_list=["VLLM_HOST_IP=10.100.8.1", "WORKER_VLLM_HOST_IP=10.100.8.2", "NCCL_SOCKET_IFNAME=enp1s0f1np1"],
        head={"id": "spark1", "qsfp_ip": "10.100.8.1"},
        workers=[{"id": "spark2", "qsfp_ip": "10.100.8.2", "ssh_host": "spark2"}],
        nnodes=2, port=8000,
    )
    head = " ".join(launch["head"]["cmd"])
    worker = " ".join(launch["workers"][0]["cmd"])
    assert "--node-rank 0" in head and "--host 0.0.0.0" in head and "--headless" not in head
    assert "--node-rank 1" in worker and "--headless" in worker and "--host" not in worker
    assert head.count("--tensor-parallel-size") == 1  # structured flag deduped
    assert "VLLM_HOST_IP=10.100.8.2" in worker
    assert launch["workers"][0]["rank"] == 1


def test_multi_node_launch_clears_image_entrypoint():
    """Regression: Anemll image ships ENTRYPOINT=vllm. Without --entrypoint bash the
    wrapper is appended as vllm arguments and the container dies with exit 2
    ('unrecognized arguments'). Verified live on spark1/spark2 2026-08-07."""
    from app.services import serve

    launch = serve.build_multi_node_launch(
        image="ghcr.io/anemll/dspark-vllm-gx10:0.1.1", model="m",
        vllm_args=["--kv-cache-dtype", "nvfp4_ds_mla"],
        env_list=[],
        head={"id": "spark1", "qsfp_ip": "10.100.8.1"},
        workers=[{"id": "spark2", "qsfp_ip": "10.100.8.2", "ssh_host": "spark2"}],
        nnodes=2, port=8000,
    )
    for cmd in (launch["head"]["cmd"], launch["workers"][0]["cmd"]):
        assert "--entrypoint" in cmd, "must clear the image ENTRYPOINT"
        assert cmd[cmd.index("--entrypoint") + 1] == "bash"
        # image must be followed by bash flags, never a second literal "bash"
        img_i = cmd.index("ghcr.io/anemll/dspark-vllm-gx10:0.1.1")
        assert cmd[img_i + 1] == "-lc", f"expected bash flags after image, got {cmd[img_i + 1]}"
        assert cmd[img_i + 1 : img_i + 2] != ["bash"]
        # the real serve command still reaches vllm
        assert "vllm" in cmd and "serve" in cmd


def test_stop_all_kills_remote_worker_without_state_file(monkeypatch):
    """Regression 2026-08-10: Stop only removed local spark-vllm-n0 when
    multinode_serve.json was missing, leaving spark2 TP worker up (~100 GiB)."""
    from app.services import serve

    monkeypatch.setattr(serve, "_MULTINODE_STATE", type(serve._MULTINODE_STATE)("/no/such/multinode_serve.json"))
    monkeypatch.setattr(serve, "SPARK_LAB", type(serve.SPARK_LAB)("/no/such/spark_lab.sh"))
    monkeypatch.setattr(serve, "list_vllm_containers", lambda: [])

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        class R:
            returncode = 0
            stdout = "spark-vllm-n1\nother\n" if "docker ps" in " ".join(cmd) else ""
            stderr = ""
        return R()

    monkeypatch.setattr(serve.subprocess, "run", fake_run)

    result = serve.stop_all()
    assert result["ok"] is True
    assert "spark2:spark-vllm-n1" in result["stopped"]
    # list + rm over ssh to spark2
    joined = [" ".join(c) for c in calls]
    assert any("ssh" in j and "spark2" in j and "docker ps" in j for j in joined)
    assert any("ssh" in j and "spark2" in j and "docker rm -f spark-vllm-n1" in j for j in joined)


def test_overlay_file_extends_builtins(monkeypatch, tmp_path):
    """User can add a future model via data/serve_overlays.json with no code change."""
    import json as _json
    import app.config as cfgmod

    (tmp_path / "serve_overlays.json").write_text(_json.dumps([
        {
            "match": {"all": ["futuremodel"], "any": ["nvfp4"]},
            "family_key": "future_x",
            "label": "Future Model X",
            "source": "https://example.com",
            "config": {"image": "img:x", "kv_cache_dtype": "nvfp4_ds_mla"},
            "rationale": [],
        }
    ]))
    monkeypatch.setattr(cfgmod, "DATA_DIR", tmp_path)
    ov = ac._family_overlay("org/FutureModel-NVFP4", {})
    assert ov is not None and ov["family_key"] == "future_x"
    # built-in still present
    assert ac._family_overlay("deepseek-ai/DeepSeek-V4-Flash-0731", {}) is not None


def test_shipped_serve_overlays_json_loads_minimax():
    """Repo data/serve_overlays.json must be valid and match MiniMax families."""
    from app.config import DATA_DIR

    f = DATA_DIR / "serve_overlays.json"
    assert f.is_file(), f"missing shipped overlays at {f}"
    ov = ac._family_overlay(
        "MiniMaxAI/MiniMax-M2-NVFP4",
        {"family": "minimax_m2"},
    )
    assert ov is not None and ov["family_key"] == "minimax_m2"
    assert ov["config"].get("reasoning_parser") == "minimax_m2"
    ov3 = ac._family_overlay("MiniMaxAI/MiniMax-M3", {"family": "minimax_m3"})
    assert ov3 is not None and ov3["family_key"] == "minimax_m3"
    assert "--block-size 128" in (ov3["config"].get("extra_flags") or "")


def test_family_overlay_matches_detected_family_without_id_hint():
    """match.family alone can select when id substrings are also satisfied loosely."""
    ov = ac._family_overlay(
        "org/Some-MiniMax-Checkpoint-M2",
        {"family": "minimax_m2"},
    )
    assert ov is not None and ov["family_key"] == "minimax_m2"

def test_size_memory_clamps_huge_context_single_node():
    cfg = {
        "max_model_len": 1048576,
        "util": 0.85,
        "max_num_seqs": 4,
        "kv_cache_dtype": "fp8",
        "tensor_parallel_size": 1,
    }
    hf = {
        "num_hidden_layers": 48,
        "num_key_value_heads": 8,
        "num_attention_heads": 32,
        "hidden_size": 4096,
        "head_dim": 128,
    }
    rationale: list[str] = []
    warnings: list[str] = []
    ac._size_memory_for_spark(
        cfg,
        hf_config=hf,
        detected={"family": "qwen"},
        weights_gib=40.0,
        node_ram_gib=121.7,
        mode="workflow_max",
        rationale=rationale,
        warnings=warnings,
    )
    assert cfg["max_model_len"] < 1048576
    assert cfg["max_model_len"] in ac._CONTEXT_LADDER
    assert any("MEMORY:" in r for r in rationale)


def test_size_memory_skips_multinode_tp():
    cfg = {
        "max_model_len": 1048576,
        "util": 0.8,
        "max_num_seqs": 6,
        "kv_cache_dtype": "nvfp4_ds_mla",
        "tensor_parallel_size": 2,
    }
    rationale: list[str] = []
    warnings: list[str] = []
    ac._size_memory_for_spark(
        cfg,
        hf_config={"num_hidden_layers": 60},
        detected={"family": "deepseek_v4"},
        weights_gib=155.4,
        node_ram_gib=121.7,
        mode="workflow_max",
        rationale=rationale,
        warnings=warnings,
    )
    assert cfg["max_model_len"] == 1048576


def test_stock_image_semver_and_at_least():
    assert ac._stock_image_semver("vllm/vllm-openai:v0.27.1") == (0, 27, 1)
    assert ac._stock_image_semver("ghcr.io/anemll/dspark-vllm-gx10:0.1.1") is None
    assert ac._image_at_least("vllm/vllm-openai:v0.27.1", 0, 27, 0) is True
    assert ac._image_at_least("vllm/vllm-openai:v0.25.0", 0, 27, 0) is False


def test_marlin_version_gate_pure_nvfp4():
    det = {
        "family": "unknown",
        "is_moe": True,
        "has_nvfp4": True,
        "is_mixed_nvfp4_fp8": False,
        "quant_flag": "modelopt_fp4",
    }
    # Old stock image: strip marlin for non-Nemotron pure NVFP4
    assert ac._marlin_unsafe_for_checkpoint(det, "vllm/vllm-openai:v0.25.0") is True
    # Current lab image: keep marlin
    assert ac._marlin_unsafe_for_checkpoint(det, "vllm/vllm-openai:v0.27.1") is False
    # Nemotron always keeps marlin
    det["family"] = "nemotron"
    assert ac._marlin_unsafe_for_checkpoint(det, "vllm/vllm-openai:v0.25.0") is False


def test_analyze_config_detects_vl():
    d = ac.analyze_config(
        {"architectures": ["Qwen2_5_VLForConditionalGeneration"], "model_type": "qwen2_5_vl"},
        "Qwen/Qwen2.5-VL-7B-Instruct",
    )
    assert d["is_vl"] is True


def test_vl_gets_language_model_only():
    cfg = {"extra_flags": "--max-num-batched-tokens 8192", "moe_backend": ""}
    warnings: list[str] = []
    rationale: list[str] = []
    ac._apply_vl_spark_defaults(
        cfg, {"is_vl": True}, warnings, rationale
    )
    assert "--language-model-only" in cfg["extra_flags"]
    assert any("language-model-only" in r for r in rationale)


def test_resolve_dspark_draft_from_card():
    readme = "export DSPARK_CKPT=nvidia/Foo-DSpark\nvllm serve x --speculative_config.method dspark"
    assert ac._resolve_dspark_draft_model(readme) == "nvidia/Foo-DSpark"


def test_ensure_dspark_fills_missing_draft():
    cfg = {
        "image": "vllm/vllm-openai:v0.27.1",
        "extra_flags": "--speculative_config.method dspark --speculative_config.num_speculative_tokens 3",
    }
    warnings: list[str] = []
    rationale: list[str] = []
    ac._ensure_dspark_draft_or_strip(
        cfg,
        "export DSPARK_CKPT=nvidia/Bar-DSpark",
        warnings,
        rationale,
    )
    assert "--speculative_config.model nvidia/Bar-DSpark" in cfg["extra_flags"]
    assert any("DSpark draft" in r for r in rationale)


def test_ensure_dspark_strips_when_no_draft():
    cfg = {
        "image": "vllm/vllm-openai:v0.27.1",
        "extra_flags": "--mamba-backend flashinfer --speculative_config.method dspark",
    }
    warnings: list[str] = []
    rationale: list[str] = []
    ac._ensure_dspark_draft_or_strip(cfg, "no draft here", warnings, rationale)
    assert "dspark" not in (cfg["extra_flags"] or "").lower()
    assert "--mamba-backend flashinfer" in cfg["extra_flags"]


def test_ensure_dspark_skips_anemll_image():
    cfg = {
        "image": "ghcr.io/anemll/dspark-vllm-gx10:0.1.1",
        "extra_flags": "--speculative-config " + json.dumps({"method": "dspark", "num_speculative_tokens": 5}),
    }
    warnings: list[str] = []
    rationale: list[str] = []
    ac._ensure_dspark_draft_or_strip(cfg, "", warnings, rationale)
    assert "dspark" in cfg["extra_flags"]

