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
    # Card may still say umbrella `modelopt`; config.json MIXED_PRECISION (FP8+NVFP4
    # layers) correctly selects a ModelOpt sibling via quant detection.
    assert (c.get("quantization") or "").startswith("modelopt"), c.get("quantization")
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
    """Mixed FP8+NVFP4 Qwen MoE still drops marlin; MTP first-boot defaults still apply."""
    detected = {
        "is_moe": True,
        "has_nvfp4": True,
        "is_mixed_nvfp4_fp8": True,
        "quant_flag": "compressed-tensors",
        "quant_method": "compressed-tensors",
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


def test_serve_keeps_marlin_for_nemotron_without_local_config():
    """Start must not strip marlin for Nemotron when HF cache is empty (a3b trap).

    Family detection from model id alone is enough for _marlin_unsafe_for_checkpoint
    — serve.py now calls analyze_config({}, model) instead of dropping on 'a3b'.
    """
    det = ac.analyze_config({}, "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
    assert det["family"] == "nemotron"
    assert ac._marlin_unsafe_for_checkpoint(det, "vllm/vllm-openai:v0.27.1") is False
    # Id-only Qwen A3B-NVFP4 is treated as pure NVFP4 → marlin legal on ≥0.27.
    q = ac.analyze_config({}, "nvidia/Qwen3.6-35B-A3B-NVFP4")
    assert q["family"] == "qwen"
    assert q["has_nvfp4"] is True
    assert ac._marlin_unsafe_for_checkpoint(q, "vllm/vllm-openai:v0.27.1") is False


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


def test_scrub_unexpanded_docker_env():
    warnings: list[str] = []
    out = ac._scrub_unexpanded_docker_env(
        ["FOO=bar", "CKPT=$MODEL_CKPT", "BAZ=ok"],
        warnings,
    )
    assert out == ["FOO=bar", "BAZ=ok"]
    assert any("MODEL_CKPT" in w or "$" in w for w in warnings)


def test_capability_min_image_for_modelopt_mixed():
    img = ac._capability_min_stock_image(
        {"quantization": "modelopt_mixed", "moe_backend": "marlin", "reasoning_parser": "nemotron_v3"},
        {"has_nvfp4": True, "family": "nemotron"},
    )
    assert img == "vllm/vllm-openai:v0.27.0"


def test_check_serve_loadability_absolute_only():
    # 50 GiB on 122 GiB node fits at 0.85 after 15 GiB reserve.
    ok85, _ = ac.check_serve_loadability(
        mode="auto",
        weights_gib=50.0,
        node_ram_gib=121.7,
        nodes_used=1,
        util=0.85,
    )
    assert ok85 is True
    # 400 GiB cannot fit even at the 0.85 ceiling.
    ok_big, msg = ac.check_serve_loadability(
        mode="lab_safe",  # ignored
        weights_gib=400.0,
        node_ram_gib=121.7,
        nodes_used=1,
        util=0.4,
    )
    assert ok_big is False
    assert msg and "do not fit" in msg


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


def test_parse_docker_run_puts_image_on_candidate():
    """Winning docker run image lands on ServeCandidate.config['image']."""
    text = (
        "docker run --gpus all -p 8000:8000 vllm/vllm-openai:v0.28.0 "
        "org/Model-NVFP4 --quantization modelopt_fp4 --moe-backend marlin"
    )
    c = ac._parse_one_serve_command(text)
    assert c is not None
    assert c.config.get("image") == "vllm/vllm-openai:v0.28.0"
    assert c.model == "org/Model-NVFP4"
    assert c.config.get("quantization") == "modelopt_fp4"
    assert c.config.get("moe_backend") == "marlin"


def test_card_image_re_matches_nvcr_and_eugr():
    """_CARD_IMAGE_RE recognizes stock, Anemll, nvcr vLLM, and eugr Spark pins."""
    text = (
        "images: vllm/vllm-openai:v0.27.1 "
        "ghcr.io/anemll/dspark-vllm-gx10:0.1.1 "
        "nvcr.io/nvidia/vllm:25.01-py3 "
        "eugr/spark-vllm:0.2.0 "
        "and untagged vllm/vllm-openai alone"
    )
    hits = ac._CARD_IMAGE_RE.findall(text)
    joined = " ".join(hits).lower()
    assert "vllm/vllm-openai:v0.27.1" in joined
    assert "ghcr.io/anemll/dspark-vllm-gx10:0.1.1" in joined
    assert "nvcr.io/nvidia/vllm:25.01-py3" in joined
    assert "eugr/spark-vllm:0.2.0" in joined


def test_docker_recipe_image_applied_before_marlin_gates(monkeypatch):
    """Fixture docker recipe image appears in applied config when no overlay.

    Image must be resolved early enough that marlin/capability see the intended pin
    (not only late _parse_card_image_requirement after safety).
    """
    readme = """## DGX Spark

```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:v0.28.0 \\
  example/Docker-Image-Model --quantization modelopt_fp4 --moe-backend marlin
```
"""
    hf_config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
        },
        "num_hidden_layers": 4,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 2048,
        "vocab_size": 32000,
        "max_position_embeddings": 8192,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
    }

    monkeypatch.setattr(ac, "fetch_hf_card", lambda m: {
        "readme": readme,
        "config": hf_config,
        "api": {"tags": ["nvfp4"]},
        "fetched": [],
        "errors": [],
    })
    monkeypatch.setattr(ac, "_family_overlay", lambda *a, **k: None)
    monkeypatch.setattr(ac, "estimate_weights_gib", lambda *a, **k: 12.0)
    monkeypatch.setattr(
        ac,
        "_cluster_topology",
        lambda: {
            "nodes": 1,
            "node_list": [{"name": "spark-1", "ram_gib": 121.7, "online": True, "local": True}],
            "head": {"name": "spark-1", "ram_gib": 121.7, "online": True, "local": True},
            "workers": [],
            "fabric_ok": False,
            "available": True,
        },
    )

    rec = ac.recommend("example/Docker-Image-Model", mode="workflow_max", fetch_remote=True)
    cfg = rec["config"]
    assert cfg["image"] == "vllm/vllm-openai:v0.28.0", (
        f"docker pin must land on config.image early; got {cfg.get('image')!r}"
    )
    # Candidate extraction also surfaces image for scoring / Apply recipe UI
    cands = rec.get("card_recipes") or []
    assert cands, "expected card_recipes from docker snippet"
    assert any((c.get("config") or {}).get("image") == "vllm/vllm-openai:v0.28.0" for c in cands) or any(
        "v0.28.0" in (c.get("raw") or "") for c in cands
    )


def test_early_image_resolve_never_replaces_anemll():
    """Stock card pin + capability must not replace Anemll/DSpark image."""
    cfg = {"image": "ghcr.io/anemll/dspark-vllm-gx10:0.1.1", "quantization": "modelopt_mixed"}
    rat: list[str] = []
    out = ac._resolve_image_for_gates(
        cfg,
        mode="workflow_max",
        candidate_image="vllm/vllm-openai:v0.28.0",
        card_image="vllm/vllm-openai:v0.28.0",
        detected={"has_nvfp4": True, "family": "nemotron"},
        rationale=rat,
        warnings=[],
    )
    assert out == "ghcr.io/anemll/dspark-vllm-gx10:0.1.1"
    assert cfg["image"] == out


def test_early_image_resolve_never_downgrades_lab_default():
    """Older card docker pin must not pull lab default down."""
    cfg: dict = {"image": "", "quantization": "modelopt_fp4", "moe_backend": "marlin"}
    rat: list[str] = []
    out = ac._resolve_image_for_gates(
        cfg,
        mode="workflow_max",
        candidate_image="vllm/vllm-openai:v0.25.0",
        card_image="vllm/vllm-openai:v0.25.0",
        detected={"has_nvfp4": True},
        rationale=rat,
        warnings=[],
    )
    # lab default is ≥0.27.1; capability floor also ≥0.27.0
    assert out.startswith("vllm/vllm-openai:")
    ver = ac._stock_image_semver(out)
    assert ver is not None and ver >= (0, 27, 0)
    assert "v0.25" not in out


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
    assert "mode=" not in at.split("recommendServe")[1][:400]
    assert "recommendServe" in st
    assert "applyConfig" in st
    assert "from_website" in st
    assert "Auto-configure" in st
    assert 'ariaLabel="Serve mode envelope"' not in st
    assert "re-run Auto-configure after switch" not in st
    assert "Lab Safe" not in st
    assert "Workflow Max" not in st
    assert "lab_safe" not in st
    assert "workflow_max" not in st
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


def test_family_overlay_ignores_deepseek_v3_flash():
    """`flash` alone must not select the V4 DSpark overlay."""
    assert ac._family_overlay("deepseek-ai/DeepSeek-V3-Flash", {}) is None
    assert ac._family_overlay("deepseek-ai/DeepSeek-V4-Flash", {}) is not None


def test_topology_two_sparks_sets_tp2_and_fabric(monkeypatch):
    """Fabric-ok two-node plan must not depend on the runner having a RoCE NIC."""
    monkeypatch.setattr(ac, "_ib_hca_for_iface", lambda iface: "rocep1s0f1")
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
    assert any(e == "NCCL_IB_HCA=rocep1s0f1" for e in env)
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
    # (21+15)/121.7 ≈ 0.30 < Workflow Max default 0.85 → leave util to the envelope
    assert p["util_computed"] is None
    assert p["fits"] is True


def test_placement_dsv4_needs_two_nodes_computed_util():
    p = ac.plan_placement(155.4, _topo(2), mode="workflow_max", overlay=None)
    assert p["nodes_needed"] == 2
    assert p["tensor_parallel_size"] == 2
    assert p["per_node_weights_gib"] == 77.7
    # (77.7+15)/121.7 ≈ 0.76 still under 0.85 default → envelope owns util
    assert p["util_computed"] is None
    assert p["fits"] is True


def test_placement_elevates_util_only_when_weights_require_it():
    # Default ceiling is 0.85; only elevate util_computed when weights need more.
    p = ac.plan_placement(40.0, _topo(1), overlay=None)
    assert p["nodes_needed"] == 1
    # (40+15)/121.7 ≈ 0.45 < 0.85 → envelope owns util
    assert p["util_computed"] is None
    p2 = ac.plan_placement(21.0, _topo(1), overlay=None)
    assert p2["util_computed"] is None


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
        # multi-node HF cache mount + HF_HOME parity target
        assert f":{serve.HF_CACHE_IN_CONTAINER}" in " ".join(cmd)
        assert f"HF_HOME={serve.HF_CACHE_IN_CONTAINER}" in cmd


def test_single_node_anemll_hf_home_and_entrypoint_parity():
    """P0.HF_HOME_PARITY: single-node Anemll mounts host HF cache at the same
    in-container path as multi-node and sets HF_HOME; clears ENTRYPOINT like multi-node."""
    from app.services import serve
    from pathlib import Path

    image = "ghcr.io/anemll/dspark-vllm-gx10:0.1.1"
    args = serve._build_vllm_args(util=0.4, max_model_len=8192, port=8000, tensor_parallel_size=1)
    cmd = serve.build_single_node_docker_cmd(
        image=image,
        model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        vllm_args=args,
        env_list=["NCCL_DEBUG=WARN"],
        container="vllm-lab-safe",
        port=8000,
        hf_token=None,
    )
    joined = " ".join(cmd)
    host_hf = str(Path.home() / ".cache" / "huggingface")
    assert f"{host_hf}:{serve.HF_CACHE_IN_CONTAINER}" in joined
    assert f"HF_HOME={serve.HF_CACHE_IN_CONTAINER}" in cmd
    # Must NOT use the old /root/.cache path (diverged from multi-node).
    assert "/root/.cache/huggingface" not in joined
    # Anemll entrypoint clear + bash wrapper (same pattern as multi-node).
    assert "--entrypoint" in cmd and cmd[cmd.index("--entrypoint") + 1] == "bash"
    img_i = cmd.index(image)
    assert cmd[img_i + 1] == "-lc"
    assert "vllm" in cmd and "serve" in cmd
    assert "NCCL_DEBUG=WARN" in cmd


def test_single_node_stock_vllm_openai_no_entrypoint_override():
    """Stock vllm-openai keeps image ENTRYPOINT; still gets HF_HOME mount parity."""
    from app.services import serve
    from pathlib import Path

    image = "vllm/vllm-openai:v0.27.1"
    args = serve._build_vllm_args(util=0.4, max_model_len=4096, port=8000)
    cmd = serve.build_single_node_docker_cmd(
        image=image,
        model="Qwen/Qwen2.5-0.5B-Instruct",
        vllm_args=args,
        env_list=[],
        container="vllm-lab-safe",
        port=8000,
    )
    joined = " ".join(cmd)
    host_hf = str(Path.home() / ".cache" / "huggingface")
    assert f"{host_hf}:{serve.HF_CACHE_IN_CONTAINER}" in joined
    assert f"HF_HOME={serve.HF_CACHE_IN_CONTAINER}" in cmd
    assert "--entrypoint" not in cmd
    # Stock: image then model (ENTRYPOINT already vllm serve).
    assert cmd[cmd.index(image) + 1] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_worker_docker_pull_quotes_image(monkeypatch):
    """Worker SSH pull must shell-quote the image ref (tags/repos safe over remote sh)."""
    from app.services import serve
    import shlex

    image = "ghcr.io/anemll/dspark-vllm-gx10:0.1.1"
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        class R:
            returncode = 0
            stdout = "ok"
            stderr = ""
        return R()

    monkeypatch.setattr(serve.subprocess, "run", fake_run)
    # Minimal launch with one worker so _launch_multi_node hits the pull path.
    launch = {
        "nnodes": 2,
        "image": image,
        "model": "m",
        "port": 8000,
        "head": {"rank": 0, "cmd": ["echo", "head"]},
        "workers": [
            {
                "node": "spark2",
                "ssh_host": "spark2",
                "rank": 1,
                "cmd": ["echo", "worker"],
            }
        ],
    }
    # Short-circuit readiness wait by making urlopen succeed immediately.
    monkeypatch.setattr(serve.urllib.request, "urlopen", lambda *a, **k: True)
    # Avoid writing multinode state under data/
    monkeypatch.setattr(serve, "_MULTINODE_STATE", type(serve._MULTINODE_STATE)("/tmp/lail-test-mn-state.json"))
    # _ensure_image_present first: image inspect fails → pull on head
    calls_n = {"n": 0}
    real_fake = fake_run

    def sequenced(cmd, **kwargs):
        calls_n["n"] += 1
        # First call is head docker image inspect — force "not local"
        if cmd[:3] == ["docker", "image", "inspect"]:
            class R:
                returncode = 1
                stdout = ""
                stderr = "not found"
            captured.append(list(cmd))
            return R()
        return real_fake(cmd, **kwargs)

    monkeypatch.setattr(serve.subprocess, "run", sequenced)
    serve._launch_multi_node(launch, port=8000)

    ssh_pulls = [
        c for c in captured
        if c and c[0] == "ssh" and any("docker pull" in str(x) for x in c)
    ]
    assert ssh_pulls, f"expected ssh docker pull, got {captured}"
    remote = ssh_pulls[0][-1]
    quoted = shlex.quote(image)
    assert quoted in remote or image in remote
    # Image must appear as a single shell-safe token (shlex.quote form).
    assert f"docker pull {quoted}" in remote


def test_stop_all_kills_remote_worker_without_state_file(monkeypatch):
    """Regression 2026-08-10: Stop only removed local spark-vllm-n0 when
    multinode_serve.json was missing, leaving spark2 TP worker up (~100 GiB)."""
    from app.services import serve

    monkeypatch.setenv(
        "LAIL_CLUSTER_JSON",
        json.dumps(
            {
                "name": "lab",
                "nodes": [
                    {"id": "head", "local": True, "vllm_url": "http://127.0.0.1:8000"},
                    {"id": "spark2", "local": False, "ssh_host": "spark2", "vllm_url": "http://127.0.0.1:8000"},
                ],
            }
        ),
    )
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
    q38 = ac._family_overlay(
        "unsloth/Qwen3.8-27B-NVFP4",
        {"family": "qwen"},
    )
    assert q38 is not None and q38["family_key"] == "qwen38_nvfp4"
    assert q38["config"].get("kv_cache_dtype") == "fp8"
    assert "--language-model-only" not in (q38["config"].get("extra_flags") or "")


def test_family_overlay_matches_detected_family_without_id_hint():
    """match.family alone can select when id substrings are also satisfied loosely."""
    ov = ac._family_overlay(
        "org/Some-MiniMax-Checkpoint-M2",
        {"family": "minimax_m2"},
    )
    assert ov is not None and ov["family_key"] == "minimax_m2"


def test_merge_extra_flags_dedupes_by_flag_name():
    """Overlay merge must not emit tokenizer/config-format (or any flag) twice."""
    merged = ac._merge_extra_flags(
        "--tokenizer-mode mistral --config-format mistral --foo 1",
        "--tokenizer-mode mistral --config-format mistral --block-size 128",
    )
    assert merged.count("--tokenizer-mode") == 1
    assert merged.count("--config-format") == 1
    assert "--block-size 128" in merged or "--block-size" in merged
    assert "--foo" in merged


def test_overlay_extra_flags_merge_no_duplicate_mistral(monkeypatch):
    """Fill + Magistral overlay: composed extra_flags keep tokenizer/config once."""
    ov = ac._family_overlay(
        "mistralai/Magistral-Small-2509",
        {"family": "mistral"},
    )
    assert ov is not None
    cfg = ac._empty_config("mistralai/Magistral-Small-2509")
    rationale: list[str] = []
    # Simulate family fill first (adds tokenizer/config-format extras).
    ac._fill_from_config_detection(
        cfg,
        {
            "family": "mistral",
            "quant_flag": "",
            "architectures": ["MagistralForCausalLM"],
            "model_type": "magistral",
        },
        rationale,
    )
    assert "--tokenizer-mode" in (cfg.get("extra_flags") or "")
    # Overlay merge path (same as recommend).
    overlay_cfg = ov["config"]
    for k, v in overlay_cfg.items():
        if k == "docker_env":
            cfg["docker_env"] = ac._dedupe_env(list(cfg.get("docker_env") or []) + list(v))
        elif k == "extra_flags":
            cfg["extra_flags"] = ac._merge_extra_flags(cfg.get("extra_flags") or "", v or "")
        else:
            cfg[k] = v
    ex = cfg.get("extra_flags") or ""
    assert ex.count("--tokenizer-mode") == 1
    assert ex.count("--config-format") == 1
    # docker_env: same key twice collapses
    env = ac._dedupe_env(["FOO=1", "BAR=2"] + ["FOO=3", "BAZ=9"])
    assert env.count("FOO=3") == 1
    assert not any(e == "FOO=1" for e in env)


def test_minimax_overlay_label_fp8_vs_nvfp4():
    """MiniMax M2 overlay label reflects quant suffix (not always NVFP4)."""
    nv = ac._family_overlay("MiniMaxAI/MiniMax-M2-NVFP4", {"family": "minimax_m2"})
    fp = ac._family_overlay("MiniMaxAI/MiniMax-M2-FP8", {"family": "minimax_m2"})
    plain = ac._family_overlay("MiniMaxAI/MiniMax-M2", {"family": "minimax_m2"})
    assert nv is not None and "NVFP4" in nv["label"]
    assert fp is not None and "FP8" in fp["label"] and "NVFP4" not in fp["label"]
    assert plain is not None and "NVFP4" not in plain["label"]


def test_nano_9b_v2_fill_prefers_nemotron_json():
    """Nemotron Nano-9B-v2 default tool parser is nemotron_json (not qwen3_coder)."""
    cfg = ac._empty_config("nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    rationale: list[str] = []
    ac._fill_from_config_detection(
        cfg,
        {
            "family": "nemotron",
            "quant_flag": "",
            "architectures": ["NemotronHForCausalLM"],
            "model_type": "nemotron_h",
        },
        rationale,
    )
    assert cfg["tool_call_parser"] == "nemotron_json"
    assert cfg.get("enable_auto_tool_choice") is True
    # Nano-v2 cards usually omit reasoning_parser; do not force nemotron_v3.
    assert cfg.get("reasoning_parser") in ("", None)


def test_nano_9b_v2_nvfp4_fill_prefers_nemotron_json():
    cfg = ac._empty_config("nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4")
    rationale: list[str] = []
    ac._fill_from_config_detection(
        cfg,
        {"family": "nemotron", "quant_flag": "modelopt_fp4", "architectures": [], "model_type": ""},
        rationale,
    )
    assert cfg["tool_call_parser"] == "nemotron_json"


def test_lightning_nemotron_keeps_qwen3_coder():
    """Lightning / Nemotron-3.x path still defaults to qwen3_coder."""
    cfg = ac._empty_config("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
    rationale: list[str] = []
    ac._fill_from_config_detection(
        cfg,
        {
            "family": "nemotron",
            "quant_flag": "modelopt_mixed",
            "architectures": [],
            "model_type": "nemotron_h",
        },
        rationale,
    )
    assert cfg["tool_call_parser"] == "qwen3_coder"
    assert cfg["reasoning_parser"] == "nemotron_v3"


def test_harvest_nemotron_json_from_tool_focused_card_recipe():
    """When winning recipe lacks tools, harvest nemotron_json from a tool-focused alt."""
    from app.services.autoconfig import ServeCandidate

    cfg = ac._empty_config("nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    # Winning bare recipe (no tool parser)
    bare = ServeCandidate(
        raw="vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 --trust-remote-code",
        env=[],
        model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        args=["--trust-remote-code"],
        config={"trust_remote_code": True, "docker_env": []},
        score=80.0,
        section="vLLM Run",
        reasons=[],
    )
    tool = ServeCandidate(
        raw=(
            "vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 "
            "--enable-auto-tool-choice "
            '--tool-parser-plugin "NVIDIA-Nemotron-Nano-9B-v2/nemotron_toolcall_parser_no_streaming.py" '
            '--tool-call-parser "nemotron_json"'
        ),
        env=[],
        model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        args=[
            "--enable-auto-tool-choice",
            "--tool-parser-plugin",
            "NVIDIA-Nemotron-Nano-9B-v2/nemotron_toolcall_parser_no_streaming.py",
            "--tool-call-parser",
            "nemotron_json",
        ],
        config={
            "enable_auto_tool_choice": True,
            "tool_call_parser": "nemotron_json",
            "extra_flags": (
                "--tool-parser-plugin "
                "NVIDIA-Nemotron-Nano-9B-v2/nemotron_toolcall_parser_no_streaming.py"
            ),
            "docker_env": [],
        },
        score=40.0,
        section="Using Tool-Calling with a vLLM Server",
        reasons=[],
    )
    ac._apply_card_candidate(cfg, bare)
    rationale: list[str] = []
    ac._harvest_tool_flags_from_candidates(cfg, [bare, tool], rationale)
    assert cfg["tool_call_parser"] == "nemotron_json"
    assert cfg.get("enable_auto_tool_choice") is True
    assert "--tool-parser-plugin" in (cfg.get("extra_flags") or "")
    assert any("nemotron_json" in r for r in rationale)


def test_card_prose_hints_nemotron_json():
    prose = ac._card_prose_hints(
        "Start with --enable-auto-tool-choice and --tool-call-parser nemotron_json for tools."
    )
    assert prose.get("tool_call_parser") == "nemotron_json"
    assert prose.get("enable_auto_tool_choice") is True


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


def test_analyze_config_qwen38_conditional_generation_is_vl():
    d = ac.analyze_config(
        {"architectures": ["Qwen3_8ForConditionalGeneration"], "model_type": "qwen3_8"},
        "Qwen/Qwen3.8-27B",
    )
    assert d["is_vl"] is True


def test_vl_keeps_vision_by_default():
    cfg = {"extra_flags": "--max-num-batched-tokens 8192 --language-model-only", "moe_backend": ""}
    warnings: list[str] = []
    rationale: list[str] = []
    ac._apply_vl_spark_defaults(cfg, {"is_vl": True}, warnings, rationale)
    assert "--language-model-only" not in (cfg.get("extra_flags") or "")
    assert "--limit-mm-per-prompt" in (cfg.get("extra_flags") or "")
    assert any("vision" in r.lower() for r in rationale)


def test_vl_legacy_mode_arg_still_keeps_vision():
    cfg = {"extra_flags": "--language-model-only", "moe_backend": ""}
    warnings: list[str] = []
    rationale: list[str] = []
    ac._apply_vl_spark_defaults(
        cfg, {"is_vl": True}, warnings, rationale, mode="lab_safe"
    )
    assert "--language-model-only" not in (cfg.get("extra_flags") or "")
    assert "--limit-mm-per-prompt" in (cfg.get("extra_flags") or "")
    assert not any("Lab Safe serves language-model-only" in w for w in warnings)


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


def test_blob_weights_not_overridden_by_expert_floor(monkeypatch):
    """Hub blob sum (~20 GiB NVFP4) must win over n_routed_experts>=64 floor (400 GiB).

    Lightning 30B-A3B has 128 experts but only ~20 GiB of NVFP4 weights. The blunt
    expert floor was max()'d onto measured blobs and blocked single-Spark serve.
    """
    blob_bytes = int(20.08 * (1024**3))
    api = json.dumps(
        {
            "siblings": [
                {"rfilename": "model-00001-of-00002.safetensors", "size": blob_bytes // 2},
                {"rfilename": "model-00002-of-00002.safetensors", "size": blob_bytes - blob_bytes // 2},
                {"rfilename": "config.json", "size": 1000},
            ]
        }
    )

    def fake_http(url, timeout=20.0):
        if "blobs=true" in url or "api/models" in url:
            return api, None
        return None, "skip"

    monkeypatch.setattr(ac, "_http_get", fake_http)
    cfg = {
        "hidden_size": 2688,
        "num_hidden_layers": 52,
        "n_routed_experts": 128,
        "moe_intermediate_size": 1856,
    }
    w = ac.estimate_weights_gib(
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", cfg
    )
    assert w is not None
    assert 15.0 <= w <= 30.0, f"expected ~20 GiB from blobs, got {w}"
    # Floor alone would still be high for empty-blob refuse paths, but must not apply here.
    floor = ac._weight_floor_gib(
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", cfg
    )
    assert floor is None or floor > w


def test_weight_floor_not_for_compact_moe_in_name():
    """30B-A3B / similar compact MoE ids must not get the 400 GiB expert refuse floor."""
    cfg = {"n_routed_experts": 128}
    assert (
        ac._weight_floor_gib(
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", cfg
        )
        is None
    )
    assert ac._weight_floor_gib("org/Huge-MoE-NoSize", cfg) is not None


def test_expand_card_exports_resolves_dspark_var():
    extra = (
        "--speculative_config.num_speculative_tokens 3 "
        "--speculative_config.model $DSPARK_CKPT --mamba-backend flashinfer"
    )
    readme = (
        "export MODEL_CKPT=nvidia/Foo\n"
        "export DSPARK_CKPT=nvidia/Foo-DSpark\n"
        "vllm serve --model $MODEL_CKPT\n"
    )
    out = ac._expand_card_exports(extra, readme)
    assert "$" not in out
    assert "--speculative_config.model nvidia/Foo-DSpark" in out
    assert "--mamba-backend flashinfer" in out


def test_ensure_dspark_after_scrub_without_method_token():
    """Live NVIDIA Spark recipes often omit method=dspark; only $DSPARK_CKPT carries the name.

    After scrubbing the unexpanded var, 'dspark' disappears from extra_flags and draft
    re-resolve must still run from the card export.
    """
    warnings: list[str] = []
    scrubbed = ac._scrub_unexpanded_shell_vars(
        "--speculative_config.num_speculative_tokens 3 "
        "--speculative_config.model $DSPARK_CKPT --mamba-backend flashinfer",
        warnings,
    )
    assert "dspark" not in scrubbed.lower()
    cfg = {"image": "vllm/vllm-openai:v0.27.1", "extra_flags": scrubbed}
    rationale: list[str] = []
    readme = "export DSPARK_CKPT=nvidia/Lightning-DSpark\n"
    # Expand-before-scrub path (recommend): vars filled first.
    expanded = ac._expand_card_exports(
        "--speculative_config.num_speculative_tokens 3 "
        "--speculative_config.model $DSPARK_CKPT --mamba-backend flashinfer",
        readme,
    )
    cfg2 = {"image": "vllm/vllm-openai:v0.27.1", "extra_flags": expanded}
    ac._ensure_dspark_draft_or_strip(cfg2, readme, warnings, rationale)
    assert "--speculative_config.model nvidia/Lightning-DSpark" in cfg2["extra_flags"]
    # Recover path when expand was skipped but speculative_config remnants remain.
    ac._ensure_dspark_draft_or_strip(cfg, readme, warnings, rationale)
    assert "--speculative_config.model nvidia/Lightning-DSpark" in cfg["extra_flags"]


def test_config_quant_overrides_ampere_prose_modelopt_fp4():
    """Card Ampere recipe mentions modelopt_fp4; config MIXED_PRECISION must win."""
    base = {"quantization": "modelopt_fp4", "model": "nvidia/Nemotron-X-NVFP4"}
    detected = {
        "quant_flag": "modelopt_mixed",
        "quant_algo": "MIXED_PRECISION",
        "has_modelopt_layers": True,
        "family": "nemotron",
    }
    rationale: list[str] = []
    ac._fill_from_config_detection(base, detected, rationale)
    assert base["quantization"] == "modelopt_mixed"


# ─── GitHub cookbook fetch when card is recipe-poor ──────────────────────────


def test_find_cookbook_urls_prefers_vllm_skips_other_frameworks():
    readme = (FIX / "card_recipe_poor.md").read_text()
    urls = ac.find_cookbook_urls(readme)
    assert urls, "expected at least one vLLM cookbook URL"
    joined = " ".join(urls).lower()
    assert "vllm_cookbook" in joined or "spark_vllm" in joined
    assert "trtllm" not in joined
    assert "sglang" not in joined


def test_github_blob_to_raw_url():
    blob = (
        "https://github.com/NVIDIA-NeMo/Nemotron/blob/main/"
        "usage-cookbook/Example/vllm_cookbook.ipynb"
    )
    raw = ac.github_blob_to_raw_url(blob)
    assert raw == (
        "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/"
        "usage-cookbook/Example/vllm_cookbook.ipynb"
    )
    # already-raw stays stable
    assert ac.github_blob_to_raw_url(raw) == raw


def test_vendor_doc_urls_follow_unsloth_guide():
    readme = (
        "Read our [How to Run Qwen3.8-27B Guide!]"
        "(https://unsloth.ai/docs/models/qwen3.8)\n"
        "Also https://evil.example/steal\n"
    )
    urls = ac.find_vendor_doc_urls(readme)
    assert urls == ["https://unsloth.ai/docs/models/qwen3.8"]
    assert (
        ac.vendor_doc_to_fetch_url(urls[0])
        == "https://unsloth.ai/docs/models/qwen3.8.md"
    )
    assert ac.vendor_doc_to_fetch_url("https://evil.example/docs/x") is None


def test_recommend_follows_unsloth_vendor_doc(monkeypatch):
    """HF card has no vllm serve — follow the Unsloth guide linked on the card."""
    readme = (
        "# Qwen3.8-27B NVFP4\n\n"
        "See the [run guide](https://unsloth.ai/docs/models/qwen3.8).\n"
    )
    guide = (
        "#### **vLLM:**\n\n"
        "```shell\n"
        "vllm serve unsloth/Qwen3.8-27B-NVFP4 "
        "--kv-cache-dtype fp8 --reasoning-parser qwen3\n"
        "```\n"
    )
    hf_config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "vision_config": {"depth": 27},
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "max_position_embeddings": 262144,
        },
        "quantization_config": {"quant_method": "compressed-tensors"},
    }

    def fake_fetch(model_id: str, timeout: float = 20.0):
        return {
            "model_id": model_id,
            "readme": readme,
            "config": hf_config,
            "api": None,
            "card_url": f"https://huggingface.co/{model_id}",
            "errors": [],
            "fetched": [],
        }

    def fake_cookbook(url: str, **kwargs):
        if "unsloth.ai" in url:
            return guide, None
        return None, f"unmocked {url}"

    monkeypatch.setattr(ac, "fetch_hf_card", fake_fetch)
    monkeypatch.setattr(ac, "fetch_cookbook_text", fake_cookbook)
    monkeypatch.setattr(ac, "_http_get", lambda *a, **k: (None, "offline"))
    monkeypatch.setattr(
        ac,
        "_cluster_topology",
        lambda: _one_node_topo(ram_gib=121.7, gpu_sku="NVIDIA GB10"),
    )
    monkeypatch.setattr(
        ac,
        "load_local_fallback",
        lambda model_id: {"config": None, "readme": None, "notes": []},
    )

    rec = ac.recommend(
        "unsloth/Qwen3.8-27B-NVFP4",
        mode="workflow_max",
        fetch_remote=True,
    )
    kinds = [s.get("kind") for s in rec.get("sources") or []]
    assert "vendor_doc" in kinds or any(
        "unsloth.ai" in str(s.get("ref") or "") for s in rec.get("sources") or []
    )
    blob = " ".join(rec.get("rationale") or [])
    assert "unsloth.ai" in blob.lower() or "vendor" in blob.lower()
    assert rec["serve_blocked"] is False
    assert "--language-model-only" not in (rec["config"].get("extra_flags") or "")


def test_notebook_source_text_extracts_vllm_serve():
    nb = (FIX / "cookbook_vllm_body.ipynb").read_text()
    text = ac.notebook_source_text(nb)
    assert "vllm serve" in text.lower()
    assert "modelopt" in text
    # plain markdown is returned unchanged
    md = "# hi\nvllm serve x\n"
    assert ac.notebook_source_text(md) == md


def test_candidates_recipe_poor_empty_and_bare():
    assert ac.candidates_recipe_poor([]) is True
    bare = ac.ServeCandidate(
        raw="vllm serve org/model",
        args=[],
        config=ac._empty_config("org/model"),
        section="usage",
        score=1.0,
        reasons=[],
    )
    assert ac.candidates_recipe_poor([bare]) is True
    rich_cfg = ac._empty_config("org/model")
    rich_cfg["quantization"] = "modelopt"
    rich_cfg["moe_backend"] = "marlin"
    rich = ac.ServeCandidate(
        raw="vllm serve org/model --quantization modelopt --moe-backend marlin",
        args=["--quantization", "modelopt", "--moe-backend", "marlin"],
        config=rich_cfg,
        section="DGX Spark",
        score=80.0,
        reasons=["quant"],
    )
    assert ac.candidates_recipe_poor([rich]) is False
    yarn_cfg = ac._empty_config("Qwen/Qwen3.8-27B")
    yarn_cfg["docker_env"] = ["VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"]
    yarn_cfg["extra_flags"] = "--hf-overrides '{...yarn...}'"
    yarn_cfg["max_model_len"] = 1000000
    yarn = ac.ServeCandidate(
        raw="VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ... --hf-overrides yarn --max-model-len 1000000",
        args=["--max-model-len", "1000000"],
        config=yarn_cfg,
        section="Best Practices",
        score=-82.0,
        reasons=["YaRN / extreme rope override demo — not default", "placeholder command"],
    )
    assert ac.candidates_recipe_poor([yarn]) is True


def test_recommend_fetches_cookbook_when_card_recipe_poor(monkeypatch):
    """README has no vllm serve; linked cookbook body supplies the recipe offline."""
    readme = (FIX / "card_recipe_poor.md").read_text()
    cookbook_md = (FIX / "cookbook_vllm_body.md").read_text()
    cookbook_nb = (FIX / "cookbook_vllm_body.ipynb").read_text()
    hf_config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
        "num_hidden_layers": 4,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 2048,
        "vocab_size": 32000,
        "max_position_embeddings": 65536,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
    }

    def fake_fetch(model_id: str, timeout: float = 20.0):
        return {
            "model_id": model_id,
            "readme": readme,
            "config": hf_config,
            "api": {"tags": ["nvfp4"]},
            "card_url": f"https://huggingface.co/{model_id}",
            "errors": [],
            "fetched": [f"https://huggingface.co/{model_id}/raw/main/README.md"],
        }

    def fake_cookbook(url: str, *, timeout: float = 12.0, max_bytes: int = 1_500_000):
        # Mirror real fetch_cookbook_text: flatten notebooks to cell source text.
        u = url.lower()
        if u.endswith(".ipynb") or "vllm_cookbook" in u:
            return ac.notebook_source_text(cookbook_nb), None
        if u.endswith(".md") or "spark_vllm" in u:
            return cookbook_md, None
        return None, f"unmocked cookbook URL: {url}"

    monkeypatch.setattr(ac, "fetch_hf_card", fake_fetch)
    monkeypatch.setattr(ac, "fetch_cookbook_text", fake_cookbook)
    monkeypatch.setattr(ac, "_family_overlay", lambda *a, **k: None)
    monkeypatch.setattr(ac, "estimate_weights_gib", lambda *a, **k: 18.0)
    monkeypatch.setattr(
        ac,
        "_cluster_topology",
        lambda: {
            "nodes": 1,
            "node_list": [{"name": "spark-1", "ram_gib": 121.7, "online": True, "local": True}],
            "head": {"name": "spark-1", "ram_gib": 121.7, "online": True, "local": True},
            "workers": [],
            "fabric_ok": False,
            "available": True,
        },
    )
    monkeypatch.setattr(
        ac,
        "load_local_fallback",
        lambda model_id: {"config": None, "readme": None, "notes": []},
    )

    rec = ac.recommend("example/Cookbook-Model-NVFP4", mode="lab_safe", fetch_remote=True)
    assert rec["from_website"] is True
    cfg = rec["config"]
    assert cfg.get("quantization") in ("modelopt", "modelopt_fp4", "modelopt_mixed") or (
        cfg.get("quantization") or ""
    ).startswith("modelopt")
    assert cfg.get("kv_cache_dtype") == "fp8" or cfg.get("reasoning_parser")
    # Cookbook must appear in sources
    kinds = {s.get("kind") for s in (rec.get("sources") or [])}
    refs = " ".join(s.get("ref") or "" for s in (rec.get("sources") or []))
    assert "github_cookbook" in kinds, rec.get("sources")
    assert "cookbook" in refs.lower() or "vllm" in refs.lower()
    # Parsed recipes must surface for UI
    assert rec.get("card_recipes"), "expected cookbook-derived card_recipes"
    assert any(
        "modelopt" in (c.get("raw") or "").lower()
        or (c.get("config") or {}).get("quantization")
        for c in rec["card_recipes"]
    )


def test_cookbook_fetch_failure_does_not_brick_recommend(monkeypatch):
    """Timeout/offline cookbook must not raise; recommend still returns."""
    readme = (FIX / "card_recipe_poor.md").read_text()
    hf_config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
        "num_hidden_layers": 2,
        "hidden_size": 512,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "intermediate_size": 1024,
        "vocab_size": 32000,
        "max_position_embeddings": 8192,
    }

    monkeypatch.setattr(
        ac,
        "fetch_hf_card",
        lambda m, timeout=20.0: {
            "model_id": m,
            "readme": readme,
            "config": hf_config,
            "api": {},
            "card_url": f"https://huggingface.co/{m}",
            "errors": [],
            "fetched": [],
        },
    )
    monkeypatch.setattr(
        ac,
        "fetch_cookbook_text",
        lambda url, **kw: (None, "TimeoutError: forced"),
    )
    monkeypatch.setattr(ac, "_family_overlay", lambda *a, **k: None)
    monkeypatch.setattr(ac, "estimate_weights_gib", lambda *a, **k: 8.0)
    monkeypatch.setattr(
        ac,
        "_cluster_topology",
        lambda: {
            "nodes": 1,
            "node_list": [{"name": "spark-1", "ram_gib": 121.7, "online": True, "local": True}],
            "head": {"name": "spark-1", "ram_gib": 121.7, "online": True, "local": True},
            "workers": [],
            "fabric_ok": False,
            "available": True,
        },
    )
    monkeypatch.setattr(
        ac,
        "load_local_fallback",
        lambda model_id: {"config": None, "readme": None, "notes": []},
    )

    rec = ac.recommend("example/Offline-Cookbook", mode="lab_safe", fetch_remote=True)
    assert "config" in rec
    assert rec["config"].get("model") == "example/Offline-Cookbook"
    # Soft warning, not a hard failure
    assert any("cookbook" in (w or "").lower() for w in (rec.get("warnings") or [])) or any(
        "cookbook" in (r or "").lower() for r in (rec.get("rationale") or [])
    )


def test_family_overlay_not_overridden_by_cookbook(monkeypatch):
    """Even with cookbook recipes available, family overlay still owns config."""
    readme = (FIX / "card_recipe_poor.md").read_text()
    cookbook_md = (FIX / "cookbook_vllm_body.md").read_text()

    monkeypatch.setattr(
        ac,
        "fetch_hf_card",
        lambda m, timeout=20.0: {
            "model_id": m,
            "readme": readme,
            "config": {"architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3"},
            "api": {},
            "card_url": f"https://huggingface.co/{m}",
            "errors": [],
            "fetched": [],
        },
    )
    called = {"n": 0}

    def fake_cookbook(url: str, **kw):
        called["n"] += 1
        return cookbook_md, None

    monkeypatch.setattr(ac, "fetch_cookbook_text", fake_cookbook)
    # Real DSv4 overlay path
    monkeypatch.setattr(ac, "estimate_weights_gib", lambda *a, **k: 40.0)
    monkeypatch.setattr(ac, "_cluster_topology", _two_spark_topo)
    monkeypatch.setattr(
        ac,
        "load_local_fallback",
        lambda model_id: {"config": None, "readme": None, "notes": []},
    )

    rec = ac.recommend(DSV4, mode="workflow_max", fetch_remote=True)
    # Vendor docs may be fetched for provenance, but overlay still owns flags.
    assert any(s.get("kind") == "family_overlay" for s in (rec.get("sources") or []))
    assert (rec["config"].get("image") or "").startswith("ghcr.io/anemll/dspark-vllm-gx10")
    assert rec["config"].get("kv_cache_dtype") == "nvfp4_ds_mla"


# ─── Any-hardware placement (SKU arch + unprobed RAM + unknown dense Qwen) ───

QWEN38_DIR = Path(__file__).resolve().parent / "corpus" / "example__Qwen3.8-27B"
QWEN38 = "Qwen/Qwen3.8-27B"


def _one_node_topo(*, ram_gib=None, gpu_sku=None, available=True):
    node = {"id": "n0", "local": True, "online": True}
    if ram_gib is not None:
        node["ram_gib"] = ram_gib
    if gpu_sku is not None:
        node["gpu_sku"] = gpu_sku
    return {
        "nodes": 1,
        "node_list": [node],
        "head": node,
        "workers": [],
        "fabric_ok": False,
        "available": available,
    }


def _env_blob(cfg_or_env) -> str:
    if isinstance(cfg_or_env, dict):
        env = cfg_or_env.get("docker_env") or []
    else:
        env = cfg_or_env or []
    if isinstance(env, dict):
        return " ".join(f"{k}={v}" for k, v in env.items())
    return " ".join(str(e) for e in env)


def test_gpu_arch_env_unknown_sku_omits_gb10_pin():
    """Unknown / empty / RTX must not inherit GB10 CUTE_DSL / 12.1a compile flags."""
    empty = ac._gpu_arch_env([])
    rtx = ac._gpu_arch_env([{"gpu_sku": "NVIDIA GeForce RTX 4090"}])
    unknown = ac._gpu_arch_env([{"gpu_sku": "unknown"}])
    missing = ac._gpu_arch_env([{"name": "local"}])
    for env in (empty, rtx, unknown, missing):
        blob = _env_blob(env)
        assert "sm_121a" not in blob
        assert "12.1a" not in blob
        assert "CUTE_DSL_ARCH" not in blob
        assert "TORCH_CUDA_ARCH_LIST" not in blob
        assert env == {}


def test_gpu_arch_env_only_when_sku_matches():
    gb10 = ac._gpu_arch_env([{"gpu_sku": "NVIDIA GB10"}])
    assert gb10.get("CUTE_DSL_ARCH") == "sm_121a"
    assert gb10.get("TORCH_CUDA_ARCH_LIST") == "12.1a"
    spark = ac._gpu_arch_env([{"gpu_sku": "NVIDIA GB10 [Spark]"}])
    assert spark.get("CUTE_DSL_ARCH") == "sm_121a"
    gb200 = ac._gpu_arch_env([{"gpu_sku": "NVIDIA GB200"}])
    assert gb200.get("CUTE_DSL_ARCH") == "sm_100a"
    assert "sm_121a" not in _env_blob(gb200)


def test_apply_topology_rtx_has_no_gb10_arch():
    cfg = ac._empty_config("org/Model")
    warnings: list[str] = []
    rationale: list[str] = []
    ac._apply_topology(
        cfg,
        overlay=None,
        topology=_one_node_topo(ram_gib=24.0, gpu_sku="NVIDIA GeForce RTX 4090"),
        weights_gib=8.0,
        mode="lab_safe",
        warnings=warnings,
        rationale=rationale,
    )
    blob = _env_blob(cfg)
    assert "sm_121a" not in blob
    assert "12.1a" not in blob


def test_apply_topology_gb10_sets_arch():
    cfg = ac._empty_config("org/Model")
    ac._apply_topology(
        cfg,
        overlay=None,
        topology=_one_node_topo(ram_gib=121.7, gpu_sku="NVIDIA GB10"),
        weights_gib=20.0,
        mode="workflow_max",
        warnings=[],
        rationale=[],
    )
    blob = _env_blob(cfg)
    assert "CUTE_DSL_ARCH=sm_121a" in blob


def test_missing_ram_probes_collect_hardware(monkeypatch):
    monkeypatch.setattr(
        "app.services.metadata.collect_hardware",
        lambda: {"ram_gib": 32.0, "gpu_sku": "NVIDIA GeForce RTX 4090", "hostname": "devbox"},
    )
    assert ac._node_ram_gib(None) == 32.0
    assert ac._node_ram_gib({"local": True, "online": True}) == 32.0
    assert ac._node_ram_gib({"ram_gib": 121.7}) == 121.7
    p = ac.plan_placement(
        20.0, _one_node_topo(), mode="lab_safe", overlay=None
    )
    assert p["node_ram_gib"] == 32.0
    # 20 GiB + 15 reserve cannot load on 32 GiB even at util=0.85
    assert p["fits"] is False
    ok, msg = ac.check_serve_loadability(
        mode="lab_safe",
        weights_gib=20.0,
        node_ram_gib=p["node_ram_gib"],
        nodes_used=1,
        util=0.4,
    )
    assert ok is False
    assert msg


def test_unprobed_ram_is_not_gb10_uma(monkeypatch):
    """Probe failure must not fall back to 121.7 GiB Spark UMA."""
    def boom():
        raise OSError("no /proc/meminfo")

    monkeypatch.setattr("app.services.metadata.collect_hardware", boom)
    ram = ac._node_ram_gib({})
    assert ram != 121.7
    assert ram <= 32.0
    p = ac.plan_placement(20.0, _one_node_topo(), mode="lab_safe", overlay=None)
    assert p["node_ram_gib"] != 121.7
    assert p["fits"] is False


def test_unavailable_topology_probes_local_hardware(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("cluster down")

    monkeypatch.setattr("app.services.cluster.collect_cluster", boom)
    monkeypatch.setattr(
        "app.services.metadata.collect_hardware",
        lambda: {
            "ram_gib": 64.0,
            "memory_capacity_gib": 64.0,
            "gpu_sku": "NVIDIA GeForce RTX 4090",
            "hostname": "devbox",
        },
    )
    t = ac._cluster_topology()
    assert t["available"] is False
    assert t["nodes"] == 1
    head = t.get("head") or {}
    assert head.get("ram_gib") == 64.0
    assert head.get("gpu_sku") == "NVIDIA GeForce RTX 4090"
    p = ac.plan_placement(20.0, t, mode="lab_safe", overlay=None)
    assert p["node_ram_gib"] == 64.0


_MODELOPT_MIXED_27B = {
    "architectures": ["Qwen3ForCausalLM"],
    "dtype": "bfloat16",
    "model_type": "qwen3",
    "text_config": {
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "max_position_embeddings": 262144,
        "model_type": "qwen3",
        "num_attention_heads": 24,
        "num_hidden_layers": 64,
        "num_key_value_heads": 4,
        "vocab_size": 151936,
    },
    "quantization_config": {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
        "config_groups": {
            "group_0": {
                "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
                "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
            }
        },
        "quantized_layers": {
            "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
            "model.layers.0.mlp.down_proj": {"quant_algo": "W4A16_NVFP4"},
        },
    },
}


def test_estimate_weights_honors_modelopt_algo_without_nvfp4_in_id(monkeypatch):
    """Offline heuristic must use quant_algo / quantized_layers, not the repo id."""
    monkeypatch.setattr(ac, "_http_get", lambda *a, **k: (None, "offline"))
    cfg = _MODELOPT_MIXED_27B
    w = ac.estimate_weights_gib(QWEN38, cfg)
    assert w is not None
    # 4-bit 27B-class dense is well under bf16 (~37 GiB from the 12 L H^2 formula).
    text = cfg["text_config"]
    bf16 = 12 * text["num_hidden_layers"] * text["hidden_size"] * text["hidden_size"] * 2.0 / (1024**3)
    assert 5.0 <= w <= 25.0
    assert w < bf16 * 0.6


def test_recommend_skips_yarn_demo_on_live_shaped_qwen38(monkeypatch):
    """Official card's only parseable serve line is the optional 1M YaRN demo — do not apply it."""
    readme = """# Qwen3.8-27B

## Best Practices

```shell
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ... --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' --max-model-len 1000000
```
"""
    hf_config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "language_model_only": False,
        "vision_config": {"depth": 27, "hidden_size": 1152},
        "text_config": {
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "max_position_embeddings": 262144,
            "num_attention_heads": 24,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "vocab_size": 248320,
        },
    }
    monkeypatch.setattr(
        ac,
        "fetch_hf_card",
        lambda model_id, timeout=20.0: {
            "model_id": model_id,
            "readme": readme,
            "config": hf_config,
            "api": None,
            "card_url": f"https://huggingface.co/{model_id}",
            "errors": [],
            "fetched": [f"fixture://{model_id}"],
        },
    )
    monkeypatch.setattr(ac, "_http_get", lambda *a, **k: (None, "offline"))
    monkeypatch.setattr(ac, "_http_get_raw", lambda *a, **k: (None, "offline"))
    monkeypatch.setattr(ac, "fetch_cookbook_text", lambda *a, **k: (None, "offline"))
    monkeypatch.setattr(
        ac,
        "_cluster_topology",
        lambda: _one_node_topo(ram_gib=121.7, gpu_sku="NVIDIA GB10"),
    )
    monkeypatch.setattr(
        ac,
        "load_local_fallback",
        lambda model_id: {"config": None, "readme": None, "notes": []},
    )

    rec = ac.recommend(QWEN38, mode="workflow_max", fetch_remote=True)
    cfg = rec["config"]
    blob = json.dumps(cfg) + " ".join(str(x) for x in (cfg.get("docker_env") or []))
    assert "yarn" not in blob.lower()
    assert "VLLM_ALLOW_LONG_MAX_MODEL_LEN" not in blob
    assert "1000000" not in blob
    assert rec["serve_blocked"] is False
    assert cfg.get("reasoning_parser") == "qwen3"
    assert "--language-model-only" not in (cfg.get("extra_flags") or "")
    assert "--limit-mm-per-prompt" in (cfg.get("extra_flags") or "")
    assert any("Skipped weak card recipe" in r for r in rec.get("rationale") or [])


def test_recommend_unknown_dense_qwen38_offline(monkeypatch):
    """Brand-new Qwen3.x id: card+config only, no overlay, single-node TP=1, fits."""
    readme = (
        "# Qwen3.8-27B\n\n"
        "```sh\n"
        "vllm serve Qwen/Qwen3.8-27B --port 8000 --quantization modelopt "
        "--max-model-len 262144 --reasoning-parser qwen3\n"
        "```\n"
    )
    hf_config = _MODELOPT_MIXED_27B

    monkeypatch.setattr(
        ac,
        "fetch_hf_card",
        lambda model_id, timeout=20.0: {
            "model_id": model_id,
            "readme": readme,
            "config": hf_config,
            "api": None,
            "card_url": f"https://huggingface.co/{model_id}",
            "errors": [],
            "fetched": [f"fixture://{model_id}"],
        },
    )
    monkeypatch.setattr(ac, "_http_get", lambda *a, **k: (None, "offline"))
    monkeypatch.setattr(
        ac,
        "_cluster_topology",
        lambda: _one_node_topo(ram_gib=121.7, gpu_sku="NVIDIA GeForce RTX 4090"),
    )
    monkeypatch.setattr(
        ac,
        "load_local_fallback",
        lambda model_id: {"config": None, "readme": None, "notes": []},
    )

    rec = ac.recommend(QWEN38, mode="workflow_max", fetch_remote=True)
    cfg = rec["config"]
    assert ac._family_overlay(QWEN38, rec.get("detected") or {}) is None
    assert rec.get("topology", {}).get("overlay") is None
    assert rec["serve_blocked"] is False
    assert (rec.get("topology") or {}).get("tensor_parallel_size") == 1
    assert cfg.get("tensor_parallel_size") in (None, 1)
    assert cfg.get("quantization") == "modelopt_mixed"
    assert cfg.get("reasoning_parser") == "qwen3"
    assert cfg.get("tool_call_parser") == "qwen3_coder"
    assert (cfg.get("moe_backend") or "") != "flashinfer_b12x"
    blob = json.dumps(cfg) + " ".join(str(x) for x in (cfg.get("docker_env") or []))
    assert "$" not in blob
    assert "flashinfer_b12x" not in blob
    # RTX / non-GB10 sku: recommend must not inject Spark compile pins.
    env = _env_blob(cfg)
    assert "sm_121a" not in env
    assert "12.1a" not in env

    rec_safe = ac.recommend(QWEN38, mode="lab_safe", fetch_remote=True)
    assert rec_safe["serve_blocked"] is False


def test_serve_model_refuses_when_recommend_would_block(monkeypatch):
    """Start must raise SERVE BLOCKED when the shared fit gate fails."""
    from app.services import serve as sv

    topo = _one_node_topo(ram_gib=32.0, gpu_sku="NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(ac, "_cluster_topology", lambda: topo)
    monkeypatch.setattr(ac, "estimate_weights_gib", lambda *a, **k: 20.0)
    monkeypatch.setattr(
        ac, "load_local_fallback", lambda m: {"config": {}, "readme": None, "notes": []}
    )
    rec = ac.recommend(QWEN38, mode="lab_safe", fetch_remote=False)
    assert rec["serve_blocked"] is True
    with pytest.raises(RuntimeError, match="SERVE BLOCKED"):
        sv.serve_model(model=QWEN38, mode="lab_safe")


def test_serve_model_does_not_refuse_when_recommend_fits(monkeypatch):
    """Start must not raise SERVE BLOCKED when recommend.serve_blocked is false."""
    from app.services import serve as sv

    topo = _one_node_topo(ram_gib=121.7, gpu_sku="NVIDIA GB10")
    monkeypatch.setattr(ac, "_cluster_topology", lambda: topo)
    monkeypatch.setattr(ac, "estimate_weights_gib", lambda *a, **k: 20.0)
    monkeypatch.setattr(
        ac, "load_local_fallback", lambda m: {"config": {}, "readme": None, "notes": []}
    )
    rec = ac.recommend(QWEN38, mode="workflow_max", fetch_remote=False)
    assert rec["serve_blocked"] is False

    def past_gate(*a, **k):
        raise RuntimeError("PAST_FIT_GATE")

    monkeypatch.setattr(sv, "_ensure_image_present", past_gate)
    with pytest.raises(RuntimeError, match="PAST_FIT_GATE"):
        sv.serve_model(model=QWEN38, mode="workflow_max")


def test_config_modelopt_mixed_overrides_card_compressed_tensors():
    """HF config.json modelopt_mixed must win over a card --quantization compressed-tensors."""
    base = {"quantization": "compressed-tensors", "model": "Qwen/Qwen3.8-27B"}
    detected = {
        "quant_flag": "modelopt_mixed",
        "quant_algo": "MIXED_PRECISION",
        "quant_method": "modelopt",
        "has_modelopt_layers": True,
        "family": "qwen",
    }
    rationale: list[str] = []
    ac._fill_from_config_detection(base, detected, rationale)
    assert base["quantization"] == "modelopt_mixed"
    assert any("compressed-tensors" in r and "modelopt_mixed" in r for r in rationale)


def test_marlin_kept_for_pure_nvfp4_qwen_on_027():
    """Pure NVFP4 Qwen may use marlin on ≥0.27; mixed / old images still drop it."""
    pure = {
        "family": "qwen",
        "is_moe": True,
        "has_nvfp4": True,
        "is_mixed_nvfp4_fp8": False,
        "quant_flag": "modelopt_fp4",
    }
    assert ac._marlin_unsafe_for_checkpoint(pure, "vllm/vllm-openai:v0.27.1") is False
    assert ac._marlin_unsafe_for_checkpoint(pure, "vllm/vllm-openai:v0.25.0") is True
    mixed = {
        "family": "qwen",
        "is_moe": True,
        "has_nvfp4": True,
        "is_mixed_nvfp4_fp8": True,
        "quant_flag": "compressed-tensors",
    }
    assert ac._marlin_unsafe_for_checkpoint(mixed, "vllm/vllm-openai:v0.27.1") is True


def test_parse_card_image_skips_nightly_but_flags_floating_only():
    readme = (
        "To serve this checkpoint start docker `vllm/vllm-openai:nightly` "
        "and run vllm serve org/New-Arch-27B"
    )
    assert ac._parse_card_image_requirement(readme) is None
    assert ac._card_has_only_floating_image(readme) is True
    warnings: list[str] = []
    ac._warn_floating_card_image(readme, "vllm/vllm-openai:v0.27.1", warnings)
    blob = " ".join(warnings).lower()
    assert "nightly" in blob
    assert "v0.27.1" in blob


def test_serve_model_refuses_when_fit_gate_probe_raises(monkeypatch):
    """Start must fail closed if the fit-gate probe itself throws."""
    from app.services import serve as sv

    topo = _one_node_topo(ram_gib=121.7, gpu_sku="NVIDIA GB10")
    monkeypatch.setattr(ac, "_cluster_topology", lambda: topo)
    monkeypatch.setattr(ac, "load_local_fallback", lambda m: {"config": {"hidden_size": 1}, "readme": None, "notes": []})

    def boom(*a, **k):
        raise OSError("hub down")

    monkeypatch.setattr(ac, "estimate_weights_gib", boom)
    monkeypatch.setattr(
        sv,
        "_ensure_image_present",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("PAST_FIT_GATE")),
    )
    with pytest.raises(RuntimeError, match="SERVE BLOCKED"):
        sv.serve_model(model=QWEN38, mode="lab_safe")


def test_multinode_weight_estimate_uses_hf_config(monkeypatch):
    """TP>=2 placement must pass local hf_config into estimate_weights_gib, not None."""
    from app.services import serve as sv

    seen: dict = {}

    def fake_est(model, cfg):
        seen.setdefault("cfgs", []).append(cfg)
        return 20.0

    cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "architectures": ["Qwen3ForCausalLM"]}
    topo = {
        "nodes": 2,
        "node_list": [
            {"id": "n0", "local": True, "online": True, "ram_gib": 121.7, "qsfp_ip": "10.0.0.1"},
            {"id": "n1", "local": False, "online": True, "ram_gib": 121.7, "qsfp_ip": "10.0.0.2"},
        ],
        "head": {"id": "n0", "local": True, "online": True, "ram_gib": 121.7, "qsfp_ip": "10.0.0.1"},
        "workers": [{"id": "n1", "local": False, "online": True, "ram_gib": 121.7, "qsfp_ip": "10.0.0.2"}],
        "fabric_ok": True,
        "available": True,
    }
    monkeypatch.setattr(ac, "_cluster_topology", lambda: topo)
    monkeypatch.setattr(ac, "estimate_weights_gib", fake_est)
    monkeypatch.setattr(ac, "load_local_fallback", lambda m: {"config": cfg, "readme": None, "notes": []})
    monkeypatch.setattr(sv, "_ensure_image_present", lambda *a, **k: None)

    def past_launch(*a, **k):
        raise RuntimeError("PAST_MULTINODE_ESTIMATE")

    monkeypatch.setattr(sv, "build_multi_node_launch", past_launch)
    with pytest.raises(RuntimeError, match="PAST_MULTINODE_ESTIMATE"):
        sv.serve_model(model=QWEN38, mode="workflow_max", tensor_parallel_size=2)
    assert seen.get("cfgs"), "estimate_weights_gib was never called"
    assert all(c == cfg for c in seen["cfgs"]), f"hf_config skipped on a call; got {seen['cfgs']!r}"


def test_lab_safe_headroom_abort_only_on_large_uma():
    from app.services import serve as sv

    assert sv._lab_safe_headroom_abort(avail=20.0, ram_total=32.0) is None
    assert sv._lab_safe_headroom_abort(avail=20.0, ram_total=None) is None
    msg = sv._lab_safe_headroom_abort(avail=20.0, ram_total=121.7)
    assert msg is not None and "ABORT" in msg and "60" in msg
    assert sv._lab_safe_headroom_abort(avail=80.0, ram_total=121.7) is None


# ─── Active vendor research (card-silent) + Qwen3.8-27B-NVFP4 ─────────────────

UNSLOTH_QWEN38 = "unsloth/Qwen3.8-27B-NVFP4"

_Q38_VL_CFG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "language_model_only": False,
    "vision_config": {"depth": 27, "hidden_size": 1152},
    "text_config": {
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "max_position_embeddings": 262144,
        "num_attention_heads": 24,
        "num_hidden_layers": 64,
        "num_key_value_heads": 4,
        "vocab_size": 248320,
    },
    "quantization_config": {"quant_method": "compressed-tensors"},
}


def test_family_doc_slugs_publisher_agnostic():
    slugs = ac.family_doc_slugs(
        "acme/Qwen3.8-27B-NVFP4",
        {"family": "qwen", "has_nvfp4": True},
    )
    assert "qwen3.8" in slugs
    assert "nvfp4" in slugs
    slugs_official = ac.family_doc_slugs("Qwen/Qwen3.8-27B", {"family": "qwen"})
    assert "qwen3.8" in slugs_official
    # Architecture-only id: Qwen3.8 ships as qwen3_5 VL classes.
    slugs_arch = ac.family_doc_slugs(
        "acme/mystery-vl-nvfp4",
        {
            "family": "qwen",
            "has_nvfp4": True,
            "is_vl": True,
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
        },
    )
    assert "qwen3.8" in slugs_arch


def test_vendor_candidate_family_mismatch_drops_gemma_for_qwen():
    gemma = ac.ServeCandidate(
        raw="vllm serve unsloth/gemma-4 --reasoning-parser gemma4",
        model="unsloth/gemma-4",
        config={"reasoning_parser": "gemma4"},
    )
    qwen = ac.ServeCandidate(
        raw="vllm serve unsloth/Qwen3.8-27B-NVFP4 --reasoning-parser qwen3",
        model="unsloth/Qwen3.8-27B-NVFP4",
        config={"reasoning_parser": "qwen3"},
    )
    assert ac._vendor_candidate_family_mismatch(gemma, "qwen") is True
    assert ac._vendor_candidate_family_mismatch(qwen, "qwen") is False


def test_discover_recipe_urls_without_card_links():
    found = ac.discover_recipe_urls(
        "acme/Qwen3.8-27B-NVFP4",
        {
            "family": "qwen",
            "has_nvfp4": True,
            "architectures": ["Qwen3_5ForConditionalGeneration"],
        },
        readme="# silent\nNo cookbook or vendor URL on this card.\n",
    )
    refs = [u.url for u in found]
    assert any("unsloth.ai/docs/models/qwen3.8" in r for r in refs)
    assert any(u.origin == "derived" and u.kind == "vendor_doc" for u in found)
    assert any(u.kind == "nvidia_playbook" for u in found)


def _patch_offline_recommend(monkeypatch, *, readme, config, topo_ram=121.7, sku="NVIDIA GB10"):
    def fake_fetch(model_id: str, timeout: float = 20.0):
        return {
            "model_id": model_id,
            "readme": readme,
            "config": config,
            "api": None,
            "card_url": f"https://huggingface.co/{model_id}",
            "errors": [],
            "fetched": [f"fixture://{model_id}"],
        }

    monkeypatch.setattr(ac, "fetch_hf_card", fake_fetch)
    monkeypatch.setattr(ac, "_http_get", lambda *a, **k: (None, "offline"))
    monkeypatch.setattr(
        ac,
        "_cluster_topology",
        lambda: _one_node_topo(ram_gib=topo_ram, gpu_sku=sku),
    )
    monkeypatch.setattr(
        ac,
        "load_local_fallback",
        lambda model_id: {"config": None, "readme": None, "notes": []},
    )
    monkeypatch.setattr(ac, "estimate_weights_gib", lambda *a, **k: 18.0)


def test_recommend_researches_vendor_when_card_has_no_url(monkeypatch):
    """(a) Silent card — derived Unsloth URL still supplies flags + sources."""
    guide = (FIX / "vendor_unsloth_qwen38.md").read_text()
    readme = "# Mystery Qwen\nNo links. No vllm serve.\n"
    hf_config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "quantization_config": {"quant_method": "compressed-tensors"},
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "max_position_embeddings": 262144,
    }
    _patch_offline_recommend(monkeypatch, readme=readme, config=hf_config)
    monkeypatch.setattr(ac, "_family_overlay", lambda *a, **k: None)

    fetched: list[str] = []

    def fake_cookbook(url: str, **kwargs):
        fetched.append(url)
        if "unsloth.ai" in url:
            return guide, None
        return None, f"unmocked {url}"

    monkeypatch.setattr(ac, "fetch_cookbook_text", fake_cookbook)

    rec = ac.recommend("acme/Silent-Qwen-27B-NVFP4", fetch_remote=True)
    kinds = {s.get("kind") for s in rec.get("sources") or []}
    refs = " ".join(s.get("ref") or "" for s in rec.get("sources") or [])
    assert any("unsloth.ai" in u for u in fetched), fetched
    assert "vendor_doc" in kinds, rec.get("sources")
    assert "unsloth" in refs.lower() or "unsloth" in " ".join(rec.get("rationale") or []).lower()
    cfg = rec["config"]
    assert cfg.get("kv_cache_dtype") == "fp8"
    assert cfg.get("reasoning_parser") == "qwen3"
    assert rec["serve_blocked"] is False


def test_recommend_unsloth_qwen38_nvfp4_spark(monkeypatch):
    """(b) Named struggle case on Spark-class RAM."""
    readme = (
        "# Qwen3.8-27B-NVFP4\n\n"
        "```shell\n"
        "vllm serve unsloth/Qwen3.8-27B-NVFP4\n"
        "```\n\n"
        "```shell\n"
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ... "
        "--hf-overrides '{\"text_config\": {\"rope_parameters\": {\"rope_type\": \"yarn\"}}}' "
        "--max-model-len 1000000\n"
        "```\n"
    )
    guide = (FIX / "vendor_unsloth_qwen38.md").read_text()
    _patch_offline_recommend(monkeypatch, readme=readme, config=_Q38_VL_CFG)

    def fake_cookbook(url: str, **kwargs):
        if "unsloth.ai" in url:
            return guide, None
        return None, f"unmocked {url}"

    monkeypatch.setattr(ac, "fetch_cookbook_text", fake_cookbook)

    rec = ac.recommend(UNSLOTH_QWEN38, fetch_remote=True)
    cfg = rec["config"]
    blob = json.dumps(cfg) + " ".join(str(x) for x in (cfg.get("docker_env") or []))
    assert rec["serve_blocked"] is False
    assert cfg.get("kv_cache_dtype") == "fp8"
    assert cfg.get("reasoning_parser") == "qwen3"
    assert cfg.get("tool_call_parser") == "qwen3_coder"
    assert cfg.get("trust_remote_code") is True
    assert "--language-model-only" not in (cfg.get("extra_flags") or "")
    assert "yarn" not in blob.lower()
    assert "1000000" not in blob
    assert "flashinfer_b12x" not in blob
    refs = " ".join(s.get("ref") or "" for s in rec.get("sources") or [])
    kinds = {s.get("kind") for s in rec.get("sources") or []}
    assert "vendor_doc" in kinds, rec.get("sources")
    assert "unsloth" in refs.lower()


def test_recommend_qwen38_twin_same_entry(monkeypatch):
    """(c) Official Qwen twin uses the same recommend() path — parsers + VL, no publisher switch."""
    readme = "# Qwen3.8-27B\n\nNo serve recipe.\n"
    _patch_offline_recommend(monkeypatch, readme=readme, config=_Q38_VL_CFG)
    monkeypatch.setattr(
        ac, "fetch_cookbook_text", lambda *a, **k: (None, "offline")
    )
    rec = ac.recommend("Qwen/Qwen3.8-27B", fetch_remote=True)
    cfg = rec["config"]
    assert rec["serve_blocked"] is False
    assert cfg.get("reasoning_parser") == "qwen3"
    assert cfg.get("tool_call_parser") == "qwen3_coder"
    assert "--language-model-only" not in (cfg.get("extra_flags") or "")
    assert "--limit-mm-per-prompt" in (cfg.get("extra_flags") or "")


def test_recommend_vendor_fetch_error_still_returns(monkeypatch):
    """(d) Extra-source fetch failure must not brick recommend."""
    readme = "# Silent\nNo links.\n"
    _patch_offline_recommend(monkeypatch, readme=readme, config=_Q38_VL_CFG)
    monkeypatch.setattr(
        ac, "fetch_cookbook_text", lambda *a, **k: (None, "timeout")
    )
    rec = ac.recommend(UNSLOTH_QWEN38, fetch_remote=True)
    assert isinstance(rec, dict)
    assert rec.get("config")
    assert rec["config"].get("model") == UNSLOTH_QWEN38
    assert rec["serve_blocked"] is False
    # Overlay / config detection still fills parsers when vendor fetch fails.
    assert rec["config"].get("reasoning_parser") == "qwen3"


def test_recommend_accepts_request_with_no_mode(monkeypatch):
    _patch_offline_recommend(
        monkeypatch,
        readme="# x\n```shell\nvllm serve org/x --reasoning-parser qwen3\n```\n",
        config=_Q38_VL_CFG,
    )
    monkeypatch.setattr(ac, "fetch_cookbook_text", lambda *a, **k: (None, "offline"))
    rec = ac.recommend(UNSLOTH_QWEN38, fetch_remote=True)
    assert rec.get("mode") in (None, "auto")
    assert rec["serve_blocked"] is False
    assert rec["config"].get("util") is not None


