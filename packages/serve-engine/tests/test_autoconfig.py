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
    flash = [c for c in cands if (c.config.get("moe_backend") or "") == "flashinfer_b12x"]
    non_flash = [c for c in cands if (c.config.get("moe_backend") or "") != "flashinfer_b12x"]
    assert flash, "fixture must include flashinfer recipe"
    assert non_flash, "fixture must include non-flashinfer recipe"
    # Best overall must not be a flashinfer recipe when checkpoint is mixed
    best = cands[0]
    assert (best.config.get("moe_backend") or "") != "flashinfer_b12x"
    assert max(c.score for c in non_flash) > max(c.score for c in flash)


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


def test_analyze_config_detects_mixed_formats():
    cfg = json.loads((FIX / "config_mixed_compressed_tensors.json").read_text())
    d = ac.analyze_config(cfg, "org/Model-A3B-NVFP4")
    assert d["is_moe"] is True
    assert d["has_nvfp4"] is True
    assert d["has_fp8"] is True
    assert d["is_mixed_nvfp4_fp8"] is True
    assert d["quant_flag"] == "compressed-tensors"


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
    root = Path(__file__).resolve().parents[2]  # local-ai-lab
    serve_tsx = root / "frontend" / "src" / "pages" / "Serve.tsx"
    api_ts = root / "frontend" / "src" / "api.ts"
    assert serve_tsx.is_file() and api_ts.is_file()
    st = serve_tsx.read_text()
    at = api_ts.read_text()
    assert "recommendServe" in at
    assert "/serve/recommend" in at
    assert "recommendServe" in st
    assert "applyConfig" in st
    assert "from_website" in st
    assert "Auto-configure" in st
    # Recipe scoring reasons (e.g. PENALTY for flashinfer) must render in UI
    assert "cr.reasons" in st
    assert "card_recipes" in st


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
