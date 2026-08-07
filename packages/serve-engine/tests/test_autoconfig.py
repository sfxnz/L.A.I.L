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
