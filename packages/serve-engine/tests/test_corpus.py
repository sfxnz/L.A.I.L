"""Regression corpus: auto-config must reproduce each vendor's published recipe.

Every case is a directory under ``corpus/`` holding a captured HF card and
config.json plus that model's expectations. Cases run offline against the real
``recommend()`` path with the cluster pinned to a single Spark, so a failure
means the engine changed rather than the network.

Add a case with::

    cd packages/serve-engine && PYTHONPATH=. python tests/corpus/_capture.py <hf-model-id>
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import autoconfig as ac
from app.services import serve as sv

CORPUS = Path(__file__).resolve().parent / "corpus"

# Substrings that must never reach a launched command, whatever the card said.
# Each has a failure mode we have actually hit:
#   "$"        — an unexpanded shell var from a card recipe becomes the model name
#   "--model " — the launcher passes the model positionally; a second one can win
FORBIDDEN_ALWAYS = ("$", "--model ")


def _case_dirs() -> list[Path]:
    if not CORPUS.is_dir():
        return []
    return sorted(p for p in CORPUS.iterdir() if (p / "case.json").is_file())


def _one_spark_topo() -> dict:
    node = {
        "id": "spark1",
        "qsfp_ip": "10.100.8.1",
        "qsfp_if": "enp1s0f1np1",
        "ram_gib": 121.7,
        "local": True,
        "ssh_host": "spark1",
    }
    return {
        "nodes": 1,
        "node_list": [node],
        "head": node,
        "workers": [],
        "fabric_ok": False,
        "available": True,
    }


def _load(case_dir: Path) -> dict:
    case = json.loads((case_dir / "case.json").read_text())
    card = case_dir / "card.md"
    cfg = case_dir / "config.json"
    case["_readme"] = card.read_text(encoding="utf-8") if card.is_file() else None
    case["_config"] = json.loads(cfg.read_text()) if cfg.is_file() else None
    return case


def _recommend_offline(monkeypatch, case: dict) -> dict:
    """Run the shipped recommend() against captured fixtures only."""
    model = case["model"]

    def _fake_fetch(model_id: str, timeout: float = 20.0) -> dict:
        return {
            "model_id": model_id,
            "readme": case["_readme"],
            "config": case["_config"],
            "api": None,
            "card_url": f"https://huggingface.co/{model_id}",
            "errors": [],
            "fetched": [f"fixture://{model_id}"],
        }

    monkeypatch.setattr(ac, "fetch_hf_card", _fake_fetch)
    monkeypatch.setattr(ac, "_cluster_topology", _one_spark_topo)
    return ac.recommend(model, mode=case.get("mode", "workflow_max"))


def _final_command(model: str, cfg: dict) -> str:
    """Compose the argv the launcher would actually exec, for end-to-end checks."""
    args = sv._build_vllm_args(
        util=float(cfg.get("util") or 0.4),
        max_model_len=int(cfg.get("max_model_len") or 65536),
        port=8000,
        quantization=cfg.get("quantization") or "",
        kv_cache_dtype=cfg.get("kv_cache_dtype") or "",
        moe_backend=cfg.get("moe_backend") or "",
        trust_remote_code=bool(cfg.get("trust_remote_code")),
        enable_auto_tool_choice=bool(cfg.get("enable_auto_tool_choice")),
        tool_call_parser=cfg.get("tool_call_parser") or "",
        reasoning_parser=cfg.get("reasoning_parser") or "",
        max_num_seqs=cfg.get("max_num_seqs"),
        mtp=bool(cfg.get("mtp")),
        mtp_num_tokens=int(cfg.get("mtp_num_tokens") or 2),
        load_format=cfg.get("load_format") or "",
        enable_chunked_prefill=bool(cfg.get("enable_chunked_prefill")),
        enable_prefix_caching=bool(cfg.get("enable_prefix_caching")),
        extra_flags=cfg.get("extra_flags") or "",
        tensor_parallel_size=int(cfg.get("tensor_parallel_size") or 1),
    )
    return "vllm serve " + model + " " + " ".join(args)


if not _case_dirs():
    pytest.skip("no corpus cases captured yet", allow_module_level=True)


@pytest.mark.parametrize("case_dir", _case_dirs(), ids=lambda p: p.name)
def test_corpus_case(monkeypatch, case_dir: Path):
    case = _load(case_dir)
    result = _recommend_offline(monkeypatch, case)
    cfg = result["config"]
    cmd = _final_command(case["model"], cfg)
    expect = case.get("expect") or {}

    # Hard rule, every case: never emit a known-bad construct.
    for bad in FORBIDDEN_ALWAYS:
        assert bad not in cmd, f"{case_dir.name}: forbidden {bad!r} in composed command:\n{cmd}"

    if "label_contains" in expect:
        assert expect["label_contains"].lower() in (result.get("label") or "").lower(), (
            f"{case_dir.name}: expected label containing {expect['label_contains']!r}, "
            f"got {result.get('label')!r}"
        )

    for key, want in (expect.get("config") or {}).items():
        assert cfg.get(key) == want, (
            f"{case_dir.name}: config[{key!r}] expected {want!r}, got {cfg.get(key)!r}"
        )

    for frag in expect.get("extra_flags_contains") or []:
        assert frag in (cfg.get("extra_flags") or ""), (
            f"{case_dir.name}: expected {frag!r} in extra_flags, "
            f"got {cfg.get('extra_flags')!r}"
        )

    for frag in expect.get("forbidden") or []:
        assert frag not in cmd, (
            f"{case_dir.name}: {frag!r} must not appear in composed command:\n{cmd}"
        )
