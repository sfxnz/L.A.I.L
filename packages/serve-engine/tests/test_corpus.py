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
import re
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


def _two_spark_topo() -> dict:
    nodes = [
        {
            "id": f"spark{i}",
            "qsfp_ip": f"10.100.8.{i}",
            "qsfp_if": "enp1s0f1np1",
            "ram_gib": 121.7,
            "local": i == 1,
            "ssh_host": f"spark{i}",
        }
        for i in (1, 2)
    ]
    return {
        "nodes": 2,
        "node_list": nodes,
        "head": nodes[0],
        "workers": nodes[1:],
        "fabric_ok": True,
        "available": True,
    }


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
    topo = _two_spark_topo if case.get("topology") == "two_spark" else _one_spark_topo
    monkeypatch.setattr(ac, "_cluster_topology", topo)
    # Pin Hub weight size so offline corpus is deterministic (no live blob API).
    if case.get("weights_gib") is not None:
        w = float(case["weights_gib"])
        monkeypatch.setattr(ac, "estimate_weights_gib", lambda *a, **k: w)
    elif case.get("topology") == "two_spark":
        # Overlay placement needs weights; pin a known-fit DSv4-sized blob when fixtures omit Hub.
        monkeypatch.setattr(ac, "estimate_weights_gib", lambda *a, **k: 155.4)
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


def _flag_occurrences(cmd: str, flag: str) -> int:
    """Count how many times ``flag`` appears as a CLI token (not a substring of another flag)."""
    return len(re.findall(rf"(?:^|\s){re.escape(flag)}(?:=|\s|$)", cmd))


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

    if "serve_blocked" in expect:
        assert bool(result.get("serve_blocked")) is bool(expect["serve_blocked"]), (
            f"{case_dir.name}: serve_blocked expected {expect['serve_blocked']!r}, "
            f"got {result.get('serve_blocked')!r}; topology={result.get('topology')}"
        )

    # Pinned Hub weight size (case.json) drives placement offline; assert bounds so a
    # bad pin / floor regression cannot silently claim a 400 GiB MoE fits one Spark.
    topo_w = (result.get("topology") or {}).get("weights_gib")
    pin_w = case.get("weights_gib")
    w_for_bounds = topo_w if topo_w is not None else pin_w
    if expect.get("weights_gib_min") is not None:
        assert w_for_bounds is not None, (
            f"{case_dir.name}: expected weights_gib_min but topology/case weights are missing"
        )
        assert float(w_for_bounds) >= float(expect["weights_gib_min"]), (
            f"{case_dir.name}: weights_gib {w_for_bounds!r} < min {expect['weights_gib_min']!r}"
        )
    if expect.get("weights_gib_max") is not None:
        assert w_for_bounds is not None, (
            f"{case_dir.name}: expected weights_gib_max but topology/case weights are missing"
        )
        assert float(w_for_bounds) <= float(expect["weights_gib_max"]), (
            f"{case_dir.name}: weights_gib {w_for_bounds!r} > max {expect['weights_gib_max']!r}"
        )
    if pin_w is not None and topo_w is not None:
        assert abs(float(topo_w) - float(pin_w)) < 0.05, (
            f"{case_dir.name}: topology weights_gib {topo_w!r} != case pin {pin_w!r}"
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

    for flag in expect.get("unique_flags") or []:
        n = _flag_occurrences(cmd, flag)
        assert n == 1, (
            f"{case_dir.name}: expected exactly one {flag!r} in composed command, "
            f"got {n}:\n{cmd}"
        )

    if expect.get("image_contains"):
        assert expect["image_contains"] in (cfg.get("image") or ""), (
            f"{case_dir.name}: expected image containing {expect['image_contains']!r}, "
            f"got {cfg.get('image')!r}"
        )
