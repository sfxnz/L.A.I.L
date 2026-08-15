#!/usr/bin/env python3
"""Capture a Hugging Face model into the auto-config regression corpus.

Fetches the card and config.json once and writes them next to a stub
``case.json`` you then fill in with that vendor's published recipe.

Also records Hub blob / safetensors-index weight size into ``case.json``
(``weights_gib`` + ``hub_api``) so offline corpus runs pin placement without
a live blob API.

Usage::

    cd packages/serve-engine
    PYTHONPATH=. python tests/corpus/_capture.py nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

from app.services import autoconfig as ac

HERE = Path(__file__).resolve().parent


def slug(model_id: str) -> str:
    return model_id.strip().strip("/").replace("/", "__")


def _weights_from_api_siblings(api: Optional[dict]) -> tuple[Optional[float], Optional[dict]]:
    """Sum one safetensors dump (not consolidated+shards) or GGUF siblings."""
    if not isinstance(api, dict):
        return None, None
    st: list[dict] = []
    gg: list[dict] = []
    for f in api.get("siblings") or []:
        if not isinstance(f, dict):
            continue
        name = str(f.get("rfilename") or f.get("path") or "")
        if name.endswith((".safetensors", ".bin")):
            st.append(f)
        elif name.endswith(".gguf"):
            gg.append(f)
    chosen = ac._select_weight_blobs(st) if st else gg
    tot = 0
    n = 0
    for f in chosen:
        size = f.get("size")
        if isinstance(size, (int, float)) and size > 0:
            tot += int(size)
            n += 1
    if tot <= 0:
        return None, {"siblings_weight_files": n, "source": None}
    gib = round(tot / (1024**3), 1)
    return gib, {
        "siblings_weight_files": n,
        "siblings_bytes": tot,
        "weights_gib": gib,
        "source": "blobs",
    }


def _weights_from_index(model_id: str, timeout: float = 20.0) -> tuple[Optional[float], Optional[dict]]:
    """Fallback: model.safetensors.index.json metadata.total_size."""
    body, err = ac._http_get(
        f"https://huggingface.co/{model_id}/resolve/main/model.safetensors.index.json",
        timeout=timeout,
    )
    if not body or err:
        return None, {"source": None, "index_error": err}
    try:
        d = json.loads(body)
    except json.JSONDecodeError:
        return None, {"source": None, "index_error": "invalid JSON"}
    tot = (d.get("metadata") or {}).get("total_size")
    if not isinstance(tot, (int, float)) or tot <= 0:
        return None, {"source": None, "index_error": "no metadata.total_size"}
    gib = round(float(tot) / (1024**3), 1)
    return gib, {
        "index_total_size": int(tot),
        "weights_gib": gib,
        "source": "index",
    }


def _capture_weights(model_id: str, card: dict[str, Any], timeout: float = 20.0) -> dict[str, Any]:
    """Prefer exact Hub blob sizes; fall back to safetensors index; then estimate."""
    hub: dict[str, Any] = {"model_id": model_id}
    weights_gib: Optional[float] = None
    source: Optional[str] = None

    # Prefer blobs=true API (same as estimate_weights_gib path).
    body, err = ac._http_get(
        f"https://huggingface.co/api/models/{model_id}?blobs=true",
        timeout=timeout,
    )
    api_blobs: Optional[dict] = None
    if body and not err:
        try:
            api_blobs = json.loads(body)
        except json.JSONDecodeError:
            hub["blobs_error"] = "invalid JSON"
    elif err:
        hub["blobs_error"] = err

    if api_blobs is None and isinstance(card.get("api"), dict):
        api_blobs = card["api"]

    gib, meta = _weights_from_api_siblings(api_blobs)
    if gib is not None:
        weights_gib, source = gib, "blobs"
        hub.update(meta or {})
    else:
        gib, meta = _weights_from_index(model_id, timeout=timeout)
        if gib is not None:
            weights_gib, source = gib, "index"
            hub.update(meta or {})
        elif meta:
            hub.update(meta)

    if weights_gib is None:
        # Last resort: same heuristic/floor path as production (may hit network again).
        try:
            est = ac.estimate_weights_gib(model_id, card.get("config"))
            if est is not None:
                weights_gib = float(est)
                source = "estimate"
                hub["weights_gib"] = weights_gib
                hub["source"] = source
        except Exception as e:  # noqa: BLE001 — capture never bricks on weight probe
            hub["estimate_error"] = str(e)

    if source:
        hub["source"] = source
    if weights_gib is not None:
        hub["weights_gib"] = weights_gib
    return {"weights_gib": weights_gib, "weights_source": source, "hub_api": hub}


def capture(model_id: str) -> int:
    card = ac.fetch_hf_card(model_id)
    if not card.get("readme") and not card.get("config"):
        print(f"FAIL {model_id}: nothing fetched — {card.get('errors')}", file=sys.stderr)
        return 1

    out = HERE / slug(model_id)
    out.mkdir(parents=True, exist_ok=True)

    if card.get("readme"):
        (out / "card.md").write_text(card["readme"], encoding="utf-8")
    if card.get("config"):
        (out / "config.json").write_text(json.dumps(card["config"], indent=2) + "\n")

    weight_info = _capture_weights(model_id, card)
    hub_path = out / "hub_api.json"
    hub_path.write_text(json.dumps(weight_info.get("hub_api") or {}, indent=2) + "\n")

    case_path = out / "case.json"
    if not case_path.is_file():
        stub: dict[str, Any] = {
            "model": model_id,
            "mode": "workflow_max",
            "note": "TODO: what the vendor's published recipe requires",
            "expect": {
                "label_contains": "",
                "serve_blocked": False,
                "config": {},
                "extra_flags_contains": [],
                "forbidden": [],
            },
        }
        if weight_info.get("weights_gib") is not None:
            stub["weights_gib"] = weight_info["weights_gib"]
            stub["weights_source"] = weight_info.get("weights_source")
            # Loose bounds so placement regression is obvious without being brittle.
            w = float(weight_info["weights_gib"])
            stub["expect"]["weights_gib_min"] = round(max(0.0, w * 0.7), 1)
            stub["expect"]["weights_gib_max"] = round(w * 1.3 + 5.0, 1)
        case_path.write_text(json.dumps(stub, indent=2) + "\n")
        print(
            f"captured {model_id} -> {out.name} "
            f"(weights_gib={weight_info.get('weights_gib')!r} "
            f"source={weight_info.get('weights_source')!r}; case.json is a stub — fill in expect)"
        )
    else:
        # Refresh weight pin when Hub probe succeeds; leave expect/hand edits alone.
        try:
            case = json.loads(case_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            case = None
        if isinstance(case, dict) and weight_info.get("weights_gib") is not None:
            case["weights_gib"] = weight_info["weights_gib"]
            case["weights_source"] = weight_info.get("weights_source")
            case_path.write_text(json.dumps(case, indent=2) + "\n")
            print(
                f"refreshed {model_id} -> {out.name} "
                f"(weights_gib={weight_info['weights_gib']!r}; kept expect)"
            )
        else:
            print(f"refreshed {model_id} -> {out.name} (kept existing case.json)")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    return max(capture(m) for m in argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
