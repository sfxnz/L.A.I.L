#!/usr/bin/env python3
"""Capture a Hugging Face model into the auto-config regression corpus.

Fetches the card and config.json once and writes them next to a stub
``case.json`` you then fill in with that vendor's published recipe.

Usage::

    cd packages/serve-engine
    PYTHONPATH=. python tests/corpus/_capture.py nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from app.services import autoconfig as ac

HERE = Path(__file__).resolve().parent


def slug(model_id: str) -> str:
    return model_id.strip().strip("/").replace("/", "__")


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

    case_path = out / "case.json"
    if not case_path.is_file():
        case_path.write_text(
            json.dumps(
                {
                    "model": model_id,
                    "mode": "workflow_max",
                    "note": "TODO: what the vendor's published recipe requires",
                    "expect": {
                        "label_contains": "",
                        "config": {},
                        "extra_flags_contains": [],
                        "forbidden": [],
                    },
                },
                indent=2,
            )
            + "\n"
        )
        print(f"captured {model_id} -> {out.name} (case.json is a stub — fill in expect)")
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
