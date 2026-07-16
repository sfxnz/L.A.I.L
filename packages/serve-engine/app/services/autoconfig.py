"""Auto-configure vLLM from the live Hugging Face model card.

Primary source of truth:
  1. README.md on huggingface.co (parsed `vllm serve` / docker recipes)
  2. config.json on huggingface.co (quantization_config, architecture)
  3. HF model API tags / cardData

Lab Safe / Workflow Max only supply the UMA envelope (util / default max-len
when the card is silent). Local cache is a fallback if the network fails —
never a substitute for the card when the card is available.
"""
from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..config import (
    DEFAULT_IMAGE_MAX,
    DEFAULT_IMAGE_SAFE,
    HOME,
    SAFE_MAX_LEN,
    SAFE_UTIL,
    WORKFLOW_MAX_LEN,
    WORKFLOW_UTIL,
)

HF_HUB = Path(os.environ.get("HF_HOME", HOME / ".cache" / "huggingface")) / "hub"
HF_UA = "local-ai-lab-autoconfig/2.0 (+https://huggingface.co)"

# Flags we map into the Serve form (everything else → extra_flags)
_FORM_FLAGS = {
    "--quantization",
    "-q",
    "--kv-cache-dtype",
    "--moe-backend",
    "--trust-remote-code",
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "--reasoning-parser",
    "--max-num-seqs",
    "--max-model-len",
    "--gpu-memory-utilization",
    "--load-format",
    "--enable-chunked-prefill",
    "--enable-prefix-caching",
    "--speculative-config",
    "--dtype",
    "--port",
    "--host",
    "--tensor-parallel-size",
    "--served-model-name",
}


# ─── HTTP ─────────────────────────────────────────────────────────────────────


def _hf_token() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if tok:
        return tok
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.is_file():
        try:
            return token_file.read_text().strip()
        except OSError:
            pass
    return ""


def _http_get(url: str, timeout: float = 20.0) -> tuple[Optional[str], Optional[str]]:
    """GET text body. Follows redirects. Returns (body, error)."""
    headers = {"User-Agent": HF_UA, "Accept": "*/*"}
    tok = _hf_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            # charset
            ctype = resp.headers.get("Content-Type", "")
            charset = "utf-8"
            if "charset=" in ctype:
                charset = ctype.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
            return raw.decode(charset, errors="replace"), None
    except HTTPError as e:
        return None, f"HTTP {e.code} for {url}"
    except (URLError, TimeoutError, OSError) as e:
        return None, f"{type(e).__name__}: {e}"


def fetch_hf_card(model_id: str, timeout: float = 20.0) -> dict[str, Any]:
    """Pull README + config + model API metadata from huggingface.co."""
    model_id = model_id.strip().rstrip("/")
    out: dict[str, Any] = {
        "model_id": model_id,
        "readme": None,
        "config": None,
        "api": None,
        "card_url": f"https://huggingface.co/{model_id}",
        "errors": [],
        "fetched": [],
    }
    if not model_id or model_id.startswith("/") or Path(model_id).is_dir():
        out["errors"].append("local path — no remote HF card")
        return out

    # Prefer /raw/main/ (follows cleanly); also try resolve/main
    readme_urls = [
        f"https://huggingface.co/{model_id}/raw/main/README.md",
        f"https://huggingface.co/{model_id}/resolve/main/README.md",
    ]
    for url in readme_urls:
        body, err = _http_get(url, timeout=timeout)
        if body and len(body) > 50 and not body.strip().startswith("<!"):
            out["readme"] = body
            out["fetched"].append(url)
            break
        if err:
            out["errors"].append(err)

    cfg_urls = [
        f"https://huggingface.co/{model_id}/raw/main/config.json",
        f"https://huggingface.co/{model_id}/resolve/main/config.json",
    ]
    for url in cfg_urls:
        body, err = _http_get(url, timeout=timeout)
        if body:
            try:
                out["config"] = json.loads(body)
                out["fetched"].append(url)
                break
            except json.JSONDecodeError:
                out["errors"].append(f"invalid JSON from {url}")
        elif err:
            out["errors"].append(err)

    api_url = f"https://huggingface.co/api/models/{model_id}"
    body, err = _http_get(api_url, timeout=timeout)
    if body:
        try:
            out["api"] = json.loads(body)
            out["fetched"].append(api_url)
            # API often embeds config — use if file missing
            if out["config"] is None and isinstance(out["api"].get("config"), dict):
                # HF API config is a subset; still useful for quant tags
                out["config"] = out["api"]["config"]
                out["fetched"].append(api_url + "#config")
        except json.JSONDecodeError:
            out["errors"].append("invalid model API JSON")
    elif err:
        out["errors"].append(err)

    return out


def load_local_fallback(model_id: str) -> dict[str, Any]:
    """Only if remote fetch failed — local HF cache README/config."""
    notes: list[str] = []
    p = Path(model_id)
    if p.is_dir() and (p / "config.json").is_file():
        cfg = _read_json(p / "config.json")
        readme = None
        if (p / "README.md").is_file():
            readme = (p / "README.md").read_text(encoding="utf-8", errors="replace")
        return {"config": cfg, "readme": readme, "notes": [f"local path {p}"]}

    parts = model_id.replace("\\", "/").split("/")
    folder = "models--" + "--".join(parts) if len(parts) >= 1 else ""
    root = HF_HUB / folder
    if not root.is_dir():
        return {"config": None, "readme": None, "notes": ["not in local HF cache"]}
    snaps = root / "snapshots"
    snap = None
    if snaps.is_dir():
        cands = [d for d in snaps.iterdir() if d.is_dir()]
        cands.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        snap = cands[0] if cands else None
    if not snap:
        return {"config": None, "readme": None, "notes": ["cache incomplete"]}
    cfg = _read_json(snap / "config.json") if (snap / "config.json").is_file() else None
    readme = None
    if (snap / "README.md").is_file():
        readme = (snap / "README.md").read_text(encoding="utf-8", errors="replace")
    notes.append(f"local cache fallback: {snap}")
    return {"config": cfg, "readme": readme, "notes": notes, "cache_path": str(snap)}


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


# ─── README parsing: extract every vllm serve candidate ───────────────────────


@dataclass
class ServeCandidate:
    raw: str
    section: str = ""
    env: list[str] = field(default_factory=list)
    model: Optional[str] = None
    args: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def _section_map(readme: str) -> list[tuple[int, str]]:
    """List of (char_offset, heading_text) for markdown headings."""
    heads: list[tuple[int, str]] = []
    for m in re.finditer(r"(?m)^(#{1,4})\s+(.+?)\s*$", readme):
        heads.append((m.start(), m.group(2).strip()))
    return heads


def _section_at(heads: list[tuple[int, str]], pos: int) -> str:
    cur = ""
    for off, title in heads:
        if off <= pos:
            cur = title
        else:
            break
    return cur


def _extract_code_blocks(readme: str) -> list[tuple[int, str, str]]:
    """Return (start_offset, lang, body) for fenced blocks."""
    out: list[tuple[int, str, str]] = []
    for m in re.finditer(r"```([a-zA-Z0-9_-]*)\n(.*?)```", readme, re.S):
        out.append((m.start(), (m.group(1) or "").lower(), m.group(2)))
    return out


def _is_vllm_block(body: str) -> bool:
    b = body.lower()
    return bool(
        re.search(r"\bvllm\s+serve\b", b)
        or re.search(r"vllm/vllm-openai", b)
        or (re.search(r"docker\s+run", b) and "vllm" in b)
        or ("--quantization" in b and "serve" in b)
    )


def _collapse_continuations(text: str) -> str:
    return re.sub(r"\\\s*\n", " ", text)


def _parse_one_serve_command(text: str) -> Optional[ServeCandidate]:
    """Parse a shell fragment that contains `vllm serve …` (and optional exports)."""
    text = _collapse_continuations(text)
    env: list[str] = []
    for m in re.finditer(r"(?:^|\s)export\s+([A-Z][A-Z0-9_]*)=([^\s\\]+)", text):
        val = m.group(2).strip().strip("'\"")
        env.append(f"{m.group(1)}={val}")
    # KEY=VAL prefixes before vllm
    for m in re.finditer(r"(?:^|\s)([A-Z][A-Z0-9_]*)=([^\s]+)\s+(?=vllm\b)", text):
        env.append(f"{m.group(1)}={m.group(2).strip().strip(chr(39)+chr(34))}")

    # docker run … image model args  → recover model + args after image
    docker_m = re.search(
        r"docker\s+run\b[^\n]*?\s(?:vllm/vllm-openai[^\s]*|nvcr\.io/[^\s]+)\s+(.+)",
        text,
        re.I | re.S,
    )
    if docker_m and "vllm serve" not in text.lower():
        try:
            tokens = shlex.split(docker_m.group(1), posix=True)
        except ValueError:
            tokens = docker_m.group(1).split()
        # skip leading "serve" if present (vllm-openai entrypoint is the model)
        if tokens and tokens[0] == "serve":
            tokens = tokens[1:]
        model = tokens[0] if tokens and not tokens[0].startswith("-") else None
        args = tokens[1:] if model else tokens
        # strip "vllm" "serve" if entrypoint style differs
        if args and args[0] == "serve":
            args = args[1:]
        cfg = _args_to_config(args, env)
        return ServeCandidate(raw=text.strip()[:500], env=env, model=model, args=args, config=cfg)

    # Find vllm serve
    m = re.search(r"\bvllm\s+serve\b", text, re.I)
    if not m:
        # bare flags after model id line?
        return None
    after = text[m.end() :].strip()
    # single logical line(s) until blank or comment-only
    lines = []
    for ln in after.splitlines():
        s = ln.strip()
        if not s:
            if lines:
                break
            continue
        if s.startswith("#") and lines:
            break
        if s.startswith("#"):
            continue
        # stop at next shell command that isn't a continuation
        if lines and re.match(r"^(export|uv |pip |source |cd |curl )\b", s):
            break
        lines.append(s)
    joined = " ".join(lines)
    # drop inline comments
    joined = re.sub(r"\s+#\s.*$", "", joined)

    try:
        tokens = shlex.split(joined, posix=True)
    except ValueError:
        tokens = joined.split()

    model = None
    args: list[str] = []
    if tokens and not tokens[0].startswith("-"):
        model = tokens[0]
        args = tokens[1:]
    else:
        args = tokens

    cfg = _args_to_config(args, env)
    return ServeCandidate(raw=text.strip()[:800], env=env, model=model, args=args, config=cfg)


def _args_to_config(args: list[str], env: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"docker_env": list(env)}
    extras: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]

        def take_val() -> str:
            nonlocal i
            if i + 1 < len(args) and not str(args[i + 1]).startswith("-"):
                i += 1
                return args[i]
            return ""

        if a in ("--quantization", "-q"):
            out["quantization"] = take_val()
        elif a == "--kv-cache-dtype":
            out["kv_cache_dtype"] = take_val()
        elif a == "--moe-backend":
            out["moe_backend"] = take_val()
        elif a == "--trust-remote-code":
            out["trust_remote_code"] = True
        elif a == "--enable-auto-tool-choice":
            out["enable_auto_tool_choice"] = True
        elif a == "--tool-call-parser":
            out["tool_call_parser"] = take_val()
        elif a == "--reasoning-parser":
            out["reasoning_parser"] = take_val()
        elif a == "--max-num-seqs":
            try:
                out["max_num_seqs"] = int(take_val())
            except ValueError:
                pass
        elif a == "--max-model-len":
            try:
                out["max_model_len"] = int(take_val())
            except ValueError:
                pass
        elif a == "--gpu-memory-utilization":
            try:
                out["util"] = float(take_val())
            except ValueError:
                pass
        elif a == "--load-format":
            out["load_format"] = take_val()
        elif a == "--enable-chunked-prefill":
            out["enable_chunked_prefill"] = True
        elif a == "--enable-prefix-caching":
            out["enable_prefix_caching"] = True
        elif a == "--speculative-config":
            cfg = take_val()
            out["mtp"] = "mtp" in cfg.lower()
            mm = re.search(r"num_speculative_tokens[\"']?\s*:\s*(\d+)", cfg)
            if mm:
                out["mtp_num_tokens"] = int(mm.group(1))
            extras += ["--speculative-config", cfg]
        elif a == "--tensor-parallel-size":
            v = take_val()
            try:
                out["tensor_parallel_size"] = int(v)
            except ValueError:
                out["tensor_parallel_size"] = v
        elif a in ("--port", "--host", "--served-model-name", "--dtype"):
            # consume value; lab controls port/host
            take_val()
        elif a.startswith("-"):
            # keep unknown flags in extra
            if i + 1 < len(args) and not str(args[i + 1]).startswith("-"):
                extras += [a, args[i + 1]]
                i += 1
            else:
                extras.append(a)
        i += 1

    if extras:
        out["extra_flags"] = " ".join(shlex.quote(x) if (" " in x or "{" in x) else x for x in extras)
    return out


def extract_serve_candidates(
    readme: str,
    *,
    detected: dict[str, Any] | None = None,
) -> list[ServeCandidate]:
    """All vllm serve recipes found on the model card."""
    if not readme:
        return []
    heads = _section_map(readme)
    candidates: list[ServeCandidate] = []
    seen_raw: set[str] = set()

    # 1) Fenced code blocks
    for start, _lang, body in _extract_code_blocks(readme):
        if not _is_vllm_block(body):
            # still check for export + serve across block
            if "vllm" not in body.lower() and "moe-backend" not in body:
                continue
        # A block may contain multiple serve lines / export+serve pairs
        fragments = _split_serve_fragments(body)
        for frag in fragments:
            cand = _parse_one_serve_command(frag)
            if not cand:
                continue
            key = re.sub(r"\s+", " ", cand.raw)[:200]
            if key in seen_raw:
                continue
            seen_raw.add(key)
            cand.section = _section_at(heads, start)
            candidates.append(cand)

    # 2) Unfenced inline `vllm serve …` lines in prose
    for m in re.finditer(r"(?m)^(?:export\s+\S+\s*\n)*[^\n]*\bvllm\s+serve\b[^\n]*(?:\n[^\n]*\\[^\n]*)*", readme):
        frag = m.group(0)
        if "```" in frag:
            continue
        cand = _parse_one_serve_command(frag)
        if not cand:
            continue
        key = re.sub(r"\s+", " ", cand.raw)[:200]
        if key in seen_raw:
            continue
        seen_raw.add(key)
        cand.section = _section_at(heads, m.start())
        candidates.append(cand)

    for c in candidates:
        c.score, c.reasons = score_candidate(c, readme, detected=detected)
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _split_serve_fragments(body: str) -> list[str]:
    """Split a code block into independent serve command fragments."""
    body = body.strip()
    if not body:
        return []
    # If multiple `vllm serve` occurrences, split before each
    parts = re.split(r"(?=(?:^|\n)(?:export\s+[A-Z0-9_]+=\S+\s*\n)*[^\n]*\bvllm\s+serve\b)", body)
    frags = [p.strip() for p in parts if p.strip() and re.search(r"\bvllm\s+serve\b|docker\s+run", p, re.I)]
    if not frags and _is_vllm_block(body):
        frags = [body]
    # Also: block that is only `export …\nvllm serve …` once
    if not frags and re.search(r"vllm\s+serve", body, re.I):
        frags = [body]
    return frags or ([body] if "vllm" in body.lower() else [])


def score_candidate(
    c: ServeCandidate,
    readme: str,
    *,
    detected: dict[str, Any] | None = None,
) -> tuple[float, list[str]]:
    """Higher = better recommended recipe from the card."""
    score = 0.0
    reasons: list[str] = []
    sec = (c.section or "").lower()
    cfg = c.config
    det = detected or {}

    # Prefer dedicated vLLM sections near the top of the card
    if "vllm run" in sec or sec.strip() in ("vllm", "vllm run instructions") or "dgx spark" in sec:
        score += 50
        reasons.append(f"in section «{c.section}»")
    elif "vllm" in sec:
        score += 35
        reasons.append(f"in section «{c.section}»")
    elif any(k in sec for k in ("deploy", "inference", "serving", "quickstart", "usage")):
        score += 15
        reasons.append(f"in section «{c.section}»")

    # Performance / Spark-relevant flags from the card
    moe = (cfg.get("moe_backend") or "").strip()
    if moe:
        score += 25
        reasons.append(f"--moe-backend {moe}")
    if any(e.startswith("CUTE_DSL_ARCH=") for e in cfg.get("docker_env") or []):
        score += 30
        reasons.append("CUTE_DSL_ARCH (Spark / cute-DSL path)")
    if cfg.get("quantization"):
        score += 20
        reasons.append(f"--quantization {cfg['quantization']}")
    if cfg.get("kv_cache_dtype"):
        score += 10
        reasons.append(f"--kv-cache-dtype {cfg['kv_cache_dtype']}")
    if cfg.get("reasoning_parser"):
        score += 12
        reasons.append(f"--reasoning-parser {cfg['reasoning_parser']}")
    if cfg.get("tool_call_parser") or cfg.get("enable_auto_tool_choice"):
        score += 10
        reasons.append("tool-call flags")
    if cfg.get("trust_remote_code"):
        score += 5
    if cfg.get("max_num_seqs"):
        score += 8
    if cfg.get("load_format"):
        score += 5
    if cfg.get("mtp") or "speculative-config" in (cfg.get("extra_flags") or ""):
        # MTP is optional on cards — slight boost but not always "best first boot"
        score += 6
        reasons.append("speculative/MTP (optional)")

    # Penalize toy/debug max lengths when other recipes exist
    ml = cfg.get("max_model_len")
    if isinstance(ml, int):
        if ml <= 8192:
            score -= 25
            reasons.append(f"short max-model-len={ml} (likely demo)")
        elif ml >= 65536:
            score += 8
            reasons.append(f"long max-model-len={ml}")

    # Penalize YaRN / extreme override demos
    raw_l = c.raw.lower()
    if "yarn" in raw_l or "hf-overrides" in raw_l or "1010000" in raw_l:
        score -= 40
        reasons.append("YaRN / extreme rope override demo — not default")
    if "..." in c.raw or "vllm serve ..." in raw_l:
        score -= 50
        reasons.append("placeholder command")

    # Bare `vllm serve model` with no flags: baseline only
    if not cfg.get("quantization") and not cfg.get("moe_backend") and not cfg.get("docker_env"):
        if len(c.args) == 0:
            score += 1
            reasons.append("minimal serve (defaults)")
        else:
            score += 3

    # Card prose near CUTE_DSL / flashinfer often marks the recommended path
    if "cute_dsl" in raw_l or "CUTE_DSL" in c.raw:
        score += 5

    # Checkpoint truth from config.json: mixed FP8+NVFP4 MoE rejects flashinfer_b12x on vLLM 0.25.x
    if _flashinfer_b12x_unsafe_for_checkpoint(det) and moe == "flashinfer_b12x":
        score -= 80
        reasons.append(
            "PENALTY: config.json has FP8 MoE layers — flashinfer_b12x crashes "
            "(ValueError: not supported for FP8 MoE)"
        )
    elif moe == "flashinfer_b12x" and not det:
        # unknown checkpoint — keep card score but no extra boost
        pass

    return score, reasons


def _flashinfer_b12x_unsafe_for_checkpoint(detected: dict[str, Any]) -> bool:
    """True when forcing flashinfer_b12x will crash (FP8 MoE path on compressed-tensors)."""
    if not detected:
        return False
    if detected.get("is_mixed_nvfp4_fp8"):
        return True
    # Any compressed-tensors MoE with FP8 groups (even if not labeled mixed)
    if (
        detected.get("is_moe")
        and detected.get("has_fp8")
        and (detected.get("quant_flag") == "compressed-tensors" or detected.get("quant_method") == "compressed-tensors")
    ):
        return True
    formats = " ".join(detected.get("quant_formats") or [])
    if detected.get("is_moe") and "float-quantized" in formats and "nvfp4" in formats:
        return True
    return False


def _card_has_flashinfer_b12x(
    candidates: list[ServeCandidate],
    readme: str | None = None,
) -> bool:
    """True if any parsed recipe (or card prose) recommends flashinfer_b12x."""
    for c in candidates:
        if (c.config.get("moe_backend") or "").strip() == "flashinfer_b12x":
            return True
        if "flashinfer_b12x" in (c.raw or ""):
            return True
    if readme and "flashinfer_b12x" in readme:
        return True
    return False


def _note_card_flashinfer_avoidance(
    *,
    candidates: list[ServeCandidate],
    readme: str | None,
    detected: dict[str, Any],
    cfg: dict[str, Any],
    warnings: list[str],
    rationale: list[str],
) -> None:
    """Always surface why flashinfer_b12x is not used when the card recommends it.

    Scoring may pick a non-flashinfer recipe so moe_backend is already empty —
    criterion 3 still requires a clear warning/rationale for that adjustment.
    """
    if not _flashinfer_b12x_unsafe_for_checkpoint(detected):
        return
    if not _card_has_flashinfer_b12x(candidates, readme):
        return

    # Already warned by _apply_checkpoint_safety strip path?
    if any("flashinfer_b12x" in w for w in warnings):
        # Still ensure rationale mentions avoidance if missing
        if not any("flashinfer_b12x" in r for r in rationale):
            rationale.append(
                "SAFETY: card recommends flashinfer_b12x; config.json mixed FP8 MoE → moe-backend left empty (auto)"
            )
        return

    warnings.append(
        "HF card recommends --moe-backend flashinfer_b12x (e.g. DGX Spark recipe), but "
        "config.json shows mixed FP8 + NVFP4 MoE (compressed-tensors). flashinfer_b12x is "
        "NVFP4-only and crashes with: 'moe_backend=flashinfer_b12x is not supported for FP8 MoE'. "
        "Using empty moe-backend (vLLM auto: TRITON for FP8 MoE, FlashInfer NVFP4 where valid)."
    )
    rationale.append(
        "SAFETY (config.json > card recipe): avoided flashinfer_b12x from card; "
        f"moe-backend={cfg.get('moe_backend')!r} (empty = auto)"
    )
    # Ensure CUTE_DSL from card Spark path is still present when card mentioned it
    if readme and ("CUTE_DSL_ARCH" in readme or "sm_121a" in readme):
        env = list(cfg.get("docker_env") or [])
        if not any(e.startswith("CUTE_DSL_ARCH=") for e in env):
            env.append("CUTE_DSL_ARCH=sm_121a")
            cfg["docker_env"] = env
            rationale.append("Card Spark path → CUTE_DSL_ARCH=sm_121a (kept without flashinfer_b12x)")


def _apply_checkpoint_safety(
    cfg: dict[str, Any],
    detected: dict[str, Any],
    warnings: list[str],
    rationale: list[str],
) -> None:
    """Override card flags that conflict with the actual checkpoint (config.json).

    The HF card is still the primary source, but config.json is ground truth for
    quantization layout. Card recipes that crash on this checkpoint are fixed here.
    """
    moe = (cfg.get("moe_backend") or "").strip()
    if moe == "flashinfer_b12x" and _flashinfer_b12x_unsafe_for_checkpoint(detected):
        cfg["moe_backend"] = ""
        warnings.append(
            "Card recipe used --moe-backend flashinfer_b12x, but HF config.json shows "
            "mixed FP8 + NVFP4 MoE (compressed-tensors). flashinfer_b12x is NVFP4-only and "
            "crashes with: 'moe_backend=flashinfer_b12x is not supported for FP8 MoE'. "
            "Cleared moe-backend so vLLM auto-selects (TRITON for FP8 MoE, FlashInfer NVFP4 where valid)."
        )
        rationale.append(
            "SAFETY (config.json > card flag): removed flashinfer_b12x for mixed FP8 MoE checkpoint"
        )
        # Keep cute-DSL if card mentioned it
        env = list(cfg.get("docker_env") or [])
        if not any(e.startswith("CUTE_DSL_ARCH=") for e in env):
            env.append("CUTE_DSL_ARCH=sm_121a")
            cfg["docker_env"] = env
            rationale.append("Kept/added CUTE_DSL_ARCH=sm_121a from card Spark guidance")

    # Lab-friendly KV for large NVFP4/MoE when card is silent
    if (
        not cfg.get("kv_cache_dtype")
        and (detected.get("has_nvfp4") or detected.get("is_moe") or detected.get("has_fp8"))
    ):
        cfg["kv_cache_dtype"] = "fp8"
        rationale.append(
            "HF card silent on KV dtype → --kv-cache-dtype fp8 (Spark UMA headroom for long context)"
        )

    if detected.get("is_moe") and (cfg.get("max_num_seqs") is None):
        cfg["max_num_seqs"] = 4
        rationale.append("MoE (from config) → --max-num-seqs 4 when card omitted it")


# ─── config.json analysis (fills gaps the card left empty) ────────────────────


def analyze_config(cfg: dict[str, Any], model_id: str = "", tags: list[str] | None = None) -> dict[str, Any]:
    mid = model_id.lower()
    tags = [t.lower() for t in (tags or [])]
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    model_type = (cfg.get("model_type") or text_cfg.get("model_type") or "").lower()
    architectures = cfg.get("architectures") or text_cfg.get("architectures") or []
    if not isinstance(architectures, list):
        architectures = [str(architectures)]

    max_pos = (
        cfg.get("max_position_embeddings")
        or text_cfg.get("max_position_embeddings")
        or cfg.get("max_sequence_length")
    )

    qc = cfg.get("quantization_config") or {}
    if not isinstance(qc, dict):
        qc = {}

    quant_method = (qc.get("quant_method") or "").lower()
    formats: set[str] = set()
    if qc.get("format"):
        formats.add(str(qc["format"]).lower())
    has_nvfp4 = False
    has_fp8 = False
    has_modelopt_layers = False

    for g in (qc.get("config_groups") or {}).values():
        if not isinstance(g, dict):
            continue
        if g.get("format"):
            formats.add(str(g["format"]).lower())
        w = g.get("weights") or {}
        if isinstance(w, dict):
            if w.get("num_bits") == 4 or "nvfp4" in str(w.get("type", "")).lower():
                has_nvfp4 = True
            if w.get("num_bits") == 8:
                has_fp8 = True

    qlayers = qc.get("quantized_layers") or {}
    if isinstance(qlayers, dict) and qlayers:
        has_modelopt_layers = True
        for meta in qlayers.values():
            if not isinstance(meta, dict):
                continue
            algo = str(meta.get("quant_algo") or "").upper()
            if "NVFP4" in algo or "FP4" in algo or "W4A16" in algo:
                has_nvfp4 = True
            if "FP8" in algo:
                has_fp8 = True

    tag_blob = " ".join(tags)
    if "nvfp4" in mid or "fp4" in mid or "fp4" in tag_blob or "nvfp4" in tag_blob:
        has_nvfp4 = True
    if "modelopt" in tag_blob or "modelopt" in tags:
        if not quant_method:
            quant_method = "modelopt"
    if "compressed-tensors" in tag_blob or "compressed-tensors" in tags:
        if not quant_method:
            quant_method = "compressed-tensors"
    if quant_method == "fp8" or "fp8" in mid:
        has_fp8 = True

    fmt_blob = " ".join(formats)
    if "nvfp4" in fmt_blob:
        has_nvfp4 = True
    if "float-quantized" in fmt_blob:
        has_fp8 = True

    if quant_method in ("compressed-tensors", "compressed_tensors"):
        quant_flag = "compressed-tensors"
    elif quant_method in ("modelopt", "modelopt_fp4"):
        quant_flag = "modelopt"
    elif quant_method == "fp8":
        quant_flag = "fp8"
    elif has_modelopt_layers and has_nvfp4:
        quant_flag = "modelopt"
    elif "modelopt" in tags:
        quant_flag = "modelopt"
    elif "compressed-tensors" in tags:
        quant_flag = "compressed-tensors"
    elif has_nvfp4:
        quant_flag = ""
    else:
        quant_flag = quant_method or ""

    is_moe = bool(
        "moe" in model_type
        or any("moe" in str(a).lower() for a in architectures)
        or re.search(r"A\d+B", model_id)
        or "a3b" in mid
        or cfg.get("num_experts")
        or text_cfg.get("num_experts")
    )
    is_mixed = has_nvfp4 and has_fp8 and (
        quant_method in ("compressed-tensors", "compressed_tensors")
        or "mixed" in fmt_blob
        or ("nvfp4-pack" in fmt_blob and "float-quantized" in fmt_blob)
    )

    family = "unknown"
    blob = f"{model_type} {' '.join(str(a) for a in architectures)} {model_id}".lower()
    if "nemotron" in blob:
        family = "nemotron"
    elif "qwen" in blob:
        family = "qwen"
    elif "llama" in blob:
        family = "llama"
    elif "gemma" in blob:
        family = "gemma"

    return {
        "model_type": model_type or None,
        "architectures": architectures,
        "quant_method": quant_method or None,
        "quant_formats": sorted(formats),
        "has_nvfp4": has_nvfp4,
        "has_fp8": has_fp8,
        "is_mixed_nvfp4_fp8": is_mixed,
        "is_moe": is_moe,
        "has_modelopt_layers": has_modelopt_layers,
        "max_position_embeddings": max_pos,
        "quant_flag": quant_flag,
        "family": family,
    }


def _empty_config(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "quantization": "",
        "kv_cache_dtype": "",
        "moe_backend": "",
        "trust_remote_code": False,
        "reasoning_parser": "",
        "tool_call_parser": "",
        "enable_auto_tool_choice": False,
        "max_num_seqs": None,
        "docker_env": [],
        "extra_flags": "",
        "mtp": False,
        "mtp_num_tokens": 2,
        "load_format": "",
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "image": "",
        "util": None,
        "max_model_len": None,
    }


def _merge_fill(base: dict[str, Any], overlay: dict[str, Any]) -> list[str]:
    """Fill empty base fields from overlay. Returns keys filled."""
    filled: list[str] = []
    for k, v in overlay.items():
        if v is None or v == "" or v == []:
            continue
        cur = base.get(k)
        if k == "docker_env":
            env = list(base.get("docker_env") or [])
            for item in v if isinstance(v, list) else [v]:
                if not item or "=" not in str(item):
                    continue
                key = str(item).split("=", 1)[0]
                env = [e for e in env if not e.startswith(key + "=")]
                env.append(str(item))
            if env != list(base.get("docker_env") or []):
                base["docker_env"] = env
                filled.append(k)
            continue
        if cur in (None, "", [], False) or (k == "mtp_num_tokens" and cur == 2 and v != 2):
            if k == "mtp" and cur is False and v is True:
                base[k] = True
                filled.append(k)
            elif cur in (None, "", []):
                base[k] = v
                filled.append(k)
            elif k in ("trust_remote_code", "enable_auto_tool_choice", "enable_chunked_prefill", "enable_prefix_caching") and not cur and v:
                base[k] = v
                filled.append(k)
    return filled


def _apply_card_candidate(base: dict[str, Any], cand: ServeCandidate) -> list[str]:
    """Apply winning card recipe onto base (card wins for set fields)."""
    applied: list[str] = []
    cfg = cand.config
    for k, v in cfg.items():
        if k == "docker_env":
            if v:
                base["docker_env"] = list(v)
                applied.append("docker_env")
            continue
        if v is None or v == "":
            continue
        base[k] = v
        applied.append(k)
    # empty moe_backend explicitly set?
    if "moe_backend" in cfg and cfg["moe_backend"] == "":
        base["moe_backend"] = ""
        if "moe_backend" not in applied:
            applied.append("moe_backend")
    return applied


def _card_prose_hints(readme: str) -> dict[str, Any]:
    """Pull extra hints from card prose outside code blocks."""
    hints: dict[str, Any] = {"docker_env": [], "notes": []}
    if not readme:
        return hints
    if re.search(r"CUTE_DSL_ARCH\s*=\s*sm_121a", readme) or "export CUTE_DSL_ARCH" in readme:
        hints["docker_env"].append("CUTE_DSL_ARCH=sm_121a")
        hints["notes"].append("Card mentions CUTE_DSL_ARCH=sm_121a")
    if re.search(r"do\s+\*\*not\*\*\s+use\s+the\s+marlin", readme, re.I) or re.search(
        r"do not use the marlin", readme, re.I
    ):
        hints["notes"].append("Card: do NOT use Marlin MoE backend")
    if re.search(r"reasoning.?parser\s+qwen3", readme, re.I):
        hints["reasoning_parser"] = "qwen3"
    if re.search(r"tool.?call.?parser\s+qwen3_coder", readme, re.I):
        hints["tool_call_parser"] = "qwen3_coder"
        hints["enable_auto_tool_choice"] = True
    if re.search(r"--quantization\s+modelopt", readme):
        hints["quantization"] = "modelopt"
    if re.search(r"--quantization\s+compressed-tensors", readme):
        hints["quantization"] = "compressed-tensors"
    return hints


def _fill_from_config_detection(
    base: dict[str, Any],
    detected: dict[str, Any],
    rationale: list[str],
) -> None:
    """Only fill gaps using HF config.json / API tags — not lab override recipes."""
    if not base.get("quantization") and detected.get("quant_flag"):
        base["quantization"] = detected["quant_flag"]
        rationale.append(
            f"HF config/tags → --quantization {detected['quant_flag']} "
            f"(card serve line had no --quantization)"
        )

    family = detected.get("family")
    if family == "qwen":
        if not base.get("reasoning_parser"):
            base["reasoning_parser"] = "qwen3"
            rationale.append("Qwen architecture (from HF config) → --reasoning-parser qwen3")
        if not base.get("tool_call_parser"):
            base["tool_call_parser"] = "qwen3_coder"
            base["enable_auto_tool_choice"] = True
            rationale.append("Qwen architecture → tool-call-parser qwen3_coder + auto tool choice")
        if not base.get("trust_remote_code"):
            base["trust_remote_code"] = True
    if family == "nemotron":
        if not base.get("reasoning_parser"):
            base["reasoning_parser"] = "nemotron_v3"
            rationale.append("Nemotron (from HF) → --reasoning-parser nemotron_v3")
        base["enable_auto_tool_choice"] = True
        if not base.get("tool_call_parser"):
            base["tool_call_parser"] = "qwen3_coder"
        base["trust_remote_code"] = True


def _apply_mode_envelope(cfg: dict[str, Any], mode: str, rationale: list[str], card_set_max_len: bool) -> None:
    if mode == "lab_safe":
        if cfg.get("util") is None:
            cfg["util"] = SAFE_UTIL
            rationale.append(f"Lab Safe envelope → util={SAFE_UTIL} (not on HF card)")
        # Card max-len may be huge (262k); Lab Safe still prefers safer default unless card was modest
        if not card_set_max_len:
            cfg["max_model_len"] = SAFE_MAX_LEN
            rationale.append(f"Lab Safe envelope → max-model-len={SAFE_MAX_LEN} (card silent)")
        elif cfg.get("max_model_len") and cfg["max_model_len"] > SAFE_MAX_LEN:
            rationale.append(
                f"Card max-model-len={cfg['max_model_len']} kept; "
                f"Lab Safe often uses {SAFE_MAX_LEN} — edit if you want more free UMA"
            )
        if not cfg.get("image"):
            cfg["image"] = DEFAULT_IMAGE_SAFE
    else:
        if cfg.get("util") is None:
            cfg["util"] = WORKFLOW_UTIL
            rationale.append(f"Workflow Max envelope → util={WORKFLOW_UTIL} (not on HF card)")
        if not card_set_max_len:
            cfg["max_model_len"] = WORKFLOW_MAX_LEN
            rationale.append(f"Workflow Max envelope → max-model-len={WORKFLOW_MAX_LEN} (card silent)")
        if not cfg.get("image"):
            cfg["image"] = DEFAULT_IMAGE_MAX

    if mode == "lab_safe" and cfg.get("util") is not None and cfg["util"] > SAFE_UTIL + 1e-9:
        cfg["util"] = SAFE_UTIL
        rationale.append(f"clamped util to Lab Safe max {SAFE_UTIL}")


# ─── Public API ───────────────────────────────────────────────────────────────


def recommend(
    model: str,
    *,
    mode: str = "lab_safe",
    fetch_remote: bool = True,
) -> dict[str, Any]:
    """Build serve config primarily from the Hugging Face model card on the website."""
    model = (model or "").strip()
    if not model:
        raise ValueError("model is required")

    rationale: list[str] = []
    warnings: list[str] = []
    sources: list[dict[str, str]] = []
    confidence = "low"
    cfg = _empty_config(model)

    readme: Optional[str] = None
    hf_config: Optional[dict] = None
    api: Optional[dict] = None
    card_url = f"https://huggingface.co/{model}" if "/" in model or not Path(model).is_dir() else None
    from_website = False

    if fetch_remote and not Path(model).is_dir() and not model.startswith("/"):
        remote = fetch_hf_card(model)
        readme = remote.get("readme")
        hf_config = remote.get("config")
        api = remote.get("api")
        for e in remote.get("errors") or []:
            if readme is None or hf_config is None:
                rationale.append(f"HF fetch note: {e}")
        for u in remote.get("fetched") or []:
            sources.append({"kind": "huggingface", "ref": u, "notes": "fetched from huggingface.co"})
        if readme:
            from_website = True
            rationale.append(f"Using live model card from {card_url}")
            confidence = "high"
        if hf_config and not readme:
            confidence = "medium"
            rationale.append("Got HF config.json from website; no README serve recipes found")

    if readme is None or hf_config is None:
        local = load_local_fallback(model)
        for n in local.get("notes") or []:
            rationale.append(n)
        if readme is None and local.get("readme"):
            readme = local["readme"]
            sources.append(
                {
                    "kind": "hf_card_local_fallback",
                    "ref": local.get("cache_path") or model,
                    "notes": "offline fallback — not live website",
                }
            )
            warnings.append("Could not fetch live HF card; used local cache README")
            if confidence == "low":
                confidence = "medium"
        if hf_config is None and local.get("config"):
            hf_config = local["config"]
            sources.append(
                {
                    "kind": "hf_config_local_fallback",
                    "ref": local.get("cache_path") or model,
                    "notes": "offline fallback config.json",
                }
            )

    tags: list[str] = []
    if api:
        tags = list(api.get("tags") or [])
        if api.get("cardData"):
            sources.append(
                {
                    "kind": "hf_api",
                    "ref": f"https://huggingface.co/api/models/{model}",
                    "notes": f"tags: {', '.join(tags[:12])}",
                }
            )

    detected = analyze_config(hf_config or {}, model, tags=tags)
    if from_website:
        detected["config_source"] = "huggingface.co"
    elif hf_config:
        detected["config_source"] = "local_cache"
    else:
        detected["config_source"] = None
        warnings.append("No HF config.json available — quant detection limited")

    # ── Parse card for best vllm serve (scored with config.json knowledge) ──
    candidates: list[ServeCandidate] = []
    if readme:
        candidates = extract_serve_candidates(readme, detected=detected)
        if candidates:
            best = candidates[0]
            applied = _apply_card_candidate(cfg, best)
            rationale.append(
                f"Best card recipe (score {best.score:.0f})"
                + (f" in «{best.section}»" if best.section else "")
                + f": {best.raw[:180].replace(chr(10), ' ')}"
            )
            for r in best.reasons:
                rationale.append(f"  · {r}")
            if applied:
                rationale.append(f"Applied from card: {', '.join(applied)}")
            sources.insert(
                0,
                {
                    "kind": "hf_card_recipe",
                    "ref": card_url or "README.md",
                    "notes": f"score={best.score:.0f}; section={best.section or 'n/a'}",
                },
            )
            confidence = "high" if from_website else confidence
        else:
            warnings.append("Model card has no parseable `vllm serve` command")
            rationale.append("No vllm serve snippets on card — using config.json + prose hints")

        prose = _card_prose_hints(readme)
        for n in prose.pop("notes", []):
            rationale.append(n)
        filled = _merge_fill(cfg, prose)
        if filled:
            rationale.append(f"Card prose filled: {', '.join(filled)}")

    card_set_max_len = cfg.get("max_model_len") is not None

    # Card may recommend multi-GPU; this lab serves single-GPU (Spark)
    tp = cfg.pop("tensor_parallel_size", None)
    if tp is not None:
        try:
            tp_n = int(tp)
        except (TypeError, ValueError):
            tp_n = 1
        if tp_n > 1:
            warnings.append(
                f"HF card uses --tensor-parallel-size {tp_n}; this lab serves with TP=1 "
                f"(single GPU). Edit extra flags if you have a multi-GPU cluster."
            )
            rationale.append(f"Card tensor-parallel-size={tp_n} noted but not applied (lab TP=1)")

    # Gaps only: HF config.json / tags (still from the model on the hub)
    _fill_from_config_detection(cfg, detected, rationale)

    # config.json is ground truth for quant layout — fix card flags that crash
    _apply_checkpoint_safety(cfg, detected, warnings, rationale)

    # Always explain flashinfer avoidance when card recommends it but checkpoint forbids it
    # (even if scoring already picked a non-flashinfer recipe and moe_backend is empty)
    _note_card_flashinfer_avoidance(
        candidates=candidates,
        readme=readme,
        detected=detected,
        cfg=cfg,
        warnings=warnings,
        rationale=rationale,
    )

    # Mode envelope for util / defaults when card silent
    _apply_mode_envelope(cfg, mode, rationale, card_set_max_len=card_set_max_len)

    max_pos = detected.get("max_position_embeddings")
    if isinstance(max_pos, int) and max_pos > 0 and cfg.get("max_model_len"):
        if cfg["max_model_len"] > max_pos:
            cfg["max_model_len"] = max_pos
            rationale.append(f"Capped max-model-len to config max_position_embeddings={max_pos}")

    # Normalize env
    env_out: list[str] = []
    seen: set[str] = set()
    for e in cfg.get("docker_env") or []:
        if not e or "=" not in e:
            continue
        k = e.split("=", 1)[0]
        if k in seen:
            env_out = [x for x in env_out if not x.startswith(k + "=")]
        seen.add(k)
        env_out.append(e)
    cfg["docker_env"] = env_out

    # Alternate recipes from the same card (for UI)
    alternatives = []
    for c in candidates[1:6]:
        alternatives.append(
            {
                "score": c.score,
                "section": c.section,
                "raw": c.raw[:300],
                "config": c.config,
                "reasons": c.reasons,
            }
        )

    label = None
    if candidates:
        label = f"HF card: {candidates[0].section or 'vLLM recipe'}"
    elif from_website:
        label = "HF config (no serve recipe on card)"
    else:
        label = "Offline / incomplete"

    notes = None
    if from_website and candidates:
        notes = (
            f"Derived from the live Hugging Face card ({card_url}). "
            f"Picked the highest-scoring of {len(candidates)} vllm serve recipe(s) on the card."
        )
    elif from_website:
        notes = f"Fetched {card_url} but found no vllm serve blocks; used config.json + tags."
    else:
        notes = "Live HF card was not available; result may be incomplete."

    return {
        "model": model,
        "mode": mode,
        "confidence": confidence,
        "label": label,
        "notes": notes,
        "card_url": card_url,
        "from_website": from_website,
        "detected": detected,
        "config": cfg,
        "sources": sources,
        "rationale": rationale,
        "warnings": warnings,
        "card_recipes": [
            {
                "score": c.score,
                "section": c.section,
                "raw": c.raw[:400],
                "selected": i == 0,
                "reasons": c.reasons,
                "config": c.config,
            }
            for i, c in enumerate(candidates[:8])
        ],
        "alternatives": alternatives,
    }


def list_known_recipes() -> list[dict[str, Any]]:
    """No static lab recipe list — cards are authoritative. Keep endpoint stable."""
    return []
