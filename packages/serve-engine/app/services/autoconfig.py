"""Auto-configure vLLM, llama.cpp, or SGLang from the live HF card plus vendor docs.

Primary sources:
  1. README.md on huggingface.co (parsed `vllm serve` / docker recipes)
  2. config.json on huggingface.co (quantization_config, architecture)
  3. HF model API tags / cardData
  4. Derived public recipes (Unsloth docs, NVIDIA cookbooks, vLLM docs, GitHub)
     looked up from the model id / family even when the card has no links.

Hardware envelope (util / max-len / VL / reserved UMA) is computed from
weights + live topology. Local cache is a fallback if the network fails —
never a substitute for the card when the card is available.
"""
from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
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
        return tok.strip()
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.is_file():
        try:
            return token_file.read_text().strip()
        except OSError:
            pass
    return ""


# Cached probe: stale/expired hf_oauth tokens 401 public card fetches; anonymous often works.
_HF_TOKEN_USABLE: Optional[bool] = None


def hf_token_usable(force: bool = False) -> bool:
    """True when a stored/env HF token authenticates against the Hub API.

    Invalid tokens must not be sent on public GETs — huggingface.co returns 401
    for bad Bearer tokens even on public models (anonymous 200).
    """
    global _HF_TOKEN_USABLE
    if not force and _HF_TOKEN_USABLE is not None:
        return _HF_TOKEN_USABLE
    tok = _hf_token()
    if not tok:
        _HF_TOKEN_USABLE = False
        return False
    body, err = _http_get_raw(
        "https://huggingface.co/api/whoami-v2",
        timeout=8.0,
        token=tok,
        allow_retry_without_auth=False,
    )
    ok = bool(body) and not err and not body.lstrip().startswith("<")
    _HF_TOKEN_USABLE = ok
    return ok


def _http_get_raw(
    url: str,
    *,
    timeout: float = 20.0,
    token: Optional[str] = None,
    allow_retry_without_auth: bool = True,
    max_bytes: Optional[int] = None,
) -> tuple[Optional[str], Optional[str]]:
    """GET text body. Follows redirects. Returns (body, error)."""
    headers = {"User-Agent": HF_UA, "Accept": "*/*"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            if max_bytes is not None:
                raw = resp.read(max_bytes + 1)
                if len(raw) > max_bytes:
                    return None, f"response exceeds {max_bytes} bytes for {url}"
            else:
                raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            charset = "utf-8"
            if "charset=" in ctype:
                charset = ctype.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
            return raw.decode(charset, errors="replace"), None
    except HTTPError as e:
        # Bad/expired token → 401/403 on public repos; retry anonymous once.
        if (
            allow_retry_without_auth
            and token
            and e.code in (401, 403)
            and "Authorization" in headers
        ):
            global _HF_TOKEN_USABLE
            _HF_TOKEN_USABLE = False
            return _http_get_raw(
                url,
                timeout=timeout,
                token=None,
                allow_retry_without_auth=False,
                max_bytes=max_bytes,
            )
        return None, f"HTTP {e.code} for {url}"
    except (URLError, TimeoutError, OSError) as e:
        return None, f"{type(e).__name__}: {e}"


def _http_get(url: str, timeout: float = 20.0) -> tuple[Optional[str], Optional[str]]:
    """GET with optional HF token; auto-falls back to anonymous on 401/403."""
    tok = _hf_token()
    # Skip known-bad tokens entirely (faster + fewer 401 log lines).
    if tok and _HF_TOKEN_USABLE is False:
        tok = ""
    elif tok and _HF_TOKEN_USABLE is None:
        # Lightweight: try with token first; 401 path marks unusable.
        pass
    return _http_get_raw(url, timeout=timeout, token=tok or None, allow_retry_without_auth=True)



def _quant_config_is_thin(qc: Any) -> bool:
    """True when transformers quantization_config lacks usable method/algo metadata."""
    if not isinstance(qc, dict) or not qc:
        return True
    method = str(qc.get("quant_method") or "").strip()
    algo = str(qc.get("quant_algo") or "").strip()
    if method or algo:
        return False
    if qc.get("quantized_layers") or qc.get("config_groups"):
        return False
    return True


def _normalize_hf_quant_blob(hf_quant: dict[str, Any]) -> dict[str, Any]:
    """Flatten ModelOpt hf_quant_config.json into transformers-style quantization_config."""
    if not isinstance(hf_quant, dict) or not hf_quant:
        return {}
    # Already transformers-shaped (or test fixture).
    if any(k in hf_quant for k in ("quant_method", "quant_algo", "config_groups", "quantized_layers")):
        out = dict(hf_quant)
    elif isinstance(hf_quant.get("quantization"), dict):
        out = dict(hf_quant["quantization"])
    else:
        out = dict(hf_quant)
    producer = hf_quant.get("producer")
    if isinstance(producer, dict):
        pname = str(producer.get("name") or "").lower()
        if pname == "modelopt" and not str(out.get("quant_method") or "").strip():
            out["quant_method"] = "modelopt"
    return out


def _merge_hf_quant_config(
    cfg: dict[str, Any], hf_quant: Optional[dict[str, Any]]
) -> dict[str, Any]:
    """Merge hf_quant_config into config.json, preferring existing quantization_config keys."""
    if not isinstance(cfg, dict):
        return cfg
    out = dict(cfg)
    if not isinstance(hf_quant, dict) or not hf_quant:
        return out
    incoming = _normalize_hf_quant_blob(hf_quant)
    if not incoming:
        return out
    existing = out.get("quantization_config")
    if not isinstance(existing, dict) or not existing:
        out["quantization_config"] = incoming
        return out
    merged = dict(incoming)
    for k, v in existing.items():
        if v is None or v == "":
            continue
        merged[k] = v
    out["quantization_config"] = merged
    return out


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

    # ModelOpt checkpoints often ship quant metadata only in hf_quant_config.json.
    if isinstance(out.get("config"), dict) and _quant_config_is_thin(
        out["config"].get("quantization_config")
    ):
        quant_urls = [
            f"https://huggingface.co/{model_id}/raw/main/hf_quant_config.json",
            f"https://huggingface.co/{model_id}/resolve/main/hf_quant_config.json",
            f"https://huggingface.co/{model_id}/raw/main/quantization_config.json",
            f"https://huggingface.co/{model_id}/resolve/main/quantization_config.json",
        ]
        for url in quant_urls:
            body, err = _http_get(url, timeout=timeout)
            if body:
                try:
                    blob = json.loads(body)
                except json.JSONDecodeError:
                    out["errors"].append(f"invalid JSON from {url}")
                    continue
                if isinstance(blob, dict) and blob:
                    out["config"] = _merge_hf_quant_config(out["config"], blob)
                    out["fetched"].append(url)
                    out["hf_quant_config"] = blob
                    break
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
        if isinstance(cfg, dict) and _quant_config_is_thin(cfg.get("quantization_config")):
            for name in ("hf_quant_config.json", "quantization_config.json"):
                side = p / name
                if side.is_file():
                    cfg = _merge_hf_quant_config(cfg, _read_json(side))
                    break
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
    if isinstance(cfg, dict) and _quant_config_is_thin(cfg.get("quantization_config")):
        for name in ("hf_quant_config.json", "quantization_config.json"):
            side = snap / name
            if side.is_file():
                cfg = _merge_hf_quant_config(cfg, _read_json(side))
                notes.append(f"merged {name} from local cache")
                break
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


def _section_is_spark(sec: str) -> bool:
    s = (sec or "").lower()
    return any(k in s for k in ("dgx spark", "gb10"))


def _section_is_other_silicon(sec: str) -> bool:
    """True when the heading names a non-Spark GPU path (incl. W4A16 Ampere)."""
    s = (sec or "").lower()
    return any(
        k in s
        for k in ("ampere", "w4a16", "a100", "h100", "h200", "gb200", "b200", "rtx", "l40")
    )


def _section_spark_rank(sec: str) -> int:
    """Specificity for de-dup: prefer a later GB10 heading over generic Quick Start."""
    if _section_is_spark(sec):
        return 3
    s = (sec or "").lower()
    if "quick start" in s or "quickstart" in s:
        return 2
    if "vllm" in s and not _section_is_other_silicon(s):
        return 1
    return 0


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

    # docker run … image model args  → recover image + model + args after image
    docker_m = re.search(
        r"docker\s+run\b[^\n]*?\s("
        r"vllm/vllm-openai[^\s]*|"
        r"ghcr\.io/anemll/dspark-vllm-gx10[^\s]*|"
        r"eugr/spark-vllm[^\s]*|"
        r"nvcr\.io/[^\s]+"
        r")\s+(.+)",
        text,
        re.I | re.S,
    )
    if docker_m and "vllm serve" not in text.lower():
        image_ref = docker_m.group(1).strip().strip("'\"")
        try:
            tokens = shlex.split(docker_m.group(2), posix=True)
        except ValueError:
            tokens = docker_m.group(2).split()
        # skip leading "serve" if present (vllm-openai entrypoint is the model)
        if tokens and tokens[0] == "serve":
            tokens = tokens[1:]
        model = tokens[0] if tokens and not tokens[0].startswith("-") else None
        args = tokens[1:] if model else tokens
        # strip "vllm" "serve" if entrypoint style differs
        if args and args[0] == "serve":
            args = args[1:]
        cfg = _args_to_config(args, env)
        if image_ref:
            cfg["image"] = image_ref
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
    out: dict[str, Any] = {"docker_env": _dedupe_env(list(env))}
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
            # Map into structured MTP fields only — do NOT also dump into
            # extra_flags (serve.py would emit --speculative-config twice).
            spec = _parse_speculative_json(cfg)
            method = str(spec.get("method") or "").lower()
            out["mtp"] = method == "mtp" or "mtp" in cfg.lower()
            ntok = spec.get("num_speculative_tokens")
            if ntok is not None:
                try:
                    out["mtp_num_tokens"] = int(ntok)
                except (TypeError, ValueError):
                    pass
            else:
                mm = re.search(r"num_speculative_tokens[\"']?\s*:\s*(\d+)", cfg)
                if mm:
                    out["mtp_num_tokens"] = int(mm.group(1))
            extra = _mtp_spark_extra(spec)
            if extra.get("moe_backend"):
                out["mtp_moe_backend"] = str(extra["moe_backend"])
            if not out.get("mtp"):
                # Non-MTP speculative methods stay as free-form extras.
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


def _parse_speculative_json(raw: str) -> dict[str, Any]:
    """Parse a --speculative-config value into a dict. Empty on garbage."""
    s = (raw or "").strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    return obj if isinstance(obj, dict) else {}


def _mtp_spark_extra(spec: dict[str, Any]) -> dict[str, Any]:
    """Spark playbook keys inside MTP speculative-config (e.g. moe_backend:triton)."""
    if not isinstance(spec, dict):
        return {}
    return {k: v for k, v in spec.items() if k not in ("method", "num_speculative_tokens")}


def _speculative_json_from_extra(extra: str) -> dict[str, Any]:
    """Pull the first --speculative-config JSON object out of extra_flags."""
    s = (extra or "").strip()
    if not s or "speculative-config" not in s:
        return {}
    try:
        parts = shlex.split(s)
    except ValueError:
        parts = s.split()
    for i, p in enumerate(parts):
        if p == "--speculative-config" and i + 1 < len(parts):
            return _parse_speculative_json(parts[i + 1])
        if p.startswith("--speculative-config="):
            return _parse_speculative_json(p.split("=", 1)[1])
    return {}


def _dedupe_env(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = (item or "").strip()
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k:
            continue
        if k in seen:
            out = [e for e in out if not e.startswith(k + "=")]
        seen.add(k)
        out.append(f"{k}={v}")
    return out


# ─── Cluster topology + model-family overlays ─────────────────────────────────

# Anemll GX10 / DGX Spark port of vLLM 0.25 with native DSpark / NVFP4 DS-MLA / b12x.
DSPARK_IMAGE = "ghcr.io/anemll/dspark-vllm-gx10:0.1.1"

# Data-driven model-family overlays. These serve checkpoints whose correct Spark recipe
# is NOT on the HF card (custom runtime image / MoE backend / KV path / speculative method).
# Card is authoritative for normal models; an overlay wins for the families below.
#
# Add future models WITHOUT touching code: drop entries into data/serve_overlays.json
# (same shape as a list entry). File entries override built-ins on key collision.
_BUILTIN_OVERLAYS: list[dict[str, Any]] = [
    {
        "match": {"all": ["deepseek"], "any": ["v4", "dspark"]},
        "family_key": "deepseek_v4_dspark",
        "label": "DeepSeek V4 Flash DSpark (2-node DGX Spark recipe)",
        "source": "https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark",
        "config": {
            "image": DSPARK_IMAGE,
            "quantization": "",  # NVFP4 weights are native; no --quantization flag on Anemll
            "kv_cache_dtype": "nvfp4_ds_mla",
            "moe_backend": "flashinfer_b12x",
            "trust_remote_code": True,
            "tool_call_parser": "deepseek_v4",
            "reasoning_parser": "deepseek_v4",
            "enable_auto_tool_choice": True,
            "max_num_seqs": 6,
            # DSpark runs through --speculative-config (method=dspark) in extra_flags,
            # NOT the structured mtp flag (serve.py would re-emit a conflicting mtp config).
            "mtp": False,
            "mtp_num_tokens": 5,  # checkpoint dspark_block_size is 5; k>=5
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "max_model_len": 1048576,  # 1M ceiling; envelope may lower
            "util": 0.80,
            "docker_env": [
                "VLLM_USE_FLASHINFER_SAMPLER=1",
                "VLLM_USE_BREAKABLE_CUDAGRAPH=0",
                "VLLM_USE_B12X_MOE=1",
                "CUTE_DSL_ARCH=sm_121a",
                "TORCH_CUDA_ARCH_LIST=12.1a",
                "FLASHINFER_CUDA_ARCH_LIST=12.1a",
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1",
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            ],
            "extra_flags": (
                "--tokenizer-mode deepseek_v4 "
                "--speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"draft_sample_method\":\"probabilistic\"}' "
                "--reasoning-config '{\"reasoning_parser\":\"deepseek_v4\",\"reasoning_start_str\":\"<think>\",\"reasoning_end_str\":\"</think>\"}' "
                "--enable-prompt-tokens-details --async-scheduling "
                "--block-size 256 --generation-config vllm --enable-flashinfer-autotune"
            ),
        },
        "rationale": [
            "DeepSeek V4 Flash serves via the DSpark/NVFP4-DS-MLA path, not the stock vLLM image.",
            "Overlay from MiaAI-Lab 2x DGX Spark recipe (Anemll dspark-vllm-gx10 image).",
            "Card's generic recipe (fp8 KV, deep_gemm moe, data-parallel) is wrong for GB10 — overridden.",
        ],
    },
]


def _load_overlays() -> list[dict[str, Any]]:
    """Built-ins + optional user file. File path: $LAIL_DATA_DIR/data/serve_overlays.json."""
    overlays = list(_BUILTIN_OVERLAYS)
    try:
        from ..config import DATA_DIR

        f = DATA_DIR / "serve_overlays.json"
        if f.is_file():
            user = json.loads(f.read_text())
            if isinstance(user, list):
                # user entries override built-ins on family_key collision
                by_key = {o.get("family_key"): o for o in overlays}
                for u in user:
                    if isinstance(u, dict) and u.get("family_key"):
                        by_key[u["family_key"]] = u
                overlays = list(by_key.values())
    except Exception:
        pass
    return overlays


def _overlay_gap_only(overlay: Optional[dict[str, Any]]) -> bool:
    """True when overlay only fills gaps and must not discard card/vendor extras.

    MiniMax / DSpark / playbook overlays own the Spark path (card is known-wrong).
    Parsers-only (Gemma 4) and last-mile fill (Qwen3.8 NVFP4) do not.
    """
    if not overlay:
        return False
    return bool(overlay.get("gap_only")) or overlay.get("family_key") == "qwen38_nvfp4"


def _family_overlay(model: str, detected: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Match a model id (+ optional detected family) against the overlay registry.

    Returns None when the card is authoritative (normal models). Data-driven so
    future models need no code change — drop entries into data/serve_overlays.json.
    """
    mid = (model or "").lower()
    fam = str((detected or {}).get("family") or "").lower()
    for ov in _load_overlays():
        m = ov.get("match") or {}
        all_terms = [str(t).lower() for t in (m.get("all") or [])]
        any_terms = [str(t).lower() for t in (m.get("any") or [])]
        fam_raw = m.get("family") or []
        if isinstance(fam_raw, str):
            fam_raw = [fam_raw]
        fam_terms = [str(t).lower() for t in fam_raw]
        if not (all_terms or any_terms or fam_terms):
            continue
        all_ok = all(t in mid for t in all_terms)
        any_ok = (not any_terms) or any(t in mid for t in any_terms)
        # If detected.family is unknown, fall back to id substrings alone.
        fam_ok = (not fam_terms) or (not fam) or (fam in fam_terms)
        if all_ok and any_ok and fam_ok:
            out = dict(ov)
            # MiniMax M2 ships FP8 and NVFP4 weights; label must not always claim NVFP4.
            if out.get("family_key") == "minimax_m2":
                out["label"] = _minimax_m2_overlay_label(mid)
            return out
    return None


def _minimax_m2_overlay_label(mid: str) -> str:
    """Quant-aware MiniMax M2 overlay label (mid already lowercased)."""
    if "nvfp4" in mid:
        return "MiniMax M2 NVFP4 (DGX Spark)"
    if re.search(r"(^|[^a-z0-9])fp8([^a-z0-9]|$)", mid):
        return "MiniMax M2 FP8 (DGX Spark)"
    return "MiniMax M2 (DGX Spark)"


def _cluster_topology() -> dict[str, Any]:
    """Live cluster shape for serve planning. Never raises — falls back to probed local host."""
    def fallback() -> dict[str, Any]:
        local = _local_hw_fallback_node()
        return {
            "nodes": 1,
            "node_list": [local],
            "head": local,
            "workers": [],
            "fabric_ok": False,
            "available": False,
        }

    try:
        from . import cluster as _cluster

        data = _cluster.collect_cluster()
        nodes = data.get("nodes") or []
        online = [n for n in nodes if n.get("state") != "offline" and (n.get("online") or n.get("local"))]
        if not online:
            return fallback()
        head = next((n for n in online if n.get("local")), online[0])
        workers = [n for n in online if n is not head]
        return {
            "nodes": len(online),
            "node_list": online,
            "head": head,
            "workers": workers,
            "fabric_ok": bool((data.get("fabric") or {}).get("ok")),
            "available": True,
        }
    except Exception:
        return fallback()


# ─── Placement engine (hardware-aware, N-node, model-agnostic) ────────────────

# Per-GPU arch hints for compile targets. Unknown / empty skus omit arch env.
_SKU_ARCH = {
    "gb10": {"cute_dsl_arch": "sm_121a", "torch_arch": "12.1a"},
    "spark": {"cute_dsl_arch": "sm_121a", "torch_arch": "12.1a"},
    "gb200": {"cute_dsl_arch": "sm_100a", "torch_arch": "12.0a"},
    "gb300": {"cute_dsl_arch": "sm_103a", "torch_arch": "12.0a"},
}
# Laptop-scale only when collect_hardware() fails — never assume GB10 121.7 UMA.
_CONSERVATIVE_NODE_RAM_GIB = 32.0


def _weight_floor_gib(model: str, hf_config: Optional[dict] = None) -> Optional[float]:
    """Conservative minimum GiB for families that never fit 1–2× GB10 UMA.

    Used **only** when Hub blobs / index / config heuristic all fail (or return
    nothing) so placement cannot claim fits=True for DeepSeek-V3/R1/Pro-class
    checkpoints. Never applied as max() over a real blob sum — compact MoEs like
    Nemotron 30B-A3B NVFP4 (~20 GiB) have 128 experts and must not be floored to 400.
    """
    mid = (model or "").lower()
    # DeepSeek V4 Flash has a Spark overlay (~155 GiB) — do not floor that path.
    if "deepseek" in mid and "v4" in mid and "pro" in mid:
        return 900.0
    if "deepseek" in mid and ("r1" in mid or re.search(r"v3(\.|$|-)", mid)) and "v4" not in mid:
        return 700.0
    if "minimax" in mid and re.search(r"(^|[^a-z])m3([^0-9]|$)", mid):
        return 800.0
    if ("kimi" in mid or "moonshot" in mid) and re.search(r"k[23]", mid):
        return 800.0
    if "llama" in mid and ("405b" in mid or "maverick" in mid):
        return 400.0
    # Full GLM 4.5/4.6/4.7 MoE (not 9b / edge / air distillations).
    if re.search(r"glm-?4\.[567]", mid) and not re.search(r"(9b|air|edge|flash)", mid):
        return 500.0
    # Step-3 classic VLM (not 3.5 Flash). Vendor: FP8 ≈ 326 GiB.
    if re.search(r"step[\s._-]*3(?![\s._-]*[57])", mid) and "flash" not in mid:
        return 320.0
    # HY-v3 / Hy3-preview ~556 GiB BF16.
    if re.search(r"(^|[^a-z])hy[\s._-]*v?3", mid) or "hy3" in mid or "hunyuan-v3" in mid:
        if "a13b" not in mid:
            return 550.0
    # ERNIE 300B-A47B class — 21B-A3B must not match.
    if "ernie" in mid and re.search(r"300b|a47b", mid):
        return 400.0
    cfg = hf_config or {}
    experts = cfg.get("n_routed_experts") or cfg.get("num_experts") or cfg.get("num_local_experts")
    try:
        n_exp = int(experts) if experts is not None else 0
    except (TypeError, ValueError):
        n_exp = 0
    if n_exp >= 64:
        # Compact MoE naming: 30B-A3B / 35B-A3B / 8x7B — expert count alone is not size.
        if re.search(r"\d+(\.\d+)?b-a\d+(\.\d+)?b", mid):
            return None
        if re.search(r"\d+x\d+b", mid):  # Mixtral-style 8x7B
            return None
        # Small total-param ids (≤40B class) even with many experts (NVFP4 fits 1 Spark).
        m_tot = re.search(r"(^|[^0-9])(\d{1,2})b([^a-z0-9]|$)", mid)
        if m_tot and int(m_tot.group(2)) <= 40:
            return None
        # Huge MoE without a Hub blob sum — refuse optimistic single-node fit.
        return 400.0
    return None


# Spark GGUF preference (hardware default, not a model recipe). Unsloth / ggml
# Spark benches land on UD-Q4_K_XL — quality that still leaves UMA for long ctx.
_SPARK_GGUF_QUANT_PREF = (
    "UD-Q4_K_XL",
    "Q4_K_XL",
    "UD-Q4_K_M",
    "Q4_K_M",
    "UD-Q5_K_XL",
    "Q5_K_M",
    "UD-Q4_K_S",
    "Q4_K_S",
    "MXFP4_MOE",
    "UD-IQ4_NL_XL",
    "UD-IQ4_NL",
    "UD-IQ4_XS",
    "UD-Q3_K_XL",
    "Q3_K_M",
    "UD-Q6_K_XL",
    "Q6_K",
    "Q8_0",
    "UD-Q8_K_XL",
    "BF16",
)
_GGUF_SPLIT_RE = re.compile(r"-\d{5}-of-\d{5}$", re.I)


def _hf_repo_ref(model: str) -> tuple[str, Optional[str]]:
    """Split ``org/name:QUANT`` from a Hub id. Local paths are returned unchanged."""
    m = (model or "").strip()
    if not m:
        return m, None
    if m.startswith("/") or m.endswith(".gguf") or Path(m).is_dir():
        return m, None
    if ":" in m:
        left, right = m.rsplit(":", 1)
        if "/" in left and right and "/" not in right and not right.startswith("//"):
            return left, right
    return m, None


def _gguf_is_sidecar(name: str) -> bool:
    """Vision projector / imatrix / isolated MTP head — not the served weights."""
    base = str(name or "").split("/")[-1].lower()
    if not base.endswith(".gguf"):
        return False
    if "mmproj" in base or "imatrix" in base:
        return True
    if base.startswith("mtp") or "-mtp." in base or base.endswith("-mtp.gguf"):
        return True
    return False


def _gguf_quant_tag(name: str) -> str:
    """Extract ``UD-Q4_K_XL`` / ``Q4_K_M`` from a GGUF filename."""
    stem = str(name or "").split("/")[-1]
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    stem = _GGUF_SPLIT_RE.sub("", stem)
    up = stem.upper()
    for tag in sorted(_SPARK_GGUF_QUANT_PREF, key=len, reverse=True):
        if tag.upper() in up:
            return tag
    m = re.search(r"(UD-)?((?:IQ|Q|BF|F|MXFP)\d+[A-Z0-9_]*)", stem, re.I)
    return m.group(0) if m else stem


def _gguf_siblings_have_mtp(model: str, siblings: list) -> bool:
    if "mtp" in (model or "").lower():
        return True
    for f in siblings or []:
        n = str((f or {}).get("rfilename") or "").lower()
        if "mtp" in n:
            return True
    return False


def pick_spark_gguf(
    siblings: list,
    *,
    preferred: Optional[str] = None,
) -> dict[str, Any]:
    """Pick one Spark quant from a multi-quant GGUF repo (sum shards of that quant only)."""
    groups: dict[str, list] = {}
    for f in siblings or []:
        if not isinstance(f, dict):
            continue
        n = str(f.get("rfilename") or "")
        if not n.endswith(".gguf") or _gguf_is_sidecar(n):
            continue
        tag = _gguf_quant_tag(n)
        groups.setdefault(tag, []).append(f)
    if not groups:
        tot = 0
        for f in siblings or []:
            if isinstance(f, dict) and str(f.get("rfilename") or "").endswith(".gguf"):
                tot += int(f.get("size") or 0)
        return {"quant": preferred, "bytes": tot, "files": []}

    pref_u = (preferred or "").upper()

    def _rank(tag: str) -> tuple[int, int]:
        tu = tag.upper()
        if pref_u and (pref_u == tu or pref_u in tu or tu in pref_u):
            return (0, 0)
        try:
            return (1, _SPARK_GGUF_QUANT_PREF.index(tag))
        except ValueError:
            return (2, 99)

    tag = min(groups, key=_rank)
    chosen = groups[tag]
    tot = sum(int(f.get("size") or 0) for f in chosen)
    return {"quant": tag, "bytes": tot, "files": chosen}


_SHARD_WEIGHT_RE = re.compile(
    r"(?:^|/)(?:model|pytorch_model)-\d+-of-\d+\.(safetensors|bin)$",
    re.I,
)
_CONSOLIDATED_WEIGHT_RE = re.compile(
    r"(?:^|/)consolidated(?:\.\d+)?\.(safetensors|bin|pth)$",
    re.I,
)
_SINGLE_WEIGHT_RE = re.compile(
    r"(?:^|/)(?:model|pytorch_model)\.(safetensors|bin)$",
    re.I,
)


def _select_weight_blobs(siblings: list) -> list[dict]:
    """Pick one on-disk dump of the same tensors.

    Mistral / Magistral Hub repos often ship ``consolidated.safetensors``
    (mistral-common) *and* ``model-*-of-*`` shards (transformers) of the
    same BF16. Summing both doubles the weight (∼44.7 → 89.4 GiB) and
    falsely serve_blocks a 1-Spark node.
    """
    files: list[dict] = []
    for f in siblings or []:
        if not isinstance(f, dict):
            continue
        n = str(f.get("rfilename") or "")
        if n.endswith((".safetensors", ".bin")):
            files.append(f)
    if not files:
        return []

    def _name(f: dict) -> str:
        return str(f.get("rfilename") or "")

    def _prefer_safetensors(group: list[dict]) -> list[dict]:
        st = [f for f in group if _name(f).endswith(".safetensors")]
        return st or group

    shards = [f for f in files if _SHARD_WEIGHT_RE.search(_name(f))]
    cons = [f for f in files if _CONSOLIDATED_WEIGHT_RE.search(_name(f))]
    singles = [f for f in files if _SINGLE_WEIGHT_RE.search(_name(f))]
    extras = [
        f
        for f in files
        if f not in shards and f not in cons and f not in singles
    ]
    if shards:
        primary = _prefer_safetensors(shards)
    elif cons:
        primary = _prefer_safetensors(cons)
    elif singles:
        primary = _prefer_safetensors(singles)
    else:
        return _prefer_safetensors(files)
    return primary + extras


def _hub_blob_measure(model: str) -> Optional[dict[str, Any]]:
    """Hub blob sizes. Safetensors: one weight dump (not consolidated+shards). GGUF: one Spark quant."""
    hub_id, want_quant = _hf_repo_ref(model)
    if not hub_id or hub_id.startswith("/") or Path(hub_id).is_dir():
        return None
    try:
        body, err = _http_get(
            f"https://huggingface.co/api/models/{hub_id}?blobs=true", timeout=20.0
        )
        if not body or err:
            return None
        d = json.loads(body)
    except Exception:
        return None
    siblings = list(d.get("siblings") or [])
    st: list[dict] = []
    gg: list[dict] = []
    for f in siblings:
        if not isinstance(f, dict):
            continue
        n = str(f.get("rfilename") or "")
        if n.endswith((".safetensors", ".bin")):
            st.append(f)
        elif n.endswith(".gguf"):
            gg.append(f)
    if st:
        tot = sum(int(f.get("size") or 0) for f in _select_weight_blobs(st))
        if tot <= 0:
            return None
        return {
            "source": "blobs",
            "kind": "safetensors",
            "bytes": tot,
            "gib": round(tot / (1024**3), 1),
            "quant": None,
            "siblings": siblings,
            "mtp": False,
        }
    if gg:
        picked = pick_spark_gguf(gg, preferred=want_quant)
        tot = int(picked.get("bytes") or 0)
        if tot <= 0:
            return None
        return {
            "source": "blobs",
            "kind": "gguf",
            "bytes": tot,
            "gib": round(tot / (1024**3), 1),
            "quant": picked.get("quant") or want_quant,
            "siblings": siblings,
            "mtp": _gguf_siblings_have_mtp(hub_id, siblings),
        }
    return None


def estimate_weights_gib(model: str, hf_config: Optional[dict]) -> Optional[float]:
    """Best-effort weight size for ANY model. Order: exact HF blob sum → safetensors
    index total_size → config param estimate → known-family floor → None.
    Used by the placement engine to decide nodes_needed + util.

    Hub blob / index measurements are authoritative — family floors only fill total
    absence (never max'd over a real ~20 GiB NVFP4 sum).

    Safetensors count **one** on-disk dump (shards *or* consolidated, not both).
    Multi-quant GGUF repos (dozens of alternative files) count **one** Spark quant,
    not the sum of every sibling.
    """
    measured: Optional[float] = None
    source: Optional[str] = None
    # 1) Exact: Hub blobs. Safetensors/bin shards are summed; GGUF picks one quant.
    try:
        blob = _hub_blob_measure(model)
        if blob and blob.get("gib"):
            measured = float(blob["gib"])
            source = "blobs"
    except Exception:
        pass
    # 2) Safetensors index metadata.total_size (when blob sizes are missing).
    if measured is None:
        try:
            body, err = _http_get(
                f"https://huggingface.co/{model}/resolve/main/model.safetensors.index.json",
                timeout=20.0,
            )
            if body and not err:
                d = json.loads(body)
                tot = (d.get("metadata") or {}).get("total_size")
                if isinstance(tot, (int, float)) and tot > 0:
                    measured = round(float(tot) / (1024**3), 1)
                    source = "index"
        except Exception:
            pass
    # 3) Estimate from config: params × bytes/param (quant-aware; MoE-aware).
    if measured is None:
        try:
            cfg0 = hf_config or {}
            text = cfg0.get("text_config") if isinstance(cfg0.get("text_config"), dict) else {}
            qc = cfg0.get("quantization_config") or {}
            if not isinstance(qc, dict):
                qc = {}
            bit_widths: list[int] = []
            for g in (qc.get("config_groups") or {}).values():
                if isinstance(g, dict):
                    w = g.get("weights") or {}
                    if isinstance(w, dict) and w.get("num_bits"):
                        bit_widths.append(int(w["num_bits"]))
            qlayers = qc.get("quantized_layers") or {}
            if isinstance(qlayers, dict):
                for meta in qlayers.values():
                    if not isinstance(meta, dict):
                        continue
                    algo = str(meta.get("quant_algo") or "").upper()
                    if any(t in algo for t in ("NVFP4", "FP4", "W4A16")):
                        bit_widths.append(4)
                    elif "FP8" in algo:
                        bit_widths.append(8)
            algo_u = str(qc.get("quant_algo") or "").upper()
            if any(t in algo_u for t in ("NVFP4", "FP4", "W4A16", "MIXED")):
                bit_widths.append(4)
            elif algo_u == "FP8" or algo_u.startswith("FP8"):
                bit_widths.append(8)
            # Mixed ModelOpt (FP8 + NVFP4): expert / NVFP4 weights dominate → min bits.
            nbits = min(bit_widths) if bit_widths else 0
            mid = (model or "").lower()
            if nbits not in (4, 8):
                if "nvfp4" in mid or "fp4" in mid or "w4a16" in mid:
                    nbits = 4
                elif re.search(r"(^|[-_/])fp8($|[-_/])", mid):
                    nbits = 8
            hidden = cfg0.get("hidden_size") or text.get("hidden_size")
            layers = cfg0.get("num_hidden_layers") or text.get("num_hidden_layers")
            if hidden and layers:
                # Hybrid (Nemotron): only MoE block types hold expert FFNs.
                block_types = (
                    (hf_config or {}).get("layers_block_type")
                    or (hf_config or {}).get("layer_types")
                    or []
                )
                moe_layers = layers
                if isinstance(block_types, list) and block_types:
                    n_moe = sum(
                        1
                        for t in block_types
                        if "moe" in str(t).lower() or str(t).lower() in ("e", "expert")
                    )
                    if n_moe > 0:
                        moe_layers = n_moe
                # Dense-ish attention/mamba: ~12 * layers * hidden^2 (+emb)
                params = 12 * layers * hidden * hidden
                experts = (
                    cfg0.get("n_routed_experts")
                    or cfg0.get("num_experts")
                    or cfg0.get("num_local_experts")
                    or text.get("n_routed_experts")
                    or text.get("num_experts")
                    or text.get("num_local_experts")
                )
                moe_inter = (
                    cfg0.get("moe_intermediate_size")
                    or cfg0.get("intermediate_size")
                    or text.get("moe_intermediate_size")
                    or text.get("intermediate_size")
                )
                if experts and moe_inter:
                    # MoE FFN dominates; dense formula alone under-reports ~0 GiB-class misses.
                    params += int(experts) * moe_layers * 3 * hidden * int(moe_inter)
                bpp = (nbits / 8.0) if nbits in (4, 8) else 2.0  # fp4/fp8/bf16
                measured = round(params * bpp / (1024**3), 1)
                source = "heuristic"
        except Exception:
            pass
    floor = _weight_floor_gib(model, hf_config)
    # Hub blobs / index are ground truth — never raise them to a family floor.
    if measured is not None and source in ("blobs", "index"):
        return measured
    # Heuristic may under-report huge named families; floor only then.
    if measured is not None and floor is not None and source == "heuristic":
        return max(measured, floor)
    if measured is not None:
        return measured
    return floor


def _probed_local_hardware() -> dict[str, Any]:
    """Live host probe. Never invents GB10 UMA — empty dict on failure."""
    try:
        from . import metadata as _metadata

        hw = _metadata.collect_hardware() or {}
        return hw if isinstance(hw, dict) else {}
    except Exception:
        return {}


def _resolved_node_ram_gib(value: Any = None) -> float:
    """Explicit ram_gib → live MemTotal → conservative laptop-scale. Never 121.7."""
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    hw = _probed_local_hardware()
    for key in ("ram_gib", "memory_capacity_gib"):
        ram = hw.get(key)
        if isinstance(ram, (int, float)) and ram > 0:
            return float(ram)
    return _CONSERVATIVE_NODE_RAM_GIB


def _node_ram_gib(node: Optional[dict]) -> float:
    ram = node.get("ram_gib") if node else None
    return _resolved_node_ram_gib(ram)


def _local_hw_fallback_node() -> dict[str, Any]:
    """Single local node from collect_hardware() when cluster topology is unavailable."""
    hw = _probed_local_hardware()
    ram = hw.get("ram_gib")
    if not (isinstance(ram, (int, float)) and ram > 0):
        ram = hw.get("memory_capacity_gib")
    if not (isinstance(ram, (int, float)) and ram > 0):
        ram = None
    sku = hw.get("gpu_sku")
    if sku and str(sku).lower() == "unknown":
        sku = None
    host = hw.get("hostname")
    return {
        "id": "local",
        "name": host or "local",
        "local": True,
        "online": True,
        "ram_gib": float(ram) if isinstance(ram, (int, float)) and ram > 0 else None,
        "gpu_sku": sku,
        "hostname": host,
    }


def _gpu_arch_env(nodes: list[dict]) -> dict[str, str]:
    """Compile-target env only when a probed sku matches _SKU_ARCH. Unknown → {}."""
    sku = ""
    for n in nodes:
        sku = (n.get("gpu_sku") or "").lower()
        if sku:
            break
    if not sku:
        return {}
    key = next((k for k in _SKU_ARCH if k in sku), None)
    if key is None:
        return {}
    a = _SKU_ARCH[key]
    return {
        "CUTE_DSL_ARCH": a["cute_dsl_arch"],
        "TORCH_CUDA_ARCH_LIST": a["torch_arch"],
        "FLASHINFER_CUDA_ARCH_LIST": a["torch_arch"],
    }


def plan_placement(
    weights_gib: Optional[float],
    topology: dict[str, Any],
    *,
    mode: str | None = None,
    overlay: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Decide nodes_needed / TP / PP / per-node util from real weights + probed hardware.

    One rule (UMA): fit weights on the fewest nodes that can hold them at the
    0.85 util ceiling after reserved RAM. ``util_computed`` is set only when
    weights need *more* than the default (envelope fills 0.85 otherwise).
    Overlay may still pin util later. ``mode`` is ignored (legacy).
    """
    del mode  # dual user envelopes are gone; reserved RAM is the safety
    nodes = topology.get("node_list") or []
    n_avail = max(1, int(topology.get("nodes") or 1))
    head = topology.get("head") or (nodes[0] if nodes else None)
    node_ram = _node_ram_gib(head)
    reserve = _UMA_RESERVE_GIB  # OS + Hermes + runtime headroom per node (lab rule)

    # Usable weights capacity per node at the util ceiling.
    util_cap = 0.85
    mode_default_util = util_cap
    single_fit_gib = node_ram * util_cap - reserve

    if weights_gib and weights_gib > 0:
        nodes_needed = 1
        while nodes_needed < n_avail and (weights_gib / nodes_needed) > single_fit_gib:
            nodes_needed += 1
        fits = (weights_gib / nodes_needed) <= (node_ram * 0.85 - reserve)
        per_node_weights = weights_gib / nodes_needed
        min_to_load = (per_node_weights + reserve) / node_ram if node_ram > 0 else 0.0
        # Only elevate util above the mode envelope when weights require it.
        if min_to_load > mode_default_util + 1e-9:
            util = min(0.85, min_to_load)
        else:
            util = None
    else:
        nodes_needed, fits, per_node_weights, util = 1, True, None, None

    tp = nodes_needed
    pp = 1  # single-GPU-per-node UMA: tensor-parallel across nodes, no pipeline split
    planned = nodes[:nodes_needed]
    used_workers = [n for n in planned if n is not head and not n.get("local")]

    return {
        "nodes_available": n_avail,
        "nodes_needed": nodes_needed,
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "per_node_weights_gib": per_node_weights,
        "weights_gib": weights_gib,
        "util_computed": (round(util, 2) if util else None),
        "fits": fits,
        "head": head,
        "planned_nodes": planned,
        "workers": used_workers,
        "node_ram_gib": node_ram,
        "reserve_gib": reserve,
    }


def _ib_hca_for_iface(iface: str) -> Optional[str]:
    """Map a QSFP netdev (enp1s0f1np1) to its RoCE HCA (rocep1s0f1).

    GB10 exposes 4 HCAs, 2 of them DOWN. Without NCCL_IB_HCA pinned, NCCL picks a
    dead one and dies with 'unhandled system error' at init_device.
    """
    try:
        import glob as _glob
        import os as _os

        for netpath in _glob.glob("/sys/class/infiniband/*/device/net/*"):
            if _os.path.basename(netpath) == iface:
                hca = netpath.split("/sys/class/infiniband/")[1].split("/")[0]
                state = ""
                try:
                    with open(f"/sys/class/infiniband/{hca}/ports/1/state") as fh:
                        state = fh.read().strip()
                except Exception:
                    pass
                if "ACTIVE" in state.upper() or not state:
                    return hca
    except Exception:
        pass
    return None


def _apply_topology(
    cfg: dict[str, Any],
    *,
    overlay: Optional[dict[str, Any]],
    topology: dict[str, Any],
    weights_gib: Optional[float],
    mode: str,
    warnings: list[str],
    rationale: list[str],
) -> None:
    """Compute TP / nnodes / fabric env from the placement plan (live cluster + weights)."""
    plan = plan_placement(weights_gib, topology, mode=mode, overlay=overlay)
    n = int(plan["nodes_needed"])
    head = plan.get("head") or {}
    workers = plan.get("workers") or []
    fabric_ok = bool(topology.get("fabric_ok"))
    multi = n >= 2

    # Strip any card-supplied data-parallel — DP is a multi-node recipe artifact that
    # does not apply to a 1/2-GB10 UMA cluster (tensor-parallel is the correct split).
    if cfg.get("extra_flags"):
        cfg["extra_flags"] = _strip_flag_from_extra(cfg["extra_flags"], "--data-parallel-size")

    # Cards write the model into their own recipe, often as an unexpanded shell var
    # (`--model $MODEL_CKPT`). serve.py passes the model positionally, so carrying
    # --model through would serve a checkpoint literally named "$MODEL_CKPT".
    if cfg.get("extra_flags"):
        cfg["extra_flags"] = _strip_flag_from_extra(cfg["extra_flags"], "--model")

    # GPU arch compile-target env for the detected sku (only when unset).
    arch_env = _gpu_arch_env(topology.get("node_list") or [])
    env0 = list(cfg.get("docker_env") or [])
    for k, v in arch_env.items():
        if not any(e.startswith(k + "=") for e in env0):
            env0.append(f"{k}={v}")
    cfg["docker_env"] = _dedupe_env(env0)

    # util: overlay pins it (recipe authoritative); else compute from weights.
    if cfg.get("util") is None and plan.get("util_computed"):
        cfg["util"] = plan["util_computed"]
        rationale.append(
            f"util={plan['util_computed']} computed from weights "
            f"({plan.get('per_node_weights_gib')} GiB/node + {int(plan['reserve_gib'])} GiB headroom "
            f"on {plan['node_ram_gib']:.0f} GiB nodes)"
        )

    if not plan.get("fits"):
        warnings.append(
            f"Model weights (~{weights_gib} GiB) exceed the {plan['nodes_available']}-node "
            f"cluster's usable UMA even spread across all nodes. Likely will not load — "
            "add nodes or pick a smaller checkpoint."
        )

    if not multi:
        # Single node: TP=1, remove multi-node residue.
        cfg.pop("tensor_parallel_size", None)
        for f in ("--nnodes", "--node-rank", "--master-addr", "--master-port", "--pipeline-parallel-size"):
            if cfg.get("extra_flags"):
                cfg["extra_flags"] = _strip_flag_from_extra(cfg["extra_flags"], f)
        cfg["docker_env"] = [
            e
            for e in (cfg.get("docker_env") or [])
            if not e.startswith(
                ("VLLM_HOST_IP=", "WORKER_VLLM_HOST_IP=", "NCCL_", "TP_SOCKET_IFNAME=", "GLOO_SOCKET_IFNAME=", "MASTER_ADDR=", "NODE_RANK=")
            )
        ]
        if overlay and (topology.get("nodes") or 1) >= 2:
            rationale.append(
                f"Placement: weights fit one node ({plan.get('per_node_weights_gib') or '?'} GiB) "
                "→ single-node TP=1 (multi-node available but not needed)"
            )
        elif overlay:
            warnings.append(
                f"{overlay['label']} is a multi-node recipe, but only 1 Spark is online — "
                "serving single-node (TP=1). Multi-node flags + fabric env dropped."
            )
        else:
            rationale.append("Placement: 1 node → TP=1 (single-node serve)")
        cfg["topology_plan"] = plan
        return

    # Multi-node: TP across the QSFP RoCE fabric, sized to the nodes actually needed.
    tp = plan["tensor_parallel_size"]
    cfg["tensor_parallel_size"] = tp
    head_if = head.get("qsfp_if") or "enp1s0f1np1"
    head_ip = head.get("qsfp_ip")
    worker_ips = [w.get("qsfp_ip") for w in workers if w.get("qsfp_ip")]

    dist_flags = (
        f"--tensor-parallel-size {tp} --pipeline-parallel-size {plan['pipeline_parallel_size']} --nnodes {n} "
        f"--node-rank 0 --master-addr {head_ip or 'HEAD_ROCE_IP'} --master-port 25000 "
        f"--distributed-executor-backend mp"
    )
    existing = cfg.get("extra_flags") or ""
    cfg["extra_flags"] = (existing + " " + dist_flags).strip()

    env = list(cfg.get("docker_env") or [])
    if head_ip:
        env.append(f"VLLM_HOST_IP={head_ip}")
    if worker_ips:
        env.append(f"WORKER_VLLM_HOST_IP={','.join(worker_ips)}")
    env += [
        f"NCCL_SOCKET_IFNAME={head_if}",
        f"TP_SOCKET_IFNAME={head_if}",
        f"GLOO_SOCKET_IFNAME={head_if}",
        "NCCL_NET=IB",
        "NCCL_IB_DISABLE=0",
        "NCCL_CROSS_NIC=1",
        "NCCL_NVLS_ENABLE=0",
    ]
    # Pin the RoCE HCA: GB10 exposes 4 HCAs (2 DOWN). Unpinned, NCCL picks a dead
    # one and fails init_device with "unhandled system error".
    hca = _ib_hca_for_iface(head_if)
    if hca:
        env.append(f"NCCL_IB_HCA={hca}")
        rationale.append(f"NCCL_IB_HCA={hca} pinned (RoCE HCA for {head_if}; other HCAs are DOWN)")
    else:
        warnings.append(
            f"Could not resolve the RoCE HCA for {head_if}. If NCCL fails with "
            "'unhandled system error', set NCCL_IB_HCA manually in docker env."
        )
    cfg["docker_env"] = _dedupe_env(env)

    wtxt = f"~{weights_gib} GiB weights" if weights_gib else "weights unknown"
    rationale.append(
        f"Placement: {wtxt} over {plan['node_ram_gib']:.0f} GiB nodes → needs {n} node(s); "
        f"TP={tp} across QSFP RoCE ({head_if})"
        + (f"; head={head_ip}" if head_ip else "")
        + (f", workers={','.join(worker_ips)}" if worker_ips else "")
    )
    if not fabric_ok:
        warnings.append(
            "Multi-node serve planned but the QSFP RoCE fabric check did not pass. "
            "Verify enp1s0f1np1 carrier + 10.100.8.x reachability on all nodes before Start."
        )
    if not head_ip or len(worker_ips) < (n - 1):
        warnings.append(
            "RoCE IPs not fully discovered — VLLM_HOST_IP / WORKER_VLLM_HOST_IP may need "
            "manual entry (10.100.8.x on this lab)."
        )
    cfg["topology_plan"] = plan


def _strip_flag_from_extra(extra: str, flag: str) -> str:
    """Remove ``flag`` and its value from a free-form extra_flags string."""
    s = (extra or "").strip()
    if not s or flag not in s:
        return s
    try:
        parts = shlex.split(s)
    except ValueError:
        parts = s.split()
    out: list[str] = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if p == flag:
            # skip flag + following value if present
            if i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
            continue
        if p.startswith(flag + "="):
            i += 1
            continue
        out.append(p)
        i += 1
    return " ".join(shlex.quote(x) if (" " in x or "{" in x) else x for x in out)


def _merge_extra_flags(existing: str, add: str) -> str:
    """Append ``add`` onto ``existing``, dropping any flag already present (first wins).

    Used when family overlays merge onto card/fill extras so tokenizer/config-format
    (and any other flag) never appear twice in composed argv.
    """
    existing = (existing or "").strip()
    add = (add or "").strip()
    if not add:
        return existing
    if not existing:
        return add
    try:
        exist_parts = shlex.split(existing)
    except ValueError:
        exist_parts = existing.split()
    try:
        add_parts = shlex.split(add)
    except ValueError:
        add_parts = add.split()

    def _consume(parts: list[str], i: int) -> tuple[str, list[str], int]:
        """Return (flag_key, tokens_for_this_flag, next_index)."""
        p = parts[i]
        key = p.split("=", 1)[0]
        if "=" in p:
            return key, [p], i + 1
        if i + 1 < len(parts) and not str(parts[i + 1]).startswith("-"):
            return key, [p, parts[i + 1]], i + 2
        return key, [p], i + 1

    seen: set[str] = set()
    out: list[str] = []
    i = 0
    while i < len(exist_parts):
        p = exist_parts[i]
        if p.startswith("-"):
            key, toks, i = _consume(exist_parts, i)
            if key in seen:
                continue
            seen.add(key)
            out.extend(toks)
        else:
            out.append(p)
            i += 1
    i = 0
    while i < len(add_parts):
        p = add_parts[i]
        if p.startswith("-"):
            key, toks, i = _consume(add_parts, i)
            if key in seen:
                continue
            seen.add(key)
            out.extend(toks)
        else:
            out.append(p)
            i += 1
    return " ".join(shlex.quote(x) if (" " in x or "{" in x) else x for x in out)


def _resolve_dspark_draft_model(readme: str | None) -> str | None:
    """Best-effort draft checkpoint id from card exports / prose."""
    if not readme:
        return None
    m = re.search(r"(?:export\s+)?DSPARK_CKPT=(\S+)", readme)
    if m:
        val = m.group(1).strip().rstrip("\\").strip("'\"")
        if val and not val.startswith("$"):
            return val
    m = re.search(r"(nvidia/[A-Za-z0-9._/-]+DSpark[A-Za-z0-9._/-]*)", readme)
    if m:
        return m.group(1).rstrip("/")
    return None


def _parse_card_exports(readme: str | None) -> dict[str, str]:
    """Collect ``export FOO=bar`` / ``FOO=bar`` assignments from a model card."""
    out: dict[str, str] = {}
    if not readme:
        return out
    for m in re.finditer(
        r"(?:^|\n)\s*(?:export\s+)?([A-Z][A-Z0-9_]{1,64})=([^\s\\#'\"]+)",
        readme,
    ):
        key, val = m.group(1), m.group(2).strip().rstrip("\\")
        if val and not val.startswith("$"):
            out[key] = val
    return out


def _expand_card_exports(extra: str, readme: str | None) -> str:
    """Substitute ``$VAR`` / ``${VAR}`` in extra_flags from card export lines.

    NVIDIA cards write ``export DSPARK_CKPT=nvidia/…-DSpark`` then
    ``--speculative_config.model $DSPARK_CKPT``. Expanding before scrub keeps
    the draft path instead of dropping the only token that contains ``dspark``.
    """
    s = (extra or "").strip()
    if not s or "$" not in s:
        return s
    exports = _parse_card_exports(readme)
    if not exports:
        return s

    def repl(m: re.Match[str]) -> str:
        name = m.group(1) or m.group(2)
        return exports.get(name, m.group(0))

    return re.sub(r"\$\{([A-Z][A-Z0-9_]*)\}|\$([A-Z][A-Z0-9_]*)", repl, s)


def _dspark_spec_present(extra: str) -> bool:
    """True when extras look like NVIDIA DSpark (or residual after $VAR scrub)."""
    ex = (extra or "").lower()
    has_spec = "speculative_config" in ex or "speculative-config" in ex
    if not has_spec:
        return False
    if "dspark" in ex:
        return True
    # Live Spark recipes often omit method=dspark; after scrubbing $DSPARK_CKPT
    # only .num_speculative_tokens / empty .model remain — still DSpark intent
    # when the card pairs them (caller also checks readme draft resolve).
    if re.search(r"speculative_config\.(model|num_speculative_tokens|method)\b", ex):
        return True
    return False


def _spec_has_draft_model(extra: str) -> bool:
    ex = extra or ""
    if re.search(r"--speculative_config\.model\s+[^\s$]+", ex):
        return True
    # JSON --speculative-config {"method":"dspark","model":"org/name",...}
    low = ex.lower()
    if "speculative-config" in low and '"model"' in low:
        # reject leftover shell vars
        if "$" in ex[ex.lower().find("speculative-config") : ex.lower().find("speculative-config") + 200]:
            return False
        return True
    return False


def _strip_dspark_speculative(extra: str) -> str:
    """Remove DSpark / speculative_config.* tokens after a failed draft resolve."""
    s = (extra or "").strip()
    if not s:
        return s
    s = _strip_flag_from_extra(s, "--speculative-config")
    for flag in (
        "--speculative_config.method",
        "--speculative_config.model",
        "--speculative_config.num_speculative_tokens",
        "--speculative_config.draft_sample_method",
    ):
        s = _strip_flag_from_extra(s, flag)
    return s.strip()


def _ensure_dspark_draft_or_strip(
    cfg: dict[str, Any],
    readme: str | None,
    warnings: list[str],
    rationale: list[str],
) -> None:
    """P0.6: DSpark speculative needs a draft model — resolve from card or strip.

    Anemll/DSpark runtime images run method=dspark natively without a separate
    HF draft id — do not strip those overlays.
    """
    img = (cfg.get("image") or "").lower()
    if "anemll" in img or "dspark-vllm" in img or "gx10" in img:
        return
    ex = cfg.get("extra_flags") or ""
    draft = _resolve_dspark_draft_model(readme)
    # Residual speculative_config.* after $DSPARK_CKPT scrub still needs draft fill
    # when the card declares a DSpark draft — even without the literal "dspark" token.
    if not _dspark_spec_present(ex):
        return
    # If only dotted speculative remnants remain and the card has no DSpark draft,
    # leave non-DSpark speculative (e.g. JSON MTP) alone unless method=dspark was set.
    if "dspark" not in ex.lower() and not draft:
        if "method" in ex.lower() and "dspark" not in ex.lower():
            return
        if not re.search(r"speculative_config\.(model|num_speculative_tokens)", ex.lower()):
            return
        # No draft on card for NVIDIA-style dotted form → strip unstable half-config.
        cfg["extra_flags"] = _strip_dspark_speculative(ex)
        warnings.append(
            "Stripped incomplete speculative_config.* flags (no DSpark draft on card)."
        )
        return
    if _spec_has_draft_model(ex):
        # Ensure method=dspark when draft path is a DSpark id (card may omit method).
        if draft and "dspark" in draft.lower() and "speculative_config.method" not in ex.lower():
            cfg["extra_flags"] = (ex + " --speculative_config.method dspark").strip()
            rationale.append("DSpark draft present → added --speculative_config.method dspark")
        return
    if draft:
        bits = [ex]
        if "speculative_config.method" not in ex.lower() and "method" not in (
            ex.lower().split("speculative-config")[-1][:80] if "speculative-config" in ex.lower() else ""
        ):
            if "dspark" in draft.lower() or "dspark" in (readme or "").lower():
                bits.append("--speculative_config.method dspark")
        bits.append(f"--speculative_config.model {draft}")
        cfg["extra_flags"] = " ".join(b for b in bits if b).strip()
        rationale.append(f"DSpark draft from card → --speculative_config.model {draft}")
        return
    cfg["extra_flags"] = _strip_dspark_speculative(ex)
    warnings.append(
        "Stripped DSpark speculative decode — no draft model path on the card "
        "(stable first boot). Re-add --speculative_config.model after downloading the draft."
    )
    rationale.append("FIRST BOOT: DSpark speculative stripped (missing draft model)")


def _scrub_unexpanded_shell_vars(extra: str, warnings: list[str]) -> str:
    """Drop argv tokens that still contain ``$VAR`` after card parsing.

    Cards write ``--model $MODEL_CKPT`` / ``--speculative_config.model $DSPARK_CKPT``.
    Leaving those through would make vLLM try to load a literal ``$…`` path.
    Prefer ``_expand_card_exports`` first so known card exports resolve.
    """
    s = (extra or "").strip()
    if not s or "$" not in s:
        return s
    try:
        parts = shlex.split(s)
    except ValueError:
        parts = s.split()
    out: list[str] = []
    dropped: list[str] = []
    i = 0
    while i < len(parts):
        p = parts[i]
        # Flag whose *value* is a shell var — drop both.
        if (
            p.startswith("--")
            and i + 1 < len(parts)
            and "$" in parts[i + 1]
            and not parts[i + 1].startswith("-")
        ):
            dropped.append(f"{p} {parts[i + 1]}")
            i += 2
            continue
        if "$" in p:
            dropped.append(p)
            i += 1
            continue
        out.append(p)
        i += 1
    if dropped:
        warnings.append(
            "Stripped unexpanded shell variables from card flags: "
            + ", ".join(dropped)
            + " (fill the real path or re-run after exporting them)."
        )
    return " ".join(shlex.quote(x) if (" " in x or "{" in x) else x for x in out)


def _scrub_unexpanded_docker_env(env: list[str], warnings: list[str]) -> list[str]:
    """Drop docker_env entries whose value still contains ``$VAR``."""
    out: list[str] = []
    dropped: list[str] = []
    for item in env or []:
        if not item or "=" not in item:
            continue
        _k, v = item.split("=", 1)
        if "$" in v:
            dropped.append(item)
            continue
        out.append(item)
    if dropped:
        warnings.append(
            "Dropped unexpanded shell vars from docker_env: " + ", ".join(dropped)
        )
    return out


# Flags / env that crash or noop-wrong on 1–2× GB10 UMA Spark serves.
_SPARK_UNSAFE_FLAGS = (
    "--enable-expert-parallel",
    "--data-parallel-size",
    "--data-parallel-address",
    "--data-parallel-rpc-port",
    "--data-parallel-backend",
)
_SPARK_UNSAFE_ENV_PREFIXES = (
    "VLLM_USE_DEEP_GEMM_MEGA_MOE=",
    "VLLM_ALL2ALL_BACKEND=",
)


def _strip_spark_unsafe_flags(
    cfg: dict[str, Any],
    warnings: list[str],
    rationale: list[str],
) -> None:
    """Drop card multi-GPU / mega-MoE knobs that do not apply on this cluster."""
    dropped: list[str] = []
    ex = cfg.get("extra_flags") or ""
    for flag in _SPARK_UNSAFE_FLAGS:
        if flag in ex:
            ex = _strip_flag_from_extra(ex, flag)
            dropped.append(flag)
    if dropped:
        cfg["extra_flags"] = ex
        warnings.append(
            "Stripped Spark-unsafe card flags: " + ", ".join(dropped)
            + " (expert/DP parallelism is not the 1–2× GB10 path)."
        )
        rationale.append("SAFETY: removed Spark-unsafe multi-GPU flags from extras")

    moe = (cfg.get("moe_backend") or "").strip().lower()
    if moe == "humming":
        cfg["moe_backend"] = ""
        warnings.append(
            "Cleared --moe-backend humming (Ampere / non-GB10 recipe; leave auto on Spark)."
        )
        rationale.append("SAFETY: humming moe-backend → empty (auto) on GB10")

    # Ampere W4A16 snippets pair --linear-backend humming with moe humming.
    # Clearing only moe leaves a leftover that still fails on GB10.
    if re.search(r"--linear-backend\s+humming\b", ex, re.I):
        ex = _strip_flag_from_extra(ex, "--linear-backend")
        cfg["extra_flags"] = ex
        warnings.append(
            "Cleared --linear-backend humming (Ampere / non-GB10 recipe; leave auto on Spark)."
        )
        rationale.append("SAFETY: humming linear-backend → stripped on GB10")

    env = list(cfg.get("docker_env") or [])
    kept = []
    env_drop = []
    for e in env:
        if any(e.startswith(p) for p in _SPARK_UNSAFE_ENV_PREFIXES):
            env_drop.append(e.split("=", 1)[0])
            continue
        # Mega MoE deep_gemm toggle often appears as VLLM_USE_DEEP_GEMM=1 with mega notes;
        # only strip the explicit mega env keys above.
        kept.append(e)
    if env_drop:
        cfg["docker_env"] = kept
        warnings.append("Stripped Spark-unsafe docker env: " + ", ".join(env_drop))
        rationale.append("SAFETY: removed mega-MoE / all2all env knobs")


_CARD_IMAGE_RE = re.compile(
    r"(?:"
    r"vllm/vllm-openai|"
    r"ghcr\.io/anemll/dspark-vllm-gx10|"
    r"eugr/spark-vllm|"
    r"nvcr\.io/nvidia/vllm|"
    r"nvcr\.io/[^\s:`'\"/]+/vllm(?:/[^\s:`'\"]+)?"
    r")(?::[vV]?[\w.\-]+)?",
    re.I,
)


def _parse_card_image_requirement(readme: str | None) -> str | None:
    """First concrete image pin mentioned on the card (bare tag preferred)."""
    if not readme:
        return None
    # Prefer Spark-section hits: walk near "DGX Spark" / "GB10" first.
    spark_hits: list[str] = []
    other: list[str] = []
    for m in _CARD_IMAGE_RE.finditer(readme):
        ref = m.group(0)
        if ":" not in ref:
            continue  # untagged — useless as a pin
        tag = ref.split(":", 1)[1].lower()
        if tag in ("latest", "nightly") or tag.startswith("nightly"):
            continue
        window = readme[max(0, m.start() - 400) : m.start()].lower()
        if "dgx spark" in window or "gb10" in window:
            spark_hits.append(ref)
        else:
            other.append(ref)
    return (spark_hits or other or [None])[0]


def _card_has_only_floating_image(readme: str | None) -> bool:
    """True when the card mentions a vLLM image but every tag is nightly/latest."""
    if not readme:
        return False
    tagged = 0
    floating = 0
    for m in _CARD_IMAGE_RE.finditer(readme):
        ref = m.group(0)
        if ":" not in ref:
            continue
        tagged += 1
        tag = ref.split(":", 1)[1].lower()
        if tag in ("latest", "nightly") or tag.startswith("nightly"):
            floating += 1
    return tagged > 0 and floating == tagged


def _warn_floating_card_image(
    readme: str | None, lab_default: str, warnings: list[str]
) -> None:
    if not _card_has_only_floating_image(readme):
        return
    msg = (
        f"Card only pins vllm/vllm-openai:nightly (or :latest); staying on lab default "
        f"{lab_default}. New architectures on that card may fail to load until a dated tag exists."
    )
    if msg not in warnings:
        warnings.append(msg)


def _semver_tuple(tag: str) -> tuple[int, ...]:
    t = tag.lstrip("vV")
    nums = re.findall(r"\d+", t)
    return tuple(int(x) for x in nums[:4]) if nums else (0,)


def _stock_image_semver(image: str | None) -> tuple[int, ...] | None:
    """Parse vLLM openai image tag; None for Anemll/custom/non-semver pins."""
    if not image:
        return None
    s = image.strip()
    if "anemll" in s.lower() or "dspark-vllm" in s.lower():
        return None  # independent version lineage
    if "vllm/vllm-openai:" not in s and not s.startswith("vllm-openai:"):
        # bare tag or other repo — try last :tag
        if ":" not in s:
            return None
    tag = s.rsplit(":", 1)[-1]
    if not tag or tag in ("latest", "nightly") or tag.startswith("nightly"):
        return None
    # Playbook pins like gemma4-cu130 are not semver — do not parse "4" as v4.0.0.
    if not re.match(r"^v?\d+\.\d+", tag):
        return None
    # strip arch/cuda suffixes: v0.27.1-aarch64 → 0.27.1
    tag = tag.split("-")[0]
    ver = _semver_tuple(tag)
    return ver if ver != (0,) else None


def _image_at_least(image: str | None, major: int, minor: int, patch: int = 0) -> bool:
    ver = _stock_image_semver(image)
    if ver is None:
        # Unknown/custom: assume current lab default capability (≥0.27).
        return True
    target = (major, minor, patch)
    # pad compare
    a = ver + (0,) * (3 - len(ver))
    b = target + (0,) * (3 - len(target))
    return a[:3] >= b[:3]


def _resolve_stock_image(default: str, card_image: str | None, rationale: list[str]) -> str:
    """Raise the lab default to the card's min stock tag; never downgrade; never replace Anemll.

    Playbook tags (gemma4-cu130, :gemma) are not semver — they are not a stock raise.
    """
    if not card_image:
        return default
    if "anemll" in (default or "").lower() or "dspark-vllm" in (default or "").lower():
        return default
    if not card_image.startswith("vllm/vllm-openai:"):
        return default
    d_ver = _stock_image_semver(default)
    c_ver = _stock_image_semver(card_image)
    if c_ver is None or d_ver is None:
        return default
    if c_ver > d_ver:
        rationale.append(
            f"Card requires newer stock image {card_image} than lab default {default} → using card pin"
        )
        return card_image
    if card_image != default and c_ver == d_ver:
        return default
    if c_ver < d_ver:
        rationale.append(
            f"Card pin {card_image} is older than lab default {default} — keeping lab default (no downgrade)"
        )
    return default


def _is_anemll_image(image: str | None) -> bool:
    s = (image or "").lower()
    return "anemll" in s or "dspark-vllm" in s


def _lab_default_image(mode: str | None = None) -> str:
    """Single stock image. ``mode`` is ignored (legacy)."""
    return DEFAULT_IMAGE_MAX or DEFAULT_IMAGE_SAFE


def _playbook_required_image(model: str, detected: dict[str, Any] | None) -> str | None:
    """NVIDIA DGX Spark playbook image pins that stock v0.27.1 cannot replace.

    Gemma 4 needs ``vllm/vllm-openai:gemma4-cu130``. DiffusionGemma needs
    ``vllm/vllm-openai:gemma``. Anemll/DSpark stays overlay-owned.
    """
    mid = (model or "").lower()
    fam = str((detected or {}).get("family") or "")
    if fam == "diffusiongemma" or "diffusiongemma" in mid:
        return "vllm/vllm-openai:gemma"
    if fam == "gemma4" or re.search(r"gemma[\s._-]*4", mid):
        return "vllm/vllm-openai:gemma4-cu130"
    return None


def _resolve_image_for_gates(
    cfg: dict[str, Any],
    *,
    mode: str,
    candidate_image: str | None,
    card_image: str | None,
    detected: dict[str, Any] | None,
    rationale: list[str],
    warnings: list[str] | None = None,
) -> str:
    """Resolve cfg['image'] before marlin/safety gates.

    Floor is the lab default for ``mode``. Stock card/docker pins and capability
    floors may only *raise*. Anemll/DSpark images are never replaced.
    Non-stock alternate pins (nvcr, eugr, …) are recorded, not auto-selected.
    """
    warnings = warnings if warnings is not None else []
    lab = _lab_default_image(mode)
    cur = (cfg.get("image") or "").strip()

    if _is_anemll_image(cur):
        return cur

    playbook = _playbook_required_image(str(cfg.get("model") or ""), detected)
    if playbook and not _is_anemll_image(playbook):
        if cur != playbook:
            rationale.append(f"NVIDIA DGX Spark playbook → image {playbook}")
        cfg["image"] = playbook
        return playbook

    # Never sit below the lab stock default (candidate may have applied an older pin).
    if not cur:
        cur = lab
    else:
        cv = _stock_image_semver(cur)
        lv = _stock_image_semver(lab)
        if cv is not None and lv is not None and cv < lv:
            rationale.append(
                f"Card/docker pin {cur} is older than lab default {lab} — keeping lab default (no downgrade)"
            )
            cur = lab
        elif cv is None and not cur.startswith("vllm/vllm-openai"):
            # Non-stock already on cfg (e.g. blind apply) — note and fall back to lab.
            warnings.append(
                f"Card image {cur} is not a stock vllm-openai pin; keeping lab default {lab}"
            )
            rationale.append(f"Card image {cur} noted (not auto-selected)")
            cur = lab
    cfg["image"] = cur

    def _note_alt(ref: str) -> None:
        msg = (
            f"Card mentions image {ref}; keeping lab stock/Anemll path "
            "(alternate image not auto-selected)"
        )
        if msg not in warnings:
            warnings.append(msg)
        note = f"Card image {ref} noted (not auto-selected)"
        if note not in rationale:
            rationale.append(note)

    def _raise_stock(pin: str | None, *, via: str) -> None:
        nonlocal cur
        if not pin:
            return
        if _is_anemll_image(pin):
            # Anemll selection is overlay-owned; do not swap stock → Anemll from prose/docker alone.
            rationale.append(f"Card mentions {pin} (overlay owns Anemll image selection)")
            return
        if pin.startswith("vllm/vllm-openai:"):
            new = _resolve_stock_image(cur, pin, rationale)
            if new != cur and via == "candidate" and not any(
                "card" in r.lower() and pin in r for r in rationale[-2:]
            ):
                # _resolve_stock_image already rationale'd raises; ensure docker path is clear
                pass
            cur = new
            cfg["image"] = cur
            return
        _note_alt(pin)

    _raise_stock(candidate_image, via="candidate")
    if card_image and card_image != candidate_image:
        _raise_stock(card_image, via="card")

    cap = _capability_min_stock_image(cfg, detected)
    if cap and not _is_anemll_image(cur):
        prev = cur
        cur = _resolve_stock_image(cur, cap, rationale)
        if cur != prev and not any("capability" in r for r in rationale[-3:]):
            rationale.append(
                f"Capability floor → stock image at least {cap} "
                f"(features: quant/moe/parsers on this config)"
            )
        cfg["image"] = cur

    return cfg.get("image") or lab


def _capability_min_stock_image(
    cfg: dict[str, Any],
    detected: dict[str, Any] | None,
) -> str | None:
    """Minimum stock vllm/vllm-openai tag required by selected features (not just card text).

    Returns a full image ref or None when no stock floor applies (custom/Anemll path).
    """
    need = (0, 0, 0)
    reasons: list[str] = []
    quant = (cfg.get("quantization") or "").lower()
    moe = (cfg.get("moe_backend") or "").lower()
    parsers = " ".join(
        [
            str(cfg.get("reasoning_parser") or ""),
            str(cfg.get("tool_call_parser") or ""),
        ]
    ).lower()
    det = detected or {}

    def raise_to(maj: int, minor: int, patch: int, why: str) -> None:
        nonlocal need
        t = (maj, minor, patch)
        if t > need:
            need = t
            reasons.append(why)

    if quant in ("modelopt_mixed", "modelopt_fp4", "modelopt_mxfp8"):
        raise_to(0, 27, 0, f"quant={quant}")
    if det.get("is_mixed_nvfp4_fp8") or det.get("has_nvfp4"):
        raise_to(0, 27, 0, "NVFP4/ModelOpt mixed checkpoint")
    if moe == "marlin" and (det.get("family") == "nemotron" or det.get("has_nvfp4")):
        raise_to(0, 27, 0, "moe-backend marlin on Spark NVFP4")
    if any(p in parsers for p in ("nemotron_v3", "nano_v3", "super_v3", "deepseek_v4", "minimax_m")):
        raise_to(0, 27, 0, f"parser set needs recent vLLM ({parsers.strip() or 'n/a'})")
    if need == (0, 0, 0):
        return None
    return f"vllm/vllm-openai:v{need[0]}.{need[1]}.{need[2]}"


def check_serve_loadability(
    *,
    mode: str,
    weights_gib: Optional[float],
    node_ram_gib: float,
    nodes_used: int,
    util: float,
    reserve_gib: float = 15.0,
) -> tuple[bool, Optional[str]]:
    """Shared Start/recommend gate: can weights load at final util on N nodes?

    Returns (fits, warning_or_none). Absolute capacity uses 0.85 after reserved
    UMA. ``mode`` is ignored (legacy).
    """
    del mode  # fit is hardware-only now
    if not weights_gib or weights_gib <= 0:
        return True, None
    n = max(1, int(nodes_used or 1))
    per = float(weights_gib) / n
    ram = _resolved_node_ram_gib(node_ram_gib)
    reserve = float(reserve_gib)
    if per > (ram * 0.85 - reserve):
        return False, (
            f"weights (~{weights_gib} GiB) do not fit {n} node(s) even at util=0.85"
        )
    return True, None


# ─── Optional GitHub cookbook fetch (card recipe-poor) ───────────────────────

COOKBOOK_FETCH_TIMEOUT = 8.0
COOKBOOK_MAX_BYTES = 1_500_000  # ~1.5 MiB hard cap
COOKBOOK_MAX_URLS = 6

_GITHUB_BLOB_OR_RAW_RE = re.compile(
    r"https?://(?:www\.)?github\.com/[\w.-]+/[\w.-]+/(?:blob|raw)/[^\s\)\]\"'<>]+",
    re.I,
)
_GITHUB_RAW_HOST_RE = re.compile(
    r"https?://raw\.githubusercontent\.com/[\w.-]+/[\w.-]+/[^\s\)\]\"'<>]+",
    re.I,
)
_COOKBOOK_DOC_EXTS = (".md", ".markdown", ".ipynb", ".txt", ".rst")


def github_blob_to_raw_url(url: str) -> Optional[str]:
    """Map github.com/{owner}/{repo}/blob|{raw}/{ref}/{path} → raw.githubusercontent.com."""
    if not url:
        return None
    u = url.strip()
    if u.startswith("https://raw.githubusercontent.com/") or u.startswith(
        "http://raw.githubusercontent.com/"
    ):
        return u.split("?", 1)[0].split("#", 1)[0]
    m = re.match(
        r"https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+)/(?:blob|raw)/([^/]+)/(.+)$",
        u.split("?", 1)[0].split("#", 1)[0],
        re.I,
    )
    if not m:
        return None
    owner, repo, ref, path = m.group(1), m.group(2), m.group(3), m.group(4)
    path = path.rstrip("/")
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"


def _looks_like_vllm_cookbook(url: str) -> bool:
    """True for public GitHub docs that look like a vLLM cookbook (not TRT/SGLang)."""
    path = (url or "").split("?", 1)[0].split("#", 1)[0].lower()
    if not any(path.endswith(ext) for ext in _COOKBOOK_DOC_EXTS):
        return False
    # Skip other-framework cookbooks unless the path also says vllm.
    if any(x in path for x in ("trtllm", "tensorrt", "sglang", "trt-llm")) and "vllm" not in path:
        return False
    name = path.rsplit("/", 1)[-1]
    return "cookbook" in path or "vllm" in name or "/vllm" in path or "vllm_" in path


_VENDOR_DOC_HOSTS = frozenset(
    {
        "unsloth.ai",
        "www.unsloth.ai",
        "docs.vllm.ai",
        "recipes.vllm.ai",
    }
)
_GENERIC_URL_RE = re.compile(r"https?://[^\s\)\]\"'<>]+", re.I)

# recipes.vllm.ai is case-sensitive: HF `google` → 500, catalog `Google` → 200.
# Only mixed-case catalog orgs need a map; lowercase HF orgs pass through.
_RECIPES_VLLM_ORGS = (
    "ByteDance-Seed",
    "Google",
    "IndexTeam",
    "JetBrains",
    "LiquidAI",
    "MiniMaxAI",
    "OpenGVLab",
    "OpenMOSS-Team",
    "PaddlePaddle",
    "Qwen",
    "Wan-AI",
    "XiaomiMiMo",
    "inclusionAI",
)
_RECIPES_VLLM_ORG_BY_LOWER = {o.lower(): o for o in _RECIPES_VLLM_ORGS}

# Official catalog org for a detected family. Quant twins (Unsloth) live under a
# different HF org; recipes.vllm.ai/{hf_org}/{name} 404s — also try this page.
_FAMILY_RECIPES_VLLM_ORG = {
    "gemma": "Google",
    "gemma4": "Google",
    "qwen": "Qwen",
}
# Trailing quant token only. Do not swallow -Fast (or any later tag) after NVFP4.
_QUANT_NAME_SUFFIX_RE = re.compile(
    r"-(?:NVFP4|FP8|GGUF|GPTQ|AWQ|BNB-4BIT|INT4|INT8)$",
    re.I,
)


def recipes_vllm_catalog_org(org: str) -> str:
    """HF org → recipes.vllm.ai catalog org (google → Google). Unknown orgs unchanged."""
    if not org:
        return org
    return _RECIPES_VLLM_ORG_BY_LOWER.get(org.lower(), org)


def _recipes_vllm_official_page(
    org: str, name: str, detected: dict[str, Any] | None
) -> str | None:
    """Official family catalog URL when the HF id is a requant / publisher twin."""
    fam = str((detected or {}).get("family") or "").lower()
    official_org = _FAMILY_RECIPES_VLLM_ORG.get(fam)
    if not official_org or not name:
        return None
    stem = _QUANT_NAME_SUFFIX_RE.sub("", name) or name
    if recipes_vllm_catalog_org(org).lower() == official_org.lower() and stem == name:
        return None
    return f"https://recipes.vllm.ai/{official_org}/{stem}"


def _want_nvidia_playbooks(
    model_id: str, detected: dict[str, Any] | None, slugs: list[str]
) -> bool:
    """NVIDIA cookbooks are Qwen / Nemotron / NVIDIA-org — not every NVFP4 family."""
    mid = (model_id or "").lower()
    fam = str((detected or {}).get("family") or "").lower()
    publisher = mid.split("/")[0] if "/" in mid else ""
    if publisher == "nvidia" or fam == "nemotron" or "nemotron" in mid:
        return True
    if fam == "qwen" or any(s.startswith("qwen") for s in slugs):
        return True
    return False


def _canonicalize_recipes_vllm_path(path: str) -> str:
    """/{org}/{name} with catalog org casing; never keep a trailing .md (those 404)."""
    parts = [p for p in (path or "").split("/") if p]
    if not parts:
        return "/"
    if parts[-1].lower().endswith(".md"):
        parts[-1] = parts[-1][:-3]
        if not parts[-1]:
            parts.pop()
    if not parts:
        return "/"
    parts[0] = recipes_vllm_catalog_org(parts[0])
    return "/" + "/".join(parts)


class _HTMLRecipeText(HTMLParser):
    """Strip scripts/styles; wrap <pre> as fences so the last serve flag is kept."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip += 1
            return
        if tag == "pre":
            self._parts.append("\n```\n")
            return
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "tr", "section"):
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            if self._skip:
                self._skip -= 1
            return
        if tag == "pre":
            self._parts.append("\n```\n")
            return
        if tag in ("p", "div", "li", "section"):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def html_recipe_text(body: str) -> str:
    """HTML catalog page → text with serve commands (scripts dropped)."""
    parser = _HTMLRecipeText()
    try:
        parser.feed(body or "")
        parser.close()
    except Exception:
        return body or ""
    return parser.text()


def vendor_doc_to_fetch_url(url: str) -> Optional[str]:
    """Allowlisted vendor docs → fetchable HTTPS URL (GitBook pages get .md)."""
    if not url:
        return None
    raw = url.strip().split("?", 1)[0].split("#", 1)[0]
    try:
        p = urlparse(raw)
    except ValueError:
        return None
    host = (p.netloc or "").lower()
    if p.scheme != "https" or host not in _VENDOR_DOC_HOSTS:
        return None
    path = p.path or "/"
    if host == "recipes.vllm.ai":
        # Catalog pages are HTML at /{Org}/{name}. Appending .md always 404s.
        return f"https://{host}{_canonicalize_recipes_vllm_path(path)}"
    if not any(path.lower().endswith(ext) for ext in _COOKBOOK_DOC_EXTS):
        path = path.rstrip("/") + ".md"
    return f"https://{host}{path}"


def find_vendor_doc_urls(readme: str) -> list[str]:
    """Scan README for Unsloth / vLLM recipe pages when the card has no serve line."""
    if not readme:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for m in _GENERIC_URL_RE.finditer(readme):
        url = m.group(0).rstrip(".,;:")
        fetch = vendor_doc_to_fetch_url(url)
        if not fetch:
            continue
        key = fetch.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(url)
    return found


def find_cookbook_urls(readme: str) -> list[str]:
    """Scan README for public GitHub blob/raw cookbook URLs (vLLM-oriented)."""
    if not readme:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for cre in (_GITHUB_BLOB_OR_RAW_RE, _GITHUB_RAW_HOST_RE):
        for m in cre.finditer(readme):
            url = m.group(0).rstrip(".,;:")
            if not _looks_like_vllm_cookbook(url):
                continue
            raw = github_blob_to_raw_url(url) or url
            key = raw.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(url)
    return found


@dataclass
class RecipeUrl:
    """A public recipe document to fetch (card-extracted or derived)."""

    url: str
    kind: str  # vendor_doc | github_cookbook | nvidia_playbook | vllm_docs
    origin: str = "card"  # card | derived


def family_doc_slugs(model_id: str, detected: dict[str, Any] | None = None) -> list[str]:
    """Stable vendor-doc slugs derived from the model id + detected family.

    Publisher-agnostic: ``Qwen/Qwen3.8-27B`` and ``unsloth/Qwen3.8-27B-NVFP4``
    both yield ``qwen3.8``.
    """
    det = detected or {}
    mid = (model_id or "").lower()
    name = mid.split("/")[-1] if mid else ""
    blob = " ".join(
        [
            mid,
            name,
            str(det.get("family") or ""),
            str(det.get("model_type") or ""),
            " ".join(str(a) for a in (det.get("architectures") or [])),
        ]
    ).lower()
    slugs: list[str] = []

    def add(slug: str) -> None:
        if slug and slug not in slugs:
            slugs.append(slug)

    if re.search(r"qwen3[\._\s-]*8\b", blob) or "qwen3_8" in blob or "qwen3.8" in blob:
        add("qwen3.8")
    elif re.search(r"qwen3[\._\s-]*6\b", blob) or "qwen3_6" in blob or "qwen3.6" in blob:
        add("qwen3.6")
    elif re.search(r"qwen3[\._\s-]*5\b", blob) or "qwen3_5" in blob or "qwen3.5" in blob:
        # Qwen3.8 reused the qwen3_5 class names for the 27B VL checkpoint.
        if det.get("is_vl") or "conditionalgeneration" in blob:
            add("qwen3.8")
        add("qwen3.5")
    elif "qwen3" in blob or det.get("family") == "qwen":
        add("qwen3")
        if det.get("is_vl") or "conditionalgeneration" in blob:
            add("qwen3.8")
    if "nemotron" in blob:
        add("nemotron")
    if "minimax" in blob and re.search(r"(^|[^a-z])m3([^0-9]|$)", blob):
        add("minimax-m3")
    elif "minimax" in blob:
        add("minimax")
    if "deepseek" in blob and "v4" in blob:
        add("deepseek-v4")
    elif "deepseek" in blob:
        add("deepseek")
    if det.get("family") == "gemma4" or re.search(r"gemma[\s._-]*4", blob):
        add("gemma-4")
    elif det.get("family") == "gemma" or "gemma" in blob:
        add("gemma")
    if det.get("family") == "glm" or "glm" in blob or "chatglm" in blob:
        add("glm")
    if det.get("family") == "kimi" or "kimi" in blob or "moonshot" in blob:
        add("kimi")
    if det.get("family") in ("llama", "llama4") or "llama" in blob:
        add("llama")
    if det.get("family") == "phi" or re.search(r"(^|[^a-z])phi", blob):
        if "phi-4" in blob or "phi4" in blob:
            add("phi-4")
        add("phi")
    if det.get("family") == "granite" or "granite" in blob:
        add("granite")
    if "gpt-oss" in blob or "gptoss" in blob:
        add("gpt-oss")
    if det.get("family") == "mistral" or any(
        k in blob for k in ("mistral", "magistral", "devstral", "mixtral", "pixtral")
    ):
        add("mistral")
    if det.get("family") == "internlm" or "internlm" in blob:
        add("internlm")
    if det.get("family") in ("hunyuan", "hy_v3") or "hunyuan" in blob or re.search(r"(^|[^a-z])hy3", blob):
        add("hunyuan")
    if det.get("family") in ("step3", "step3p5") or "step" in blob:
        add("step")
    if det.get("family") == "ernie" or "ernie" in blob:
        add("ernie")
    if det.get("has_nvfp4") or "nvfp4" in blob:
        add("nvfp4")
    return slugs


def _recipe_source_kind(url: str) -> str:
    u = (url or "").lower()
    host = ""
    try:
        host = (urlparse(url).netloc or "").lower()
    except ValueError:
        host = ""
    if (
        "nvidia.com" in host
        or "nvidia-nemo" in u
        or "dgx-spark" in u
        or ("nemotron" in u and ("github.com" in host or "githubusercontent.com" in host))
    ):
        if "github.com" in host or "githubusercontent.com" in host or "nvidia.com" in host:
            return "nvidia_playbook"
    if host in ("docs.vllm.ai", "recipes.vllm.ai"):
        return "vllm_docs"
    if host in ("unsloth.ai", "www.unsloth.ai"):
        return "vendor_doc"
    if "github.com" in host or "githubusercontent.com" in host:
        return "github_cookbook"
    return "vendor_doc"


# Deterministic NVIDIA vLLM cookbook paths (fail closed if 404 / HTML).
_NVIDIA_VLLM_COOKBOOKS = (
    "https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Example/vllm_cookbook.ipynb",
    "https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/main/nvidia/vllm/README.md",
)


def discover_recipe_urls(
    model_id: str,
    detected: dict[str, Any] | None = None,
    readme: str = "",
) -> list[RecipeUrl]:
    """Card-extracted URLs plus derived Unsloth / NVIDIA / vLLM / GitHub recipes.

    Derived URLs do not require a link on the HF card. Hosts are allowlisted;
    HTML-only pages fail later at fetch.
    """
    out: list[RecipeUrl] = []
    seen: set[str] = set()

    def add(url: str, *, kind: str | None = None, origin: str) -> None:
        if not url:
            return
        key = (github_blob_to_raw_url(url) or vendor_doc_to_fetch_url(url) or url).lower()
        if key in seen:
            return
        seen.add(key)
        out.append(RecipeUrl(url=url, kind=kind or _recipe_source_kind(url), origin=origin))

    for u in find_cookbook_urls(readme or ""):
        add(u, origin="card")
    for u in find_vendor_doc_urls(readme or ""):
        add(u, origin="card")

    slugs = family_doc_slugs(model_id, detected)

    for slug in slugs:
        if slug == "nvfp4":
            add("https://unsloth.ai/docs/basics/nvfp4", kind="vendor_doc", origin="derived")
            continue
        add(f"https://unsloth.ai/docs/models/{slug}", kind="vendor_doc", origin="derived")

    # Canonical recipes.vllm.ai page is /{Org}/{name}, not /{slug}.md (those 404).
    # Catalog org is case-sensitive (google → 500, Google → 200). Quant twins
    # (unsloth/gemma-4-*-NVFP4) 404 at the HF org — also try the official family page.
    if "/" in (model_id or "") and not str(model_id).startswith("/"):
        org, name = model_id.split("/", 1)
        if org and name:
            add(
                f"https://recipes.vllm.ai/{recipes_vllm_catalog_org(org)}/{name}",
                kind="vllm_docs",
                origin="derived",
            )
            official = _recipes_vllm_official_page(org, name, detected)
            if official:
                add(official, kind="vllm_docs", origin="derived")

    if _want_nvidia_playbooks(model_id, detected, slugs):
        for u in _NVIDIA_VLLM_COOKBOOKS:
            add(u, kind="nvidia_playbook", origin="derived")

    return out


def notebook_source_text(body: str) -> str:
    """Flatten Jupyter notebook JSON cell sources; plain text returned as-is."""
    if not body:
        return body
    stripped = body.lstrip()
    if not (stripped.startswith("{") and '"cells"' in stripped[:4000]):
        return body
    try:
        nb = json.loads(body)
    except json.JSONDecodeError:
        return body
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        return body
    parts: list[str] = []
    for cell in nb["cells"]:
        if not isinstance(cell, dict):
            continue
        src = cell.get("source")
        if isinstance(src, list):
            parts.append("".join(str(x) for x in src))
        elif isinstance(src, str):
            parts.append(src)
    return "\n\n".join(parts) if parts else body


def fetch_cookbook_text(
    url: str,
    *,
    timeout: float = COOKBOOK_FETCH_TIMEOUT,
    max_bytes: int = COOKBOOK_MAX_BYTES,
) -> tuple[Optional[str], Optional[str]]:
    """Fetch a public GitHub cookbook or allowlisted vendor doc. Returns (text, error). Never raises."""
    if not url:
        return None, "empty cookbook URL"
    raw_url = github_blob_to_raw_url(url)
    if raw_url:
        if not raw_url.startswith("https://raw.githubusercontent.com/"):
            return None, f"refusing non-raw GitHub host: {raw_url}"
    else:
        raw_url = vendor_doc_to_fetch_url(url)
        if not raw_url:
            return None, f"not a public GitHub/vendor-doc URL: {url}"
    # Public docs only — never send HF tokens off-Hub.
    body, err = _http_get_raw(
        raw_url,
        timeout=timeout,
        token=None,
        allow_retry_without_auth=False,
        max_bytes=max_bytes,
    )
    if err or not body:
        return None, err or "empty cookbook body"
    if body.lstrip().startswith("<!"):
        host = ""
        try:
            host = (urlparse(raw_url).netloc or "").lower()
        except ValueError:
            host = ""
        if host == "recipes.vllm.ai":
            text = html_recipe_text(body)
            if text.strip():
                return text, None
        return None, f"HTML response (not raw content) for {raw_url}"
    return notebook_source_text(body), None


def candidates_recipe_poor(candidates: list[ServeCandidate]) -> bool:
    """True when card recipes are missing or only bare/demo-level."""
    if not candidates:
        return True
    best = candidates[0]
    cfg = best.config or {}
    rich = bool(
        cfg.get("quantization")
        or cfg.get("moe_backend")
        or cfg.get("docker_env")
        or cfg.get("kv_cache_dtype")
        or cfg.get("tool_call_parser")
        or cfg.get("reasoning_parser")
        or cfg.get("load_format")
        or cfg.get("image")
        or (cfg.get("extra_flags") or "").strip()
    )
    if rich and best.score >= 20:
        return False
    return (not rich) or best.score < 20


def _vendor_candidate_family_mismatch(cand: ServeCandidate, want_fam: str) -> bool:
    """True when a vendor-doc snippet is clearly for another model family.

    Unsloth / NVIDIA pages mix several models on one document; a high-scoring
    Gemma serve line must not win for a Qwen checkpoint.
    """
    if not want_fam or want_fam == "unknown":
        return False
    if cand.model:
        other = analyze_config({}, cand.model).get("family") or ""
        if other and other != "unknown" and other != want_fam:
            return True
    rp = str((cand.config or {}).get("reasoning_parser") or "").lower()
    tcp = str((cand.config or {}).get("tool_call_parser") or "").lower()
    blob = f"{rp} {tcp} {cand.model or ''} {cand.raw or ''}".lower()
    if want_fam == "qwen" and re.search(r"gemma", blob) and "qwen" not in (cand.model or "").lower():
        return True
    if want_fam.startswith("gemma") and re.search(r"qwen", blob) and "gemma" not in (cand.model or "").lower():
        return True
    return False


# E2B/E4B before bare NNB so Gemma-4-E4B stays e4b; 26B-A4B stays one MoE token.
_MODEL_SIZE_TOKEN_RE = re.compile(
    r"(?i)(?:e\d+b|\d+(?:\.\d+)?[bt](?:-a\d+[bt])?)"
)
# Qwen3.8 / Gemma-4 / Llama-3.1 — generation, not the trailing 27B/31B size.
_MODEL_GENERATION_RE = re.compile(r"(?i)(?:qwen|gemma|llama|glm)-?\d+(?:\.\d+)?")
_PLACEHOLDER_MODEL_RE = re.compile(r"^\$[A-Z][A-Z0-9_]*$")
_HF_REPO_ID_RE = re.compile(
    r"\b([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?/[A-Za-z0-9._-]+)\b"
)
_WEIGHT_FORK_ORGS = frozenset({"nvidia", "unsloth", "bartowski", "lmstudio-community"})


def _model_size_tokens(model_id: str) -> frozenset[str]:
    """Size/variant tokens from a HF id (31b, 26b-a4b, e4b). Empty if none."""
    name = (model_id or "").split("/")[-1]
    return frozenset(m.group(0).lower() for m in _MODEL_SIZE_TOKEN_RE.finditer(name))


def _model_generation_tokens(model_id: str) -> frozenset[str]:
    """Generation stems from a HF id (qwen3.8, qwen3.6, gemma-4). Empty if none."""
    name = (model_id or "").split("/")[-1]
    return frozenset(m.group(0).lower() for m in _MODEL_GENERATION_RE.finditer(name))


_IQ_QUANT_SUFFIX_RE = re.compile(r"(?i)-iq\d+[a-z0-9_]*$")


def _model_core_name(model_id: str) -> str:
    """Repo name minus org and trailing quant/pack suffixes (nvfp4, fp8, gguf, iq*).

    Empty when the token is not a resolvable HF name (flags, ``...``, ``$VAR``).
    Does not strip instruction ``-it`` (gemma-4-31B-it vs gemma-4-31B-it-NVFP4)
    or checkpoint tags after a quant (``-NVFP4-Fast`` keeps ``-fast``).
    Repeats when several quant tokens trail (``-GPTQ-INT4``).
    """
    raw = (model_id or "").strip().strip("\"'")
    name = raw.split("/")[-1].strip()
    if (
        not name
        or name.startswith("-")
        or name.startswith("$")
        or name in {".", "..", "..."}
        or not re.search(r"[A-Za-z]", name)
    ):
        return ""
    stripped = name
    while True:
        nxt = _QUANT_NAME_SUFFIX_RE.sub("", stripped)
        nxt = _IQ_QUANT_SUFFIX_RE.sub("", nxt)
        if nxt == stripped:
            break
        stripped = nxt
    return stripped.lower()


def _core_names_compatible(a: str, b: str) -> bool:
    """Equal cores, or a delimited prefix whose extra tokens are omitted size/gen.

    ``gemma-4`` vs ``gemma-4-31b-it`` is compatible (missing size).
    ``qwen3.6-35b-a3b`` vs ``qwen3.6-35b-a3b-fast`` is not (distinct checkpoint).
    """
    if not a or not b or a == b:
        return True
    lo, hi = (a, b) if len(a) <= len(b) else (b, a)
    if not (hi.startswith(lo) and hi[len(lo) : len(lo) + 1] in "-_."):
        return False
    extra = hi[len(lo) + 1 :]
    return bool(_MODEL_SIZE_TOKEN_RE.search(extra) or _MODEL_GENERATION_RE.search(extra))


def _is_placeholder_model_id(mid: str) -> bool:
    s = (mid or "").strip().strip("\"'")
    return (not s) or bool(_PLACEHOLDER_MODEL_RE.match(s))


def _hf_repo_ids_in_text(text: str) -> list[str]:
    return [m.group(1) for m in _HF_REPO_ID_RE.finditer(text or "")]


def _candidate_model_id(cand: ServeCandidate) -> str:
    """Resolve the snippet's HF id; skip empty / $MODEL placeholders.

    When ``cand.model`` is empty or a shell placeholder, scan ``vllm serve <id>``
    and ``--model-path <id>`` (SGLang / vendor templates).
    """
    mid = (cand.model or "").strip().strip("\"'")
    if mid and not _is_placeholder_model_id(mid):
        return mid
    raw = cand.raw or ""
    for cre in (
        re.compile(r"vllm\s+serve\s+(\S+)", re.I),
        re.compile(r"--model-path(?:\s+|=)(\S+)", re.I),
    ):
        m = cre.search(raw)
        if not m:
            continue
        tok = m.group(1).strip("\"'")
        if tok and not _is_placeholder_model_id(tok):
            return tok
    return ""


def _model_ids_sibling_mismatch(other: str, model_id: str) -> bool:
    """True when two HF ids are not publisher/quant twins of the same model.

    SAME: Qwen/Qwen3.8-27B vs unsloth/Qwen3.8-27B-NVFP4 (publisher/quant).
    DIFFERENT: core name (Widget-Chat vs Gadget-Instruct), size, generation
    (Qwen3.6 vs Qwen3.8), MoE variant, weight forks.
    Missing tokens on one side do not invent a mismatch.
    """
    if not other or not model_id:
        return False
    if "/" in model_id and "/" in other:
        pa = model_id.split("/")[0].lower()
        pb = other.split("/")[0].lower()
        if pa != pb and pa in _WEIGHT_FORK_ORGS and pb in _WEIGHT_FORK_ORGS:
            return True
    want_c = _model_core_name(model_id)
    got_c = _model_core_name(other)
    if want_c and got_c and not _core_names_compatible(want_c, got_c):
        return True
    want = _model_size_tokens(model_id)
    got = _model_size_tokens(other)
    if want and got and want != got:
        return True
    want_g = _model_generation_tokens(model_id)
    got_g = _model_generation_tokens(other)
    if want_g and got_g and want_g != got_g:
        return True
    return False


def _vendor_candidate_sibling_mismatch(cand: ServeCandidate, model_id: str) -> bool:
    """True when a vendor snippet is a different size/generation/variant.

    Unsloth's Gemma-4 page scores the 26B-A4B Spark MoE line (flashinfer_b12x)
    above the official dense 31B recipe. That backend is wrong for 31B.
    NVIDIA Agent-Ready lines name nvidia/… and must not win for unsloth/….
    Qwen3.6-27B must not win for a Qwen3.8-27B paste (same size, new generation).
    Fail-closed: raw naming a different org/name drops even when cand.model is
    empty or ``$MODEL``.
    """
    if not model_id:
        return False
    other = _candidate_model_id(cand)
    if other:
        return _model_ids_sibling_mismatch(other, model_id)
    named = _hf_repo_ids_in_text(cand.raw or "")
    if not named:
        return False
    return any(_model_ids_sibling_mismatch(hid, model_id) for hid in named)


def _drop_family_mismatched_hints(hints: dict[str, Any], detected: dict[str, Any] | None) -> None:
    """Strip parsers harvested from a mixed vendor page that name another family."""
    want_fam = str((detected or {}).get("family") or "")
    if not want_fam or want_fam == "unknown" or not hints:
        return
    dummy = ServeCandidate(
        raw="",
        model="",
        config={
            "reasoning_parser": hints.get("reasoning_parser") or "",
            "tool_call_parser": hints.get("tool_call_parser") or "",
        },
    )
    if _vendor_candidate_family_mismatch(dummy, want_fam):
        hints.pop("reasoning_parser", None)
        hints.pop("tool_call_parser", None)
        hints.pop("enable_auto_tool_choice", None)


def _augment_candidates_from_cookbooks(
    readme: str,
    *,
    candidates: list[ServeCandidate],
    detected: dict[str, Any] | None,
    sources: list[dict[str, str]],
    rationale: list[str],
    warnings: list[str],
    model_id: str = "",
    vendor_bodies: list[str] | None = None,
) -> list[ServeCandidate]:
    """Fetch card-linked *and derived* vendor docs when recipes are weak or thin."""
    best_cfg = (candidates[0].config if candidates else {}) or {}
    thin = not (
        best_cfg.get("reasoning_parser")
        or best_cfg.get("tool_call_parser")
        or best_cfg.get("kv_cache_dtype")
        or best_cfg.get("quantization")
    )
    if not candidates_recipe_poor(candidates) and not thin:
        return candidates
    found = discover_recipe_urls(model_id, detected, readme or "")
    if not found:
        return candidates
    derived_n = sum(1 for r in found if r.origin == "derived")
    rationale.append(
        f"Card recipes empty/weak — trying {min(len(found), COOKBOOK_MAX_URLS)} "
        f"cookbook/vendor doc(s)"
        + (f" ({derived_n} derived from model id/family)" if derived_n else "")
    )
    merged = list(candidates)
    seen_raw = {re.sub(r"\s+", " ", c.raw)[:200] for c in merged}
    for rec in found[:COOKBOOK_MAX_URLS]:
        url = rec.url
        try:
            text, err = fetch_cookbook_text(url)
        except Exception as e:  # never brick recommend
            warnings.append(f"Cookbook fetch error for {url}: {type(e).__name__}: {e}")
            continue
        if err or not text:
            warnings.append(f"Cookbook fetch skipped: {url} ({err or 'empty'})")
            continue
        if vendor_bodies is not None:
            vendor_bodies.append(text)
        sources.append(
            {
                "kind": rec.kind,
                "ref": url,
                "notes": (
                    f"fetched {rec.kind} ({rec.origin})"
                    if rec.origin == "derived"
                    else f"fetched {rec.kind} (card recipe-poor)"
                ),
            }
        )
        extra = extract_serve_candidates(text, detected=detected)
        added = 0
        want_fam = str((detected or {}).get("family") or "")
        for c in extra:
            if not c.section:
                c.section = rec.kind.replace("_", " ")
            if want_fam and want_fam != "unknown" and _vendor_candidate_family_mismatch(
                c, want_fam
            ):
                continue
            if _vendor_candidate_sibling_mismatch(c, model_id):
                continue
            key = re.sub(r"\s+", " ", c.raw)[:200]
            if key in seen_raw:
                continue
            seen_raw.add(key)
            merged.append(c)
            added += 1
        if added:
            rationale.append(f"{rec.kind} {url}: added {added} vllm serve recipe(s)")
        else:
            rationale.append(f"{rec.kind} {url}: fetched but no parseable vllm serve")
    merged.sort(key=lambda c: c.score, reverse=True)
    return merged


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
    seen_raw: dict[str, ServeCandidate] = {}

    def _keep_or_upgrade(cand: ServeCandidate, section: str) -> None:
        key = re.sub(r"\s+", " ", cand.raw)[:200]
        prev = seen_raw.get(key)
        if prev:
            # Same command later under #### 1x DGX Spark — keep the hardware heading
            # so scoring/label use Spark, not the earlier generic Quick Start.
            if _section_spark_rank(section) > _section_spark_rank(prev.section):
                prev.section = section
            return
        cand.section = section
        seen_raw[key] = cand
        candidates.append(cand)

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
            _keep_or_upgrade(cand, _section_at(heads, start))

    # 2) Unfenced inline `vllm serve …` lines in prose
    for m in re.finditer(r"(?m)^(?:export\s+\S+\s*\n)*[^\n]*\bvllm\s+serve\b[^\n]*(?:\n[^\n]*\\[^\n]*)*", readme):
        frag = m.group(0)
        if "```" in frag:
            continue
        cand = _parse_one_serve_command(frag)
        if not cand:
            continue
        _keep_or_upgrade(cand, _section_at(heads, m.start()))

    # 3) Notebook-style shell lines (`!vllm serve …`) after cell flatten
    for m in re.finditer(r"(?m)^!\s*vllm\s+serve\b[^\n]*(?:\n[^\n]*\\[^\n]*)*", readme):
        frag = m.group(0).lstrip("!").strip()
        cand = _parse_one_serve_command(frag)
        if not cand:
            continue
        _keep_or_upgrade(cand, _section_at(heads, m.start()) or "notebook")

    for c in candidates:
        c.score, c.reasons = score_candidate(c, readme, detected=detected)
        _sanitize_moe_backend_on_candidate(c, detected)
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


_SGLANG_LAUNCH_RE = re.compile(
    r"(?:python(?:3(?:\.\d+)?)?\s+-m\s+)?sglang\.launch_server\b",
    re.I,
)


def _is_sglang_block(body: str) -> bool:
    b = (body or "").lower()
    return "sglang.launch_server" in b or "sglang.launch" in b


def _parse_one_sglang_command(text: str) -> Optional[ServeCandidate]:
    """Parse ``python -m sglang.launch_server …`` (or bare ``sglang.launch_server``)."""
    text = _collapse_continuations(text or "")
    m = _SGLANG_LAUNCH_RE.search(text)
    if not m:
        return None
    after = text[m.end() :].strip()
    lines: list[str] = []
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
        if lines and re.match(r"^(export|uv |pip |source |cd |curl |vllm )\b", s):
            break
        lines.append(s)
    joined = re.sub(r"\s+#\s.*$", "", " ".join(lines))
    try:
        tokens = shlex.split(joined, posix=True)
    except ValueError:
        tokens = joined.split()
    cfg: dict[str, Any] = {
        "engine": "sglang",
        "model_path": "",
        "tp_size": None,
        "mem_fraction_static": None,
        "context_length": None,
        "trust_remote_code": False,
        "host": "0.0.0.0",
        "port": 30000,
        "quantization": "",
        "tool_call_parser": "",
        "reasoning_parser": "",
        "spec_algorithm": "",
        "spec_num_steps": None,
        "spec_eagle_topk": None,
        "spec_draft_tokens": None,
        "extra_flags": "",
    }
    extras: list[str] = []
    model = None
    i = 0
    while i < len(tokens):
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None

        if t in ("--model-path", "--model") or t.startswith("--model-path=") or t.startswith("--model="):
            val = t.split("=", 1)[1] if "=" in t and t.startswith("--") else (nxt or "")
            if "=" in t and t.startswith("--"):
                i += 1
            else:
                i += 2
            cfg["model_path"] = val
            model = val
            continue
        if t in ("--tp-size", "--tensor-parallel-size") or t.startswith("--tp-size="):
            val = t.split("=", 1)[1] if "=" in t else (nxt or "")
            i += 1 if "=" in t else 2
            try:
                cfg["tp_size"] = int(val)
            except (TypeError, ValueError):
                pass
            continue
        if t == "--mem-fraction-static" or t.startswith("--mem-fraction-static="):
            val = t.split("=", 1)[1] if "=" in t else (nxt or "")
            i += 1 if "=" in t else 2
            try:
                cfg["mem_fraction_static"] = float(val)
            except (TypeError, ValueError):
                pass
            continue
        if t == "--context-length" or t.startswith("--context-length="):
            val = t.split("=", 1)[1] if "=" in t else (nxt or "")
            i += 1 if "=" in t else 2
            try:
                cfg["context_length"] = int(val)
            except (TypeError, ValueError):
                pass
            continue
        if t == "--trust-remote-code":
            cfg["trust_remote_code"] = True
            i += 1
            continue
        if t in ("--host", "--port", "--quantization", "--tool-call-parser", "--reasoning-parser") or any(
            t.startswith(p) for p in (
                "--host=", "--port=", "--quantization=",
                "--tool-call-parser=", "--reasoning-parser=",
            )
        ):
            key = t.lstrip("-").split("=", 1)[0].replace("-", "_")
            val = t.split("=", 1)[1] if "=" in t else (nxt or "")
            i += 1 if "=" in t else 2
            if key == "port":
                try:
                    cfg["port"] = int(val)
                except (TypeError, ValueError):
                    pass
            else:
                cfg[key] = val
            continue
        # Unsloth cards use --speculative-algo; some docs use --speculative-algorithm.
        if t in ("--speculative-algorithm", "--speculative-algo") or t.startswith(
            "--speculative-algorithm="
        ) or t.startswith("--speculative-algo="):
            val = t.split("=", 1)[1] if "=" in t else (nxt or "")
            i += 1 if "=" in t else 2
            cfg["spec_algorithm"] = val
            continue
        if t == "--speculative-num-steps" or t.startswith("--speculative-num-steps="):
            val = t.split("=", 1)[1] if "=" in t else (nxt or "")
            i += 1 if "=" in t else 2
            try:
                cfg["spec_num_steps"] = int(val)
            except (TypeError, ValueError):
                pass
            continue
        if t == "--speculative-eagle-topk" or t.startswith("--speculative-eagle-topk="):
            val = t.split("=", 1)[1] if "=" in t else (nxt or "")
            i += 1 if "=" in t else 2
            try:
                cfg["spec_eagle_topk"] = int(val)
            except (TypeError, ValueError):
                pass
            continue
        if t == "--speculative-num-draft-tokens" or t.startswith("--speculative-num-draft-tokens="):
            val = t.split("=", 1)[1] if "=" in t else (nxt or "")
            i += 1 if "=" in t else 2
            try:
                cfg["spec_draft_tokens"] = int(val)
            except (TypeError, ValueError):
                pass
            continue
        extras.append(t)
        i += 1
    extra = " ".join(x for x in extras if x and not x.startswith("$"))
    cfg["extra_flags"] = extra.strip()
    reasons = []
    score = 10.0
    if cfg.get("spec_algorithm"):
        score += 20
        reasons.append(f"speculative {cfg['spec_algorithm']}")
    if cfg.get("trust_remote_code"):
        score += 5
    return ServeCandidate(
        raw=text.strip()[:500],
        model=model,
        args=tokens,
        config=cfg,
        score=score,
        reasons=reasons,
    )


def extract_sglang_candidates(readme: str) -> list[ServeCandidate]:
    """SGLang launch recipes on a card or vendor doc."""
    if not readme:
        return []
    heads = _section_map(readme)
    out: list[ServeCandidate] = []
    seen: set[str] = set()
    for start, _lang, body in _extract_code_blocks(readme):
        if not _is_sglang_block(body):
            continue
        cand = _parse_one_sglang_command(_collapse_continuations(body))
        if not cand:
            continue
        key = re.sub(r"\s+", " ", cand.raw)[:200]
        if key in seen:
            continue
        seen.add(key)
        cand.section = _section_at(heads, start)
        out.append(cand)
    for m in _SGLANG_LAUNCH_RE.finditer(readme):
        cand = _parse_one_sglang_command(readme[m.start() : m.start() + 800])
        if not cand:
            continue
        key = re.sub(r"\s+", " ", cand.raw)[:200]
        if key in seen:
            continue
        seen.add(key)
        cand.section = _section_at(heads, m.start())
        out.append(cand)
    out.sort(key=lambda c: c.score, reverse=True)
    return out


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

    # Cards ship per-GPU recipes. One headed for other silicon is the wrong path on a
    # Spark even when its flag list scores well (NVFP4 cards carry a W4A16 Ampere
    # fallback whose flags out-score the real GB10 recipe). Live heading
    # "W4A16 (vLLM)" used to take the +35 vLLM bonus and skip the Ampere penalty.
    spark_section = _section_is_spark(sec)
    other_silicon = _section_is_other_silicon(sec)

    # Prefer dedicated vLLM / Spark sections — not "W4A16 (vLLM)" / "8x H100 (vLLM)".
    if not other_silicon and (
        "vllm run" in sec or sec.strip() in ("vllm", "vllm run instructions") or spark_section
    ):
        score += 50
        reasons.append(f"in section «{c.section}»")
    elif "vllm" in sec and not other_silicon:
        score += 35
        reasons.append(f"in section «{c.section}»")
    elif any(k in sec for k in ("deploy", "inference", "serving", "quickstart", "quick start", "usage")):
        score += 15
        reasons.append(f"in section «{c.section}»")

    if not spark_section and other_silicon:
        score -= 30
        reasons.append(f"section «{c.section}» targets non-Spark hardware")

    # Performance / Spark-relevant flags from the card
    moe = (cfg.get("moe_backend") or "").strip()
    flashinfer_unsafe = _flashinfer_b12x_unsafe_for_checkpoint(det, cfg.get("image"))
    marlin_unsafe = _marlin_unsafe_for_checkpoint(det, cfg.get("image"))
    if moe:
        if moe == "humming":
            # Ampere leftover — stripped on GB10; do not let it out-score Spark/DSpark.
            reasons.append(
                "--moe-backend humming (Ampere / non-GB10 — will be cleared on Spark)"
            )
        elif moe == "flashinfer_b12x" and flashinfer_unsafe:
            reasons.append(f"--moe-backend {moe} (will be stripped — unsafe for checkpoint)")
        elif moe == "marlin" and marlin_unsafe:
            reasons.append(
                f"--moe-backend {moe} (will be stripped — unsupported for this MoE on selected image)"
            )
        else:
            score += 25
            reasons.append(f"--moe-backend {moe}")
    has_cute = any(e.startswith("CUTE_DSL_ARCH=") for e in cfg.get("docker_env") or [])
    if has_cute:
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
    extra_l = (cfg.get("extra_flags") or "").lower()
    if "--attention-backend" in extra_l and "flashinfer" in extra_l:
        score += 12
        reasons.append("Spark --attention-backend flashinfer")
    if cfg.get("mtp") or "speculative-config" in (cfg.get("extra_flags") or ""):
        # MTP is optional — never preferred as first-boot default over stable recipes
        score -= 25
        reasons.append("speculative/MTP (optional — not default first boot)")

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

    # Bare `vllm serve model` with no flags: weak default when checkpoint needs flags
    bare = (
        not cfg.get("quantization")
        and not cfg.get("moe_backend")
        and not cfg.get("docker_env")
        and len(c.args) == 0
    )
    if bare:
        if det.get("quant_flag") or det.get("is_moe") or det.get("has_nvfp4"):
            score -= 20
            reasons.append(
                "bare serve — weak default when config.json needs quant/MoE (prefer Spark recipe)"
            )
        else:
            score += 1
            reasons.append("minimal serve (defaults)")
    elif not cfg.get("quantization") and not cfg.get("moe_backend") and not cfg.get("docker_env"):
        score += 3

    # Card prose near CUTE_DSL / flashinfer often marks the recommended path
    if "cute_dsl" in raw_l or "CUTE_DSL" in c.raw:
        score += 5

    # Prefer recipes that already set quantization for ModelOpt / NVFP4 MoE
    if _quant_flags_compatible(str(det.get("quant_flag") or ""), str(cfg.get("quantization") or "")):
        score += 15
        reasons.append(f"quantization matches config.json ({det.get('quant_flag')})")

    # Checkpoint truth from config.json: mixed FP8+NVFP4 MoE rejects flashinfer_b12x on vLLM 0.25.x
    if flashinfer_unsafe and moe == "flashinfer_b12x":
        score -= 80
        reasons.append(
            "PENALTY: config.json has FP8 MoE layers — flashinfer_b12x crashes "
            "(ValueError: not supported for FP8 MoE)"
        )
        # Salvage: Spark section + CUTE_DSL still mark the right path after moe strip.
        if has_cute or "dgx spark" in sec or "spark" in sec:
            score += 55
            reasons.append(
                "salvage: Spark/CUTE path kept after stripping unsafe flashinfer_b12x"
            )
    elif moe == "flashinfer_b12x" and not det:
        pass

    # Marlin is on many NVIDIA cards but crashes MoE on vLLM 0.25 (unquantized MoE path)
    if marlin_unsafe and moe == "marlin":
        score -= 80
        reasons.append(
            "PENALTY: moe_backend=marlin unsupported for this MoE on selected image "
            "(ValueError: not supported for unquantized MoE)"
        )
        # Salvage mirrors the flashinfer_b12x case above: the rest of a vendor GB10
        # recipe is still the right path once marlin is stripped to auto.
        if has_cute or spark_section:
            score += 55
            reasons.append(
                "salvage: Spark recipe kept after stripping unsupported marlin"
            )

    return score, reasons


def _flashinfer_b12x_unsafe_for_checkpoint(
    detected: dict[str, Any],
    image: str | None = None,
) -> bool:
    """True when forcing flashinfer_b12x will crash on this checkpoint×image.

    Unsloth's Aug 2026 DGX Spark recipe *requires* ``--moe-backend flashinfer_b12x``
    plus ``CUTE_DSL_ARCH=sm_121a`` on vLLM ≥0.27 (lab default ``v0.27.1``) once
    FlashInfer ships b12x MoE kernels. Older 0.25.x images still raise
    ``moe_backend=flashinfer_b12x is not supported for FP8 MoE`` on mixed
    compressed-tensors checkpoints.
    """
    if not detected:
        return False
    # flashinfer_b12x is an MoE kernel. Dense 31B Gemma must not keep a 26B-A4B Spark line.
    if not detected.get("is_moe"):
        return True
    # Unknown/custom/Anemll and stock ≥0.27: keep the vendor Spark flag.
    if _image_at_least(image, 0, 27, 0):
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
    # ModelOpt MIXED_PRECISION / dual-algo layers also route FP8 MoE experts.
    qf = (detected.get("quant_flag") or "").lower()
    if detected.get("is_moe") and detected.get("has_fp8") and qf.startswith("modelopt"):
        return True
    return False


def _modelopt_quant_flag(
    quant_method: str,
    quant_algo: str,
    *,
    has_nvfp4: bool,
    has_fp8: bool,
    has_modelopt_layers: bool,
) -> str:
    """Map umbrella ModelOpt metadata to the vLLM override sibling (0.27+)."""
    method = (quant_method or "").lower()
    algo = (quant_algo or "").upper()
    if method in ("modelopt_fp4", "modelopt_mixed", "modelopt_mxfp8"):
        return method
    if "MIXED" in algo:
        return "modelopt_mixed"
    if any(tok in algo for tok in ("NVFP4", "FP4", "W4A16")):
        return "modelopt_fp4"
    if algo == "FP8" or algo.startswith("FP8"):
        return "modelopt"
    if has_nvfp4 and has_fp8 and has_modelopt_layers:
        return "modelopt_mixed"
    if has_nvfp4 and not has_fp8:
        return "modelopt_fp4"
    if method.startswith("modelopt"):
        return "modelopt"
    return "modelopt"


def _quant_flags_compatible(detected_flag: str, candidate_flag: str) -> bool:
    """True when candidate --quantization matches (or is a ModelOpt sibling of) detected."""
    a = (detected_flag or "").lower()
    b = (candidate_flag or "").lower()
    if not a or not b:
        return False
    if a == b:
        return True
    modelopt = {"modelopt", "modelopt_fp4", "modelopt_mixed", "modelopt_mxfp8"}
    return a in modelopt and b in modelopt


def _marlin_unsafe_for_checkpoint(
    detected: dict[str, Any],
    image: str | None = None,
) -> bool:
    """True when --moe-backend marlin will crash on this checkpoint×image.

    Observed on NVIDIA Qwen3.6-35B-A3B-NVFP4 (ModelOpt MoE) under vLLM 0.25:
      ValueError: moe_backend='marlin' is not supported for unquantized MoE.

    Nemotron hybrid Spark cards *require* marlin on GB10. Pure NVFP4 MoE
    (including Qwen) is legal on ≥0.27; on older images treat non-Nemotron
    marlin as unsafe. compressed-tensors mixed FP8+NVFP4 still leaves moe
    auto. ModelOpt MIXED_PRECISION Spark playbooks (NVIDIA Qwen3.6-35B)
    keep marlin on ≥0.27. Qwen MoE without a proven NVFP4 layout still
    rejects marlin.
    """
    if not detected:
        return True
    family = detected.get("family") or ""
    if family == "nemotron":
        return False
    if detected.get("is_mixed_nvfp4_fp8"):
        qf = (detected.get("quant_flag") or "").lower()
        # NVIDIA Qwen3.6-35B-A3B-NVFP4 playbook: marlin on vLLM ≥0.27.
        if qf.startswith("modelopt") and _image_at_least(image, 0, 27, 0):
            return False
        return True
    # Pure NVFP4 MoE (including Qwen): keep marlin on ≥0.27 (SM121).
    if detected.get("has_nvfp4") and not detected.get("is_mixed_nvfp4_fp8"):
        if _image_at_least(image, 0, 27, 0):
            return False
        return True
    if family == "qwen":
        return True
    if detected.get("is_moe"):
        return True
    qf = (detected.get("quant_flag") or "").lower()
    if qf.startswith("modelopt") or qf == "compressed-tensors":
        return True
    return False


def _sanitize_moe_backend_on_candidate(
    c: "ServeCandidate",
    detected: dict[str, Any] | None,
) -> None:
    """Clear moe backends that will crash when the UI Applies this recipe."""
    det = detected or {}
    moe = (c.config.get("moe_backend") or "").strip().lower()
    if not moe:
        return
    if moe == "flashinfer_b12x" and _flashinfer_b12x_unsafe_for_checkpoint(
        det, c.config.get("image")
    ):
        c.config["moe_backend"] = ""
        if not any("cleared moe_backend" in r for r in c.reasons):
            c.reasons.append(
                "cleared moe_backend=flashinfer_b12x on this recipe (unsafe for checkpoint×image)"
            )
    elif moe == "marlin" and _marlin_unsafe_for_checkpoint(det, c.config.get("image")):
        c.config["moe_backend"] = ""
        if not any("cleared moe_backend" in r for r in c.reasons):
            c.reasons.append(
                "cleared moe_backend=marlin on this recipe "
                "(unsupported for this MoE path on the selected image — use auto)"
            )


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
    if not _flashinfer_b12x_unsafe_for_checkpoint(detected, cfg.get("image")):
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
    img = cfg.get("image")
    if moe == "flashinfer_b12x" and _flashinfer_b12x_unsafe_for_checkpoint(detected, img):
        cfg["moe_backend"] = ""
        warnings.append(
            "Card recipe used --moe-backend flashinfer_b12x, but this image still "
            "crashes mixed FP8 + NVFP4 MoE (compressed-tensors) with: "
            "'moe_backend=flashinfer_b12x is not supported for FP8 MoE'. "
            "Cleared moe-backend so vLLM auto-selects. On vLLM ≥0.27 the Spark "
            "Unsloth recipe keeps flashinfer_b12x."
        )
        rationale.append(
            "SAFETY (image <0.27 × mixed FP8 MoE): removed flashinfer_b12x"
        )
        # Keep cute-DSL if card mentioned it
        env = list(cfg.get("docker_env") or [])
        if not any(e.startswith("CUTE_DSL_ARCH=") for e in env):
            env.append("CUTE_DSL_ARCH=sm_121a")
            cfg["docker_env"] = env
            rationale.append("Kept/added CUTE_DSL_ARCH=sm_121a from card Spark guidance")
    elif moe == "flashinfer_b12x":
        rationale.append(
            "Spark Unsloth/NVIDIA path: keep --moe-backend flashinfer_b12x on vLLM ≥0.27"
        )
        env = list(cfg.get("docker_env") or [])
        if not any(e.startswith("CUTE_DSL_ARCH=") for e in env):
            env.append("CUTE_DSL_ARCH=sm_121a")
            cfg["docker_env"] = env
            rationale.append("Unsloth Spark recipe → CUTE_DSL_ARCH=sm_121a")

    moe = (cfg.get("moe_backend") or "").strip()
    if moe == "marlin" and _marlin_unsafe_for_checkpoint(detected, cfg.get("image")):
        cfg["moe_backend"] = ""
        warnings.append(
            "Card recipe used --moe-backend marlin, but this image rejects marlin on this MoE "
            "path (ValueError: moe_backend='marlin' is not supported for unquantized MoE). "
            "Cleared moe-backend so vLLM auto-selects a supported backend."
        )
        rationale.append(
            "SAFETY (image capability > card flag): removed marlin MoE backend — use auto"
        )

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


_VL_MM_LIMIT = "--limit-mm-per-prompt '{\"image\":4,\"video\":1}'"


def _apply_vl_spark_defaults(
    cfg: dict[str, Any],
    detected: dict[str, Any],
    warnings: list[str],
    rationale: list[str],
    *,
    mode: str | None = None,
) -> None:
    """Keep vision on by default with an MM cap. ``mode`` is ignored (legacy)."""
    if not detected.get("is_vl"):
        return
    ex = cfg.get("extra_flags") or ""
    if "--language-model-only" in ex:
        ex = _strip_flag_from_extra(ex, "--language-model-only")
        cfg["extra_flags"] = ex
        rationale.append("Keep vision on — drop --language-model-only")
    ex = cfg.get("extra_flags") or ""
    if "--limit-mm-per-prompt" not in ex:
        cfg["extra_flags"] = (ex + " " + _VL_MM_LIMIT).strip()
        rationale.append("VL → keep vision/video, cap with --limit-mm-per-prompt")


def _apply_first_boot_defaults(
    cfg: dict[str, Any],
    *,
    mode: str,
    detected: dict[str, Any],
    warnings: list[str],
    rationale: list[str],
) -> None:
    """Make Auto-configure → Start serve work on first try (Spark lab posture).

    Cards often ship kitchen-sink demos (MTP + exotic moe backends). First boot
    should be stable: correct quant, safe moe auto. Playbook MTP stays when the
    surviving moe backend is known-safe for this checkpoint×image.
    """
    # Empty moe = vLLM auto — preferred on Spark unless user/card forces a known-good backend
    moe = (cfg.get("moe_backend") or "").strip().lower()
    img = cfg.get("image")
    if moe == "flashinfer_b12x" and _flashinfer_b12x_unsafe_for_checkpoint(detected, img):
        cfg["moe_backend"] = ""
    elif moe == "marlin" and _marlin_unsafe_for_checkpoint(detected, img):
        cfg["moe_backend"] = ""

    # Prefer empty moe for MoE first boot even if card set something exotic we didn't list.
    # Keep backends that are known-safe for this checkpoint×image (incl. Nemotron marlin
    # and Unsloth Spark flashinfer_b12x on ≥0.27).
    moe = (cfg.get("moe_backend") or "").strip().lower()
    allowed = {"", "auto", "triton", "flashinfer_trtllm", "flashinfer_cutlass", "aiter"}
    if not _marlin_unsafe_for_checkpoint(detected, img):
        allowed.add("marlin")
    if not _flashinfer_b12x_unsafe_for_checkpoint(detected, img):
        allowed.add("flashinfer_b12x")
    if detected.get("is_moe") and moe and moe not in allowed:
        cfg["moe_backend"] = ""
        if moe not in ("marlin", "flashinfer_b12x"):
            warnings.append(
                f"Cleared --moe-backend {moe} for first-boot MoE safety (vLLM auto)."
            )
            rationale.append(f"FIRST BOOT: moe-backend {moe!r} → empty (auto)")

    moe = (cfg.get("moe_backend") or "").strip().lower()
    # NVIDIA Spark playbook ships MTP 3 with marlin. Keep it when that backend
    # is still legal. Unsloth's winning Spark recipe has no MTP, so first boot
    # stays MTP-off there.
    keep_playbook_mtp = bool(cfg.get("mtp")) and moe in {"marlin", "flashinfer_b12x"}
    if cfg.get("mtp") and not keep_playbook_mtp:
        cfg["mtp"] = False
        cfg["mtp_num_tokens"] = 2
        cfg["mtp_moe_backend"] = ""
        if cfg.get("extra_flags"):
            cfg["extra_flags"] = _strip_flag_from_extra(cfg["extra_flags"], "--speculative-config")
        warnings.append(
            "Disabled MTP / speculative decode for first boot (card had it on). "
            "Re-enable MTP in the form after a healthy serve if you want it."
        )
        rationale.append("FIRST BOOT: MTP off (stable serve; re-enable later if needed)")
    elif keep_playbook_mtp:
        spec_extra = _mtp_spark_extra(_speculative_json_from_extra(cfg.get("extra_flags") or ""))
        if spec_extra.get("moe_backend") and not (cfg.get("mtp_moe_backend") or "").strip():
            cfg["mtp_moe_backend"] = str(spec_extra["moe_backend"])
        rationale.append(
            f"Playbook MTP kept (moe-backend={moe}, num_speculative_tokens="
            f"{cfg.get('mtp_num_tokens')}"
            + (
                f", mtp_moe_backend={cfg.get('mtp_moe_backend')}"
                if cfg.get("mtp_moe_backend")
                else ""
            )
            + ")"
        )

    # Long card contexts are clamped by _apply_mode_envelope on single-node Spark.
    # Multi-node overlays (TP>=2) keep their pin.


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
    quant_algo = str(qc.get("quant_algo") or "")
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

    # Top-level ModelOpt algo (even with empty quantized_layers).
    algo_u = quant_algo.upper()
    if any(tok in algo_u for tok in ("NVFP4", "FP4", "W4A16")):
        has_nvfp4 = True
    if algo_u == "FP8" or algo_u.startswith("FP8") or "MIXED" in algo_u:
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
    # Prefer token boundaries over substring "fp8" (avoids odd id false positives).
    if quant_method == "fp8" or re.search(r"(^|[-_/])fp8($|[-_/])", mid):
        has_fp8 = True

    fmt_blob = " ".join(formats)
    if "nvfp4" in fmt_blob:
        has_nvfp4 = True
    if "float-quantized" in fmt_blob:
        has_fp8 = True

    if quant_method in ("compressed-tensors", "compressed_tensors"):
        quant_flag = "compressed-tensors"
    elif quant_method.startswith("modelopt") or (
        has_modelopt_layers and (has_nvfp4 or has_fp8 or quant_algo)
    ):
        quant_flag = _modelopt_quant_flag(
            quant_method or "modelopt",
            quant_algo,
            has_nvfp4=has_nvfp4,
            has_fp8=has_fp8,
            has_modelopt_layers=has_modelopt_layers,
        )
    elif quant_method == "fp8":
        quant_flag = "fp8"
    elif "modelopt" in tags:
        quant_flag = _modelopt_quant_flag(
            "modelopt",
            quant_algo,
            has_nvfp4=has_nvfp4,
            has_fp8=has_fp8,
            has_modelopt_layers=has_modelopt_layers,
        )
    elif "compressed-tensors" in tags:
        quant_flag = "compressed-tensors"
    elif has_nvfp4 and "nvfp4-pack" in fmt_blob:
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
    arch_blob = " ".join(str(a).lower() for a in architectures)
    is_vl = bool(
        "vl" in model_type
        or "vision" in model_type
        or re.search(r"(^|[_-])vl([_-]|$)", mid)
        or "vl" in arch_blob
        or (
            "conditionalgeneration" in arch_blob
            and ("qwen3" in model_type or "qwen3" in mid or "qwen3" in arch_blob)
        )
        or "llava" in mid
        or "pixtral" in mid
        or "internvl" in mid
    )
    is_mixed = has_nvfp4 and has_fp8 and (
        quant_method in ("compressed-tensors", "compressed_tensors")
        or "mixed" in fmt_blob
        or ("nvfp4-pack" in fmt_blob and "float-quantized" in fmt_blob)
        or "MIXED" in algo_u
        or (has_modelopt_layers and has_nvfp4 and has_fp8)
        or quant_flag == "modelopt_mixed"
    )

    # Checkpoint-declared KV preference (ModelOpt kv_cache_scheme / kv_cache_quant_algo).
    suggested_kv = None
    kv_scheme = qc.get("kv_cache_scheme")
    kv_algo = str(qc.get("kv_cache_quant_algo") or "").upper()
    if isinstance(kv_scheme, dict):
        bits = kv_scheme.get("num_bits")
        typ = str(kv_scheme.get("type") or "").lower()
        if bits == 8 or "float" in typ or "fp8" in typ:
            suggested_kv = "fp8"
    if "FP8" in kv_algo:
        suggested_kv = "fp8"

    family = "unknown"
    blob = f"{model_type} {' '.join(str(a) for a in architectures)} {model_id}".lower()
    # Most-specific first — MiniMax M3 / DeepSeek V4 must not collapse into siblings.
    if "nemotron" in blob:
        family = "nemotron"
    elif "minimax" in blob and ("m3" in blob or "minimaxm3" in blob):
        family = "minimax_m3"
    elif "minimax" in blob:
        family = "minimax_m2"
    elif "deepseek" in blob and ("v4" in blob or "dspark" in blob):
        family = "deepseek_v4"
    elif "deepseek" in blob and "r1" in blob:
        family = "deepseek_r1"
    elif "deepseek" in blob:
        family = "deepseek_v3"
    elif "qwen" in blob:
        family = "qwen"
    elif "mistral" in blob or "mixtral" in blob or "magistral" in blob or "devstral" in blob or "pixtral" in blob:
        family = "mistral"
    elif "glm" in blob or "chatglm" in blob:
        family = "glm"
    elif "kimi" in blob or "moonshot" in blob:
        family = "kimi"
    elif "granite" in blob:
        family = "granite"
    elif "phi" in blob or model_type.startswith("phi"):
        family = "phi"
    elif "llama" in blob:
        # Llama 4 Scout/Maverick vs 3.x — tool parsers differ.
        family = "llama4" if ("llama4" in blob or "llama-4" in mid or "scout" in mid or "maverick" in mid) else "llama"
    elif "diffusiongemma" in blob:
        family = "diffusiongemma"
    elif "gemma" in blob:
        family = "gemma4" if ("gemma4" in blob or "gemma-4" in mid or "gemma_4" in mid) else "gemma"
    elif "internlm" in blob:
        family = "internlm"
    elif "hunyuan" in blob or re.search(r"(^|[^a-z])hy3", blob) or "hy_v3" in blob:
        family = "hy_v3" if (
            "hy3" in blob or "hy_v3" in blob or "hy-v3" in mid or "hunyuan-v3" in mid
        ) else "hunyuan"
    elif re.search(r"step[\s._-]*3", blob) or "step3" in blob:
        family = "step3p5" if (
            "step3p5" in blob or "step-3.5" in mid or "step3.5" in mid or "3.5" in mid
        ) else "step3"
    elif "ernie" in blob:
        family = "ernie"

    return {
        "model_type": model_type or None,
        "architectures": architectures,
        "quant_method": quant_method or None,
        "quant_algo": quant_algo or None,
        "quant_formats": sorted(formats),
        "has_nvfp4": has_nvfp4,
        "has_fp8": has_fp8,
        "is_mixed_nvfp4_fp8": is_mixed,
        "is_moe": is_moe,
        "is_vl": is_vl,
        "has_modelopt_layers": has_modelopt_layers,
        "max_position_embeddings": max_pos,
        "quant_flag": quant_flag,
        "suggested_kv_cache_dtype": suggested_kv,
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
        "mtp_moe_backend": "",
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
    """Apply winning card recipe onto base (card wins for set fields).

    Image is *not* copied here — it is resolved via ``_resolve_image_for_gates``
    (lab floor + stock raise only; Anemll never replaced) before safety/marlin.
    Refuses a snippet that names a different model than ``base['model']``.
    """
    want = str(base.get("model") or "")
    if want and _vendor_candidate_sibling_mismatch(cand, want):
        return []
    applied: list[str] = []
    cfg = cand.config
    for k, v in cfg.items():
        if k == "image":
            # Keep on candidate.config for scoring/UI; resolve into base separately.
            continue
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


def _harvest_tool_flags_from_candidates(
    cfg: dict[str, Any],
    candidates: list[ServeCandidate],
    rationale: list[str],
) -> None:
    """If the winning recipe omitted tools, pull tool parser from a tool-focused alt.

    Nano-9B-v2 cards ship a bare serve line (high score) and a separate tool-calling
    block with ``nemotron_json`` + ``--tool-parser-plugin`` — prefer those when present.
    """
    if (cfg.get("tool_call_parser") or "").strip():
        return
    if not candidates:
        return
    want_model = str(cfg.get("model") or "")
    ranked: list[tuple[int, int, float, ServeCandidate]] = []
    for c in candidates:
        if _vendor_candidate_sibling_mismatch(c, want_model):
            continue
        tcp = str((c.config or {}).get("tool_call_parser") or "").strip()
        if not tcp:
            continue
        sec = (c.section or "").lower()
        raw_l = (c.raw or "").lower()
        toolish = (
            "tool" in sec
            or "tool" in raw_l
            or bool((c.config or {}).get("enable_auto_tool_choice"))
            or tcp in ("nemotron_json", "llama_nemotron_json")
        )
        ranked.append(
            (
                1 if tcp == "nemotron_json" else 0,
                1 if toolish else 0,
                float(c.score or 0.0),
                c,
            )
        )
    if not ranked:
        return
    ranked.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    pick = ranked[0][3]
    tcp = str(pick.config.get("tool_call_parser") or "").strip()
    if not tcp:
        return
    cfg["tool_call_parser"] = tcp
    if pick.config.get("enable_auto_tool_choice"):
        cfg["enable_auto_tool_choice"] = True
    # Keep --tool-parser-plugin when the card recipe provides it.
    add_ex = (pick.config.get("extra_flags") or "").strip()
    if "--tool-parser-plugin" in add_ex:
        try:
            parts = shlex.split(add_ex)
        except ValueError:
            parts = add_ex.split()
        plugin_bits: list[str] = []
        i = 0
        while i < len(parts):
            if parts[i] == "--tool-parser-plugin" or parts[i].startswith(
                "--tool-parser-plugin="
            ):
                if parts[i].startswith("--tool-parser-plugin="):
                    plugin_bits.append(parts[i])
                    i += 1
                else:
                    plugin_bits.append(parts[i])
                    if i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                        plugin_bits.append(parts[i + 1])
                        i += 2
                    else:
                        i += 1
            else:
                i += 1
        if plugin_bits:
            plugin_s = " ".join(
                shlex.quote(x) if (" " in x or "{" in x) else x for x in plugin_bits
            )
            cfg["extra_flags"] = _merge_extra_flags(cfg.get("extra_flags") or "", plugin_s)
    rationale.append(
        f"Tool parser from card alt recipe (score {pick.score:.0f}"
        + (f", «{pick.section}»" if pick.section else "")
        + f"): {tcp}"
    )


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
    # Prefer more-specific tool parsers when both appear on a card.
    if re.search(r"tool.?call.?parser\s+[\"']?qwen3_xml", readme, re.I):
        hints["tool_call_parser"] = "qwen3_xml"
        hints["enable_auto_tool_choice"] = True
    elif re.search(r"tool.?call.?parser\s+[\"']?nemotron_json", readme, re.I):
        hints["tool_call_parser"] = "nemotron_json"
        hints["enable_auto_tool_choice"] = True
    elif re.search(r"tool.?call.?parser\s+qwen3_coder", readme, re.I):
        hints["tool_call_parser"] = "qwen3_coder"
        hints["enable_auto_tool_choice"] = True
    if re.search(r"--quantization\s+modelopt_mixed\b", readme):
        hints["quantization"] = "modelopt_mixed"
    elif re.search(r"--quantization\s+modelopt_fp4\b", readme):
        hints["quantization"] = "modelopt_fp4"
    elif re.search(r"--quantization\s+modelopt\b", readme):
        hints["quantization"] = "modelopt"
    if re.search(r"--quantization\s+compressed-tensors", readme):
        hints["quantization"] = "compressed-tensors"
    if re.search(r"--quantization\s+fp8\b", readme):
        hints["quantization"] = "fp8"
    if re.search(r"fp8\s+kv(\s+cache)?", readme, re.I) or re.search(
        r"kv[\s-]?cache[^\n]{0,40}fp8", readme, re.I
    ):
        hints["kv_cache_dtype"] = "fp8"
        hints["notes"].append("Vendor/card prose: FP8 KV cache")
    if re.search(r"--trust-remote-code\b", readme, re.I):
        hints["trust_remote_code"] = True
    return hints


def _fill_from_config_detection(
    base: dict[str, Any],
    detected: dict[str, Any],
    rationale: list[str],
) -> None:
    """Only fill gaps using HF config.json / API tags — not lab override recipes."""
    det_q = (detected.get("quant_flag") or "").strip()
    cur_q = (base.get("quantization") or "").strip()
    if det_q:
        if not cur_q:
            base["quantization"] = det_q
            rationale.append(
                f"HF config/tags → --quantization {det_q} "
                f"(card serve line had no --quantization)"
            )
        elif cur_q.lower() != det_q.lower() and (
            detected.get("quant_algo")
            or detected.get("has_modelopt_layers")
            or detected.get("is_mixed_nvfp4_fp8")
            or detected.get("quant_method")
        ):
            # config.json is ground truth for layout: ModelOpt siblings, and
            # card compressed-tensors vs config modelopt_* (or the reverse).
            base["quantization"] = det_q
            rationale.append(
                f"HF config.json overrides card/prose quant {cur_q} → {det_q}"
            )
    if not base.get("kv_cache_dtype") and detected.get("suggested_kv_cache_dtype"):
        base["kv_cache_dtype"] = detected["suggested_kv_cache_dtype"]
        rationale.append(
            f"HF quantization_config → --kv-cache-dtype {detected['suggested_kv_cache_dtype']}"
        )

    family = detected.get("family")
    if family == "qwen":
        mid = (
            str(base.get("model") or "")
            + " "
            + str(detected.get("model_type") or "")
            + " "
            + " ".join(str(a) for a in (detected.get("architectures") or []))
        ).lower()
        # qwen3 parsers break Qwen2.5 / classic Qwen2 chat templates.
        is_qwen25_or_2 = bool(
            re.search(r"qwen2\.5|qwen2_5|qwen25", mid)
            or ("qwen2" in mid and "qwen3" not in mid and "qwen2.5" not in mid)
        )
        if is_qwen25_or_2:
            if not base.get("trust_remote_code"):
                base["trust_remote_code"] = True
            rationale.append(
                "Qwen2.5/Qwen2 checkpoint → skipping qwen3 reasoning/tool parsers (card may still set them)"
            )
        else:
            if not base.get("reasoning_parser"):
                base["reasoning_parser"] = "qwen3"
                rationale.append("Qwen architecture (from HF config) → --reasoning-parser qwen3")
            if not base.get("tool_call_parser"):
                # vLLM Coder checkpoints use qwen3_xml; chat/reasoners use qwen3_coder.
                tool = "qwen3_xml" if re.search(r"(^|[^a-z])coder([^a-z]|$)", mid) else "qwen3_coder"
                base["tool_call_parser"] = tool
                base["enable_auto_tool_choice"] = True
                rationale.append(f"Qwen architecture → tool-call-parser {tool} + auto tool choice")
            if not base.get("trust_remote_code"):
                base["trust_remote_code"] = True
            if not base.get("kv_cache_dtype") and (
                detected.get("has_nvfp4") or detected.get("suggested_kv_cache_dtype") == "fp8"
            ):
                base["kv_cache_dtype"] = "fp8"
                rationale.append("Qwen NVFP4 / calibrated KV → --kv-cache-dtype fp8")
    elif family == "nemotron":
        mid = (
            str(base.get("model") or "")
            + " "
            + str(detected.get("model_type") or "")
            + " "
            + " ".join(str(a) for a in (detected.get("architectures") or []))
        ).lower()
        # Nano-9B/12B-v2: card uses nemotron_json (+ plugin); not Lightning's qwen3_coder.
        is_nano_v2 = bool(
            re.search(r"nano[-_]?(9|12)b[-_]?v2", mid)
            or re.search(r"nano[-_]?v2", mid)
        )
        if not base.get("reasoning_parser") and not is_nano_v2:
            base["reasoning_parser"] = "nemotron_v3"
            rationale.append("Nemotron (from HF) → --reasoning-parser nemotron_v3")
        base["enable_auto_tool_choice"] = True
        if not base.get("tool_call_parser"):
            if is_nano_v2:
                base["tool_call_parser"] = "nemotron_json"
                rationale.append(
                    "Nemotron Nano-v2 → tool-call-parser nemotron_json "
                    "(card plugin path optional via --tool-parser-plugin)"
                )
            else:
                base["tool_call_parser"] = "qwen3_coder"
        base["trust_remote_code"] = True
    elif family == "minimax_m2":
        if not base.get("reasoning_parser"):
            base["reasoning_parser"] = "minimax_m2"
            rationale.append("MiniMax M2 → --reasoning-parser minimax_m2")
        if not base.get("tool_call_parser"):
            base["tool_call_parser"] = "minimax_m2"
            base["enable_auto_tool_choice"] = True
            rationale.append("MiniMax M2 → tool-call-parser minimax_m2 + auto tool choice")
        base["trust_remote_code"] = True
    elif family == "minimax_m3":
        if not base.get("reasoning_parser"):
            base["reasoning_parser"] = "minimax_m3"
            rationale.append("MiniMax M3 → --reasoning-parser minimax_m3")
        if not base.get("tool_call_parser"):
            base["tool_call_parser"] = "minimax_m3"
            base["enable_auto_tool_choice"] = True
            rationale.append("MiniMax M3 → tool-call-parser minimax_m3 + auto tool choice")
        base["trust_remote_code"] = True
        # MSA sparse attention requires block-size 128; refuse is a placement concern.
        ex = base.get("extra_flags") or ""
        if "--block-size" not in ex:
            base["extra_flags"] = (ex + " --block-size 128").strip()
            rationale.append("MiniMax M3 MSA → --block-size 128")
    elif family == "mistral":
        if not base.get("tool_call_parser"):
            base["tool_call_parser"] = "mistral"
            base["enable_auto_tool_choice"] = True
            rationale.append("Mistral family → tool-call-parser mistral + auto tool choice")
        mid = (detected.get("model_type") or "") + " " + " ".join(
            str(a) for a in (detected.get("architectures") or [])
        )
        # Magistral / reasoning variants use the mistral reasoning parser.
        if "magistral" in mid.lower() or "reasoning" in mid.lower():
            if not base.get("reasoning_parser"):
                base["reasoning_parser"] = "mistral"
                rationale.append("Mistral reasoning variant → --reasoning-parser mistral")
        if not base.get("load_format"):
            base["load_format"] = "mistral"
            rationale.append("Mistral family → --load-format mistral")
        ex = base.get("extra_flags") or ""
        if "--tokenizer-mode" not in ex:
            base["extra_flags"] = (ex + " --tokenizer-mode mistral --config-format mistral").strip()
    elif family == "deepseek_r1":
        if not base.get("reasoning_parser"):
            base["reasoning_parser"] = "deepseek_r1"
            rationale.append("DeepSeek-R1 → --reasoning-parser deepseek_r1")
        base["trust_remote_code"] = True
    elif family == "deepseek_v3":
        if not base.get("reasoning_parser"):
            base["reasoning_parser"] = "deepseek_v3"
            rationale.append("DeepSeek-V3 → --reasoning-parser deepseek_v3")
        if not base.get("tool_call_parser"):
            base["tool_call_parser"] = "deepseek_v3"
            base["enable_auto_tool_choice"] = True
        base["trust_remote_code"] = True
    elif family == "deepseek_v4":
        # Overlay usually owns this; fill gaps if card/overlay missed parsers.
        if not base.get("reasoning_parser"):
            base["reasoning_parser"] = "deepseek_v4"
            rationale.append("DeepSeek-V4 → --reasoning-parser deepseek_v4")
        if not base.get("tool_call_parser"):
            base["tool_call_parser"] = "deepseek_v4"
            base["enable_auto_tool_choice"] = True
        base["trust_remote_code"] = True
    elif family == "glm":
        mid = (
            (detected.get("model_type") or "")
            + " "
            + " ".join(str(a) for a in (detected.get("architectures") or []))
            + " "
            + str(base.get("model") or "")
        ).lower()
        # Reasoning always glm45 for MoE 4.x; tools split by generation.
        if not base.get("reasoning_parser"):
            base["reasoning_parser"] = "glm45"
            rationale.append("GLM family → --reasoning-parser glm45")
        tool = "glm47" if ("4.7" in mid or "glm47" in mid or "flash" in mid) else "glm45"
        if not base.get("tool_call_parser"):
            base["tool_call_parser"] = tool
            base["enable_auto_tool_choice"] = True
            rationale.append(f"GLM family → tool-call-parser {tool}")
        base["trust_remote_code"] = True
    elif family == "kimi":
        # Prefer K3 parser when the id/arch says so; else K2.
        mid = (detected.get("model_type") or "") + " " + " ".join(
            str(a) for a in (detected.get("architectures") or [])
        )
        kimi = "kimi_k3" if "k3" in mid.lower() else "kimi_k2"
        if not base.get("reasoning_parser"):
            base["reasoning_parser"] = kimi
            rationale.append(f"Kimi → --reasoning-parser {kimi}")
        if not base.get("tool_call_parser"):
            base["tool_call_parser"] = kimi
            base["enable_auto_tool_choice"] = True
        base["trust_remote_code"] = True
    elif family == "llama4":
        if not base.get("tool_call_parser"):
            base["tool_call_parser"] = "llama4_pythonic"
            base["enable_auto_tool_choice"] = True
            rationale.append("Llama 4 → tool-call-parser llama4_pythonic + auto tool choice")
        # No Meta Llama reasoning parser.
    elif family == "llama":
        if not base.get("tool_call_parser"):
            base["tool_call_parser"] = "llama3_json"
            base["enable_auto_tool_choice"] = True
            rationale.append("Llama 3.x → tool-call-parser llama3_json + auto tool choice")
    elif family in ("gemma4", "diffusiongemma"):
        if not base.get("reasoning_parser"):
            base["reasoning_parser"] = "gemma4"
            rationale.append("Gemma 4 → --reasoning-parser gemma4")
        if not base.get("tool_call_parser"):
            base["tool_call_parser"] = "gemma4"
            base["enable_auto_tool_choice"] = True
            rationale.append("Gemma 4 → tool-call-parser gemma4 + auto tool choice")
    elif family == "gemma":
        # Gemma 2/3: no gemma4 parsers; leave empty unless the card set them.
        pass
    elif family == "phi":
        mid = (
            (detected.get("model_type") or "")
            + " "
            + " ".join(str(a) for a in (detected.get("architectures") or []))
        ).lower()
        # Prefer model id when present on the empty config.
        mid += " " + str(base.get("model") or "").lower()
        base["trust_remote_code"] = True
        if "mini" in mid and "instruct" in mid:
            if not base.get("tool_call_parser"):
                base["tool_call_parser"] = "phi4_mini_json"
                base["enable_auto_tool_choice"] = True
                rationale.append("Phi-4-mini-instruct → tool-call-parser phi4_mini_json")
        elif "reasoning" in mid and "mini" not in mid:
            if not base.get("reasoning_parser"):
                base["reasoning_parser"] = "deepseek_r1"
                rationale.append("Phi-4-reasoning → --reasoning-parser deepseek_r1")
        # Strip obsolete --enable-reasoning if a card left it in extras.
        if base.get("extra_flags") and "--enable-reasoning" in base["extra_flags"]:
            base["extra_flags"] = _strip_flag_from_extra(base["extra_flags"], "--enable-reasoning")
            rationale.append("Stripped obsolete --enable-reasoning (removed in vLLM ≥0.10)")
    elif family == "granite":
        mid = (
            (detected.get("model_type") or "")
            + " "
            + " ".join(str(a) for a in (detected.get("architectures") or []))
            + " "
            + str(base.get("model") or "")
        ).lower()
        if "granite-4" in mid or "granite4" in mid or "granite_4" in mid:
            if not base.get("tool_call_parser"):
                base["tool_call_parser"] = "granite4"
                base["enable_auto_tool_choice"] = True
                rationale.append("Granite 4 → tool-call-parser granite4")
        else:
            if not base.get("tool_call_parser"):
                base["tool_call_parser"] = "granite"
                base["enable_auto_tool_choice"] = True
                rationale.append("Granite 3.x → tool-call-parser granite")
            # Reasoning parser only for 3.2 prose markers — not 3.3 <think>.
            if "3.2" in mid or "granite-3.2" in mid:
                if not base.get("reasoning_parser"):
                    base["reasoning_parser"] = "granite"
                    rationale.append("Granite 3.2 → --reasoning-parser granite")
    elif family == "internlm":
        if not base.get("tool_call_parser"):
            base["tool_call_parser"] = "internlm"
            base["enable_auto_tool_choice"] = True
            rationale.append("InternLM → tool-call-parser internlm + auto tool choice")
        base["trust_remote_code"] = True
    elif family == "hunyuan":
        if not base.get("reasoning_parser"):
            base["reasoning_parser"] = "hunyuan_a13b"
            rationale.append("Hunyuan A13B → --reasoning-parser hunyuan_a13b")
        if not base.get("tool_call_parser"):
            base["tool_call_parser"] = "hunyuan_a13b"
            base["enable_auto_tool_choice"] = True
        base["trust_remote_code"] = True
    elif family == "hy_v3":
        if not base.get("reasoning_parser"):
            base["reasoning_parser"] = "hy_v3"
            rationale.append("HY-v3 / Hy3 → --reasoning-parser hy_v3")
        if not base.get("tool_call_parser"):
            base["tool_call_parser"] = "hy_v3"
            base["enable_auto_tool_choice"] = True
        base["trust_remote_code"] = True
    elif family == "step3p5":
        if not base.get("reasoning_parser"):
            base["reasoning_parser"] = "step3p5"
            rationale.append("Step-3.5 → --reasoning-parser step3p5")
        if not base.get("tool_call_parser"):
            base["tool_call_parser"] = "step3p5"
            base["enable_auto_tool_choice"] = True
        base["trust_remote_code"] = True
    elif family == "step3":
        if not base.get("reasoning_parser"):
            base["reasoning_parser"] = "step3"
            rationale.append("Step-3 → --reasoning-parser step3")
        if not base.get("tool_call_parser"):
            base["tool_call_parser"] = "step3"
            base["enable_auto_tool_choice"] = True
        base["trust_remote_code"] = True
    elif family == "ernie":
        mid = (
            (detected.get("model_type") or "")
            + " "
            + " ".join(str(a) for a in (detected.get("architectures") or []))
            + " "
            + str(base.get("model") or "")
        ).lower()
        # Thinking variants use ernie45; PT-only omit parsers unless the card set them.
        if "thinking" in mid:
            if not base.get("reasoning_parser"):
                base["reasoning_parser"] = "ernie45"
                rationale.append("ERNIE-4.5 Thinking → --reasoning-parser ernie45")
            if not base.get("tool_call_parser"):
                base["tool_call_parser"] = "ernie45"
                base["enable_auto_tool_choice"] = True
        base["trust_remote_code"] = True


_CONTEXT_LADDER = (1048576, 524288, 262144, 131072, 65536, 32768, 16384)
_UMA_RESERVE_GIB = 15.0
_RUNTIME_PAD_GIB = 4.0
_UTIL_HARD_CAP = 0.90
_UTIL_SLACK = 1.08


def recommended_gpu_util(
    node_ram_gib: float,
    weights_gib: float,
    kv_gib: float,
    *,
    reserve_gib: float = _UMA_RESERVE_GIB,
) -> float:
    """Util that holds weights + this KV window and still leaves reserved UMA.

    Not a leftover 0.85 default: sized to the chosen (max-model-len, max-num-seqs).
    """
    ram = float(node_ram_gib or 0.0)
    if ram <= 0:
        return WORKFLOW_UTIL
    need = (
        max(0.0, float(weights_gib or 0.0))
        + max(0.0, float(kv_gib or 0.0))
        + _RUNTIME_PAD_GIB
    ) * _UTIL_SLACK
    raw = need / ram
    cap = min(_UTIL_HARD_CAP, max(0.50, 1.0 - float(reserve_gib) / ram))
    floor = min(cap, (max(0.0, float(weights_gib or 0.0)) + float(reserve_gib)) / ram)
    return round(min(cap, max(floor, raw, 0.50)), 2)


def _kv_bytes_per_token(
    hf_config: Optional[dict],
    *,
    kv_cache_dtype: str,
    family: str,
) -> float:
    """Conservative bytes/token for residual KV budget (GQA / MLA / mamba-hybrid)."""
    cfg = hf_config or {}
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}

    def g(k: str, default: Any = None) -> Any:
        return cfg.get(k, text.get(k, default))

    dtype = (kv_cache_dtype or "fp8").lower()
    if "nvfp4" in dtype:
        b = 0.5
    elif "fp8" in dtype:
        b = 1.0
    else:
        b = 2.0

    # MLA / DeepSeek latent path (KV is typically replicated across TP ranks).
    if "nvfp4_ds_mla" in dtype or "fp8_ds_mla" in dtype or (family or "").startswith("deepseek"):
        layers = int(g("num_hidden_layers") or 60)
        per_layer = 320.0 if "nvfp4" in dtype else 584.0
        return float(layers * per_layer)

    layers = int(g("num_hidden_layers") or 32)
    n_kv = int(g("num_key_value_heads") or g("num_attention_heads") or 8)
    n_heads = int(g("num_attention_heads") or n_kv)
    hidden = int(g("hidden_size") or 4096)
    head_dim = int(g("head_dim") or (hidden // max(n_heads, 1)))

    # Nemotron hybrid: only attention layers grow with context length.
    block_types = g("layers_block_type") or g("layer_types") or []
    if isinstance(block_types, list) and block_types:
        n_attn = sum(
            1
            for t in block_types
            if "attention" in str(t).lower() or str(t).lower() in ("attn", "a")
        )
        if n_attn > 0:
            layers = n_attn

    return float(layers * 2 * n_kv * head_dim * b)


def _size_memory_for_spark(
    cfg: dict[str, Any],
    *,
    hf_config: Optional[dict],
    detected: dict[str, Any],
    weights_gib: Optional[float],
    node_ram_gib: float,
    mode: str,
    rationale: list[str],
    warnings: list[str],
) -> None:
    """Pick the best max-model-len (native first) and the util that holds it.

    Prefer keeping the target context and dropping max-num-seqs before cutting
    length. Then set util from weights + that KV window + reserved UMA unless
    a recipe already pinned util. Runs for every model, including TP>=2
    (``weights_gib`` must be per-node).
    """
    del mode
    per_node_w = float(weights_gib) if weights_gib and weights_gib > 0 else 0.0
    ml = cfg.get("max_model_len")
    if not isinstance(ml, int) or ml <= 0:
        cfg["util"] = recommended_gpu_util(node_ram_gib, per_node_w, 0.0)
        rationale.append(f"Recommended util={cfg['util']} (no max-model-len to size)")
        return
    seqs = int(cfg.get("max_num_seqs") or 4)
    # Always size against reserved-UMA capacity so a low overlay/card util
    # cannot shrink native/recipe context. Util is rewritten to match the window.
    budget = float(node_ram_gib) - _UMA_RESERVE_GIB - per_node_w - _RUNTIME_PAD_GIB
    if budget < 8.0:
        cfg["util"] = recommended_gpu_util(node_ram_gib, per_node_w, 0.0)
        rationale.append(
            f"Recommended util={cfg['util']} (weights≈{per_node_w:.1f} GiB; "
            "KV budget too tight to size context)"
        )
        return

    bpt = _kv_bytes_per_token(
        hf_config,
        kv_cache_dtype=str(cfg.get("kv_cache_dtype") or "fp8"),
        family=str((detected or {}).get("family") or ""),
    )
    if bpt <= 0:
        cfg["util"] = recommended_gpu_util(node_ram_gib, per_node_w, 0.0)
        rationale.append(f"Recommended util={cfg['util']} (no KV shape; weights + reserve)")
        return

    def need_gib(length: int, nseq: int) -> float:
        return (bpt * length * nseq / (1024**3)) * 1.10

    lengths = [ml] + [n for n in _CONTEXT_LADDER if n < ml]
    seq_opts = []
    for nseq in (seqs, 4, 2, 1):
        if nseq > 0 and nseq <= seqs and nseq not in seq_opts:
            seq_opts.append(nseq)

    chosen: Optional[int] = None
    chosen_seqs = seqs
    for length in lengths:
        for nseq in seq_opts:
            if need_gib(length, nseq) <= budget:
                chosen = length
                chosen_seqs = nseq
                break
        if chosen is not None:
            break

    if chosen is None:
        chosen = 16384
        chosen_seqs = 1
        warnings.append(
            f"MEMORY: even 16k×1 seq may be tight (budget≈{budget:.0f} GiB after weights); "
            "consider more nodes or a smaller checkpoint."
        )

    if chosen != ml or chosen_seqs != seqs:
        rationale.append(
            f"MEMORY: keep longest fit → max-model-len {ml} → {chosen}"
            + (f", max-num-seqs {seqs} → {chosen_seqs}" if chosen_seqs != seqs else "")
            + f" (kv≈{need_gib(chosen, chosen_seqs):.1f} GiB ≤ budget {budget:.1f} GiB, "
            f"weights≈{per_node_w:.1f} GiB)"
        )
        cfg["max_model_len"] = chosen
        if chosen_seqs != seqs:
            cfg["max_num_seqs"] = chosen_seqs

    kv_used = need_gib(int(cfg["max_model_len"]), int(cfg.get("max_num_seqs") or chosen_seqs))
    cfg["util"] = recommended_gpu_util(node_ram_gib, per_node_w, kv_used)
    rationale.append(
        f"Recommended util={cfg['util']} for max-model-len={cfg['max_model_len']} "
        f"× {int(cfg.get('max_num_seqs') or chosen_seqs)} seq "
        f"(kv≈{kv_used:.1f} GiB + weights≈{per_node_w:.1f} GiB on {node_ram_gib:.0f} GiB)"
    )


def _apply_hardware_envelope(
    cfg: dict[str, Any],
    rationale: list[str],
    card_set_max_len: bool,
    card_set_util: bool = False,
    *,
    mode: str | None = None,
    native_max_len: int | None = None,
) -> None:
    """Fill max-len / image when the recipe is silent. Util is sized later.

    Target context is the checkpoint's native window when known — not a
    one-size 262k default. ``mode`` is ignored (legacy).
    """
    del mode, card_set_util
    tp = int(cfg.get("tensor_parallel_size") or 1)
    if not card_set_max_len:
        if isinstance(native_max_len, int) and native_max_len > 0:
            cfg["max_model_len"] = native_max_len
            rationale.append(
                f"Native max_position_embeddings → max-model-len={native_max_len}"
            )
        else:
            cfg["max_model_len"] = WORKFLOW_MAX_LEN
            rationale.append(
                f"Hardware envelope → max-model-len={WORKFLOW_MAX_LEN} (card silent, no native window)"
            )
    if not cfg.get("image"):
        cfg["image"] = DEFAULT_IMAGE_MAX

    # Single-node: card/demo 1M contexts OOMs under realistic util. Multi-node
    # overlays (TP>=2) that intentionally pin a huge window keep it as the *target*;
    # the sizer still drops seqs / length if KV cannot fit per node.
    ceiling = WORKFLOW_MAX_LEN
    ml = cfg.get("max_model_len")
    if isinstance(ml, int) and ml > ceiling and tp < 2:
        cfg["max_model_len"] = ceiling
        rationale.append(
            f"clamped max-model-len {ml} → {ceiling} "
            "(single-node Spark; raise TP or edit form for longer ctx)"
        )


def _apply_mode_envelope(
    cfg: dict[str, Any],
    mode: str | None,
    rationale: list[str],
    card_set_max_len: bool,
    card_set_util: bool = False,
) -> None:
    """Legacy wrapper — ``mode`` is ignored."""
    _apply_hardware_envelope(
        cfg, rationale, card_set_max_len, card_set_util=card_set_util, mode=mode
    )


# ─── Public API ───────────────────────────────────────────────────────────────


def recommend(
    model: str,
    *,
    mode: str | None = None,
    fetch_remote: bool = True,
    backend: str | None = None,
) -> dict[str, Any]:
    """Build serve config from the HF card plus researched vendor recipes.

    ``mode`` is accepted for old clients and ignored — util / max-len / VL
    come from the researched recipe + live hardware envelope.

    ``backend`` is ``vllm`` (default), ``llamacpp``, or ``sglang``. GGUF ids
    should use ``llamacpp``. None of these paths start a server.
    """
    mode = None  # user-facing dual envelopes are gone
    model = (model or "").strip()
    if not model:
        raise ValueError("model is required")

    eng = (backend or "vllm").strip().lower()
    if eng in ("llamacpp", "llama.cpp", "gguf"):
        return recommend_llamacpp(model, fetch_remote=fetch_remote)
    if eng in ("sglang", "sg-lang", "sgl"):
        return recommend_sglang(model, fetch_remote=fetch_remote)

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

    # Model-family overlay (e.g. DeepSeek DSpark) + live cluster topology. The overlay
    # supplies the correct serve path when the card's generic recipe is wrong; topology
    # decides TP/nnodes/fabric from how many Sparks are actually online.
    overlay = _family_overlay(model, detected)
    topology = _cluster_topology()
    # Real weight size (exact HF blob sum → config estimate) drives the placement engine.
    weights_gib = estimate_weights_gib(model, hf_config)

    # ── Parse card for best vllm serve (scored with config.json knowledge) ──
    # Weak cards (Unsloth) often only link a vendor guide — follow those docs.
    # When the card is silent we still *derive* Unsloth / NVIDIA / vLLM / GitHub
    # URLs from the model id and family. Overlay still owns flags when it matches
    # (MiniMax/DSpark), but we still fetch so sources name every class used.
    candidates: list[ServeCandidate] = []
    vendor_bodies: list[str] = []
    if readme:
        candidates = extract_serve_candidates(readme, detected=detected)
        want_fam = str(detected.get("family") or "")
        candidates = [
            c
            for c in candidates
            if not (
                (want_fam and want_fam != "unknown" and _vendor_candidate_family_mismatch(c, want_fam))
                or _vendor_candidate_sibling_mismatch(c, model)
            )
        ]
    candidates = _augment_candidates_from_cookbooks(
        readme or "",
        candidates=candidates,
        detected=detected,
        sources=sources,
        rationale=rationale,
        warnings=warnings,
        model_id=model,
        vendor_bodies=vendor_bodies,
    )
    if readme or candidates:
        # Parsers-only / last-mile overlays must still apply official extras
        # (Gemma 4 31B --chat-template / --async-scheduling / --limit-mm-per-prompt).
        if overlay and not _overlay_gap_only(overlay):
            if candidates:
                rationale.append(
                    f"Family overlay owns serve flags; also found {len(candidates)} "
                    "card/vendor serve snippet(s) as supporting source"
                )
        elif candidates:
            best = candidates[0]
            if candidates_recipe_poor(candidates):
                rationale.append(
                    f"Skipped weak card recipe (score {best.score:.0f})"
                    + (f" in «{best.section}»" if best.section else "")
                    + " — native config.json defaults"
                )
                for r in best.reasons:
                    rationale.append(f"  · {r}")
                sources.insert(
                    0,
                    {
                        "kind": "hf_card_recipe_skipped",
                        "ref": card_url or "README.md",
                        "notes": f"score={best.score:.0f}; section={best.section or 'n/a'}",
                    },
                )
            elif _vendor_candidate_sibling_mismatch(best, model):
                rationale.append(
                    "Skipped vendor recipe naming a different model than the paste"
                )
            else:
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
                # Winning recipe may omit tools; harvest from tool-focused alts (e.g. Nano nemotron_json).
                _harvest_tool_flags_from_candidates(cfg, candidates, rationale)
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

        if readme:
            prose = _card_prose_hints(readme)
            _drop_family_mismatched_hints(prose, detected)
            for n in prose.pop("notes", []):
                rationale.append(n)
            filled = _merge_fill(cfg, prose)
            if filled:
                rationale.append(f"Card prose filled: {', '.join(filled)}")

    for body in vendor_bodies:
        prose = _card_prose_hints(body)
        _drop_family_mismatched_hints(prose, detected)
        for n in prose.pop("notes", []):
            rationale.append(n)
        filled = _merge_fill(cfg, prose)
        if filled:
            rationale.append(f"Vendor-doc prose filled: {', '.join(filled)}")

    card_set_max_len = cfg.get("max_model_len") is not None

    # Gaps only: HF config.json / tags (still from the model on the hub)
    _fill_from_config_detection(cfg, detected, rationale)

    # Image pin (docker recipe + card prose + capability) *before* marlin/safety so
    # gates see the intended stock tag. Overlay Anemll applied below still wins.
    cand_image = None
    if candidates:
        cand_image = (candidates[0].config or {}).get("image") or None
    card_image_pin = _parse_card_image_requirement(readme)
    _resolve_image_for_gates(
        cfg,
        mode=mode,
        candidate_image=cand_image,
        card_image=card_image_pin,
        detected=detected,
        rationale=rationale,
        warnings=warnings,
    )
    _warn_floating_card_image(readme, _lab_default_image(mode), warnings)

    # config.json is ground truth for quant layout — fix card flags that crash
    _apply_checkpoint_safety(cfg, detected, warnings, rationale)

    # Model-family overlay wins over the card when the card's recipe is wrong for this
    # checkpoint's Spark path (DeepSeek DSpark: custom image + b12x + nvfp4_ds_mla).
    if overlay:
        overlay_cfg = overlay["config"]
        # Qwen3.8 NVFP4 / Gemma 4 overlays are last-mile fill. MiniMax / DSpark
        # still own flags because the card recipe is known-wrong for those Spark paths.
        gap_only = _overlay_gap_only(overlay)
        for k, v in overlay_cfg.items():
            if k == "docker_env":
                cfg["docker_env"] = _dedupe_env(list(cfg.get("docker_env") or []) + list(v))
            elif k == "extra_flags":
                # Fill/card may already own tokenizer/config-format; never emit twice.
                cfg["extra_flags"] = _merge_extra_flags(
                    cfg.get("extra_flags") or "", v or ""
                )
            elif (
                gap_only
                and k
                in (
                    "quantization",
                    "kv_cache_dtype",
                    "reasoning_parser",
                    "tool_call_parser",
                    "enable_auto_tool_choice",
                    "trust_remote_code",
                    "moe_backend",
                    "load_format",
                )
                and cfg.get(k) not in (None, "", False)
            ):
                continue
            else:
                cfg[k] = v
        for r in overlay["rationale"]:
            rationale.append(r)
        sources.insert(
            0,
            {"kind": "family_overlay", "ref": overlay["source"], "notes": overlay["label"]},
        )
        confidence = "high"

    # Topology: TP / nnodes / fabric env from the placement plan (replaces the old TP=1 strip).
    _apply_topology(
        cfg,
        overlay=overlay,
        topology=topology,
        weights_gib=weights_gib,
        mode=mode,
        warnings=warnings,
        rationale=rationale,
    )

    # First-boot lab posture: no MTP, no crashing moe backends (Auto-configure → Start).
    # Skip when a family overlay deliberately enables MTP/exotic backends (DSpark needs them).
    if not overlay:
        _apply_first_boot_defaults(
            cfg,
            mode=mode,
            detected=detected,
            warnings=warnings,
            rationale=rationale,
        )
    _apply_vl_spark_defaults(cfg, detected, warnings, rationale, mode=mode)

    # Always explain flashinfer avoidance when card recommends it but checkpoint forbids it.
    # Overlay already encodes the correct backend, so only run for card-driven configs.
    if not overlay:
        _note_card_flashinfer_avoidance(
            candidates=candidates,
            readme=readme,
            detected=detected,
            cfg=cfg,
            warnings=warnings,
            rationale=rationale,
        )

    # Hardware envelope for util / defaults when card silent. util/max-len set by the card OR a
    # family overlay are authoritative — the envelope only fills gaps, never overrides them.
    card_set_util = cfg.get("util") is not None
    card_set_max_len = cfg.get("max_model_len") is not None
    native_max = detected.get("max_position_embeddings")
    if not isinstance(native_max, int) or native_max <= 0:
        native_max = None
    _apply_hardware_envelope(
        cfg,
        rationale,
        card_set_max_len=card_set_max_len,
        card_set_util=card_set_util,
        native_max_len=native_max,
    )

    # Final image pass after overlay/mode envelope: re-apply capability floor on the
    # fully filled cfg. Anemll overlay image is preserved (never replaced).
    _resolve_image_for_gates(
        cfg,
        mode=mode,
        candidate_image=cand_image if not overlay else None,
        card_image=card_image_pin if not overlay else None,
        detected=detected,
        rationale=rationale,
        warnings=warnings,
    )

    # Resolve card exports ($DSPARK_CKPT → real HF id) before scrubbing leftovers.
    if cfg.get("extra_flags"):
        cfg["extra_flags"] = _expand_card_exports(cfg["extra_flags"], readme)
        cfg["extra_flags"] = _scrub_unexpanded_shell_vars(cfg["extra_flags"], warnings)

    # DSpark speculative without a draft model path will fail at load — fill or strip.
    _ensure_dspark_draft_or_strip(cfg, readme, warnings, rationale)

    # Card multi-GPU / mega-MoE knobs crash or mis-route on 1–2× GB10.
    _strip_spark_unsafe_flags(cfg, warnings, rationale)

    max_pos = detected.get("max_position_embeddings")
    if isinstance(max_pos, int) and max_pos > 0 and cfg.get("max_model_len"):
        if cfg["max_model_len"] > max_pos:
            cfg["max_model_len"] = max_pos
            rationale.append(f"Capped max-model-len to config max_position_embeddings={max_pos}")

    plan = cfg.get("topology_plan") or {}
    node_ram = _resolved_node_ram_gib(plan.get("node_ram_gib"))
    per_node = plan.get("per_node_weights_gib")
    if per_node is None and weights_gib and weights_gib > 0:
        n_used = max(1, int(plan.get("nodes_needed") or cfg.get("tensor_parallel_size") or 1))
        per_node = float(weights_gib) / n_used
    _size_memory_for_spark(
        cfg,
        hf_config=hf_config,
        detected=detected,
        weights_gib=per_node if per_node is not None else weights_gib,
        node_ram_gib=node_ram,
        mode=mode,
        rationale=rationale,
        warnings=warnings,
    )


    # Normalize env (dedupe keys; last wins)
    env_out: list[str] = []
    seen: set[str] = set()
    for e in cfg.get("docker_env") or []:
        if not e or "=" not in e:
            continue
        k = e.split("=", 1)[0].strip()
        v = e.split("=", 1)[1].strip()
        if not k:
            continue
        if k in seen:
            env_out = [x for x in env_out if not x.startswith(k + "=")]
        seen.add(k)
        env_out.append(f"{k}={v}")
    cfg["docker_env"] = _scrub_unexpanded_docker_env(env_out, warnings)

    # If MTP structured flag is on, strip duplicate --speculative-config from extras
    # after harvesting Spark MoE keys the structured emit must carry.
    if cfg.get("mtp") and cfg.get("extra_flags"):
        spec_extra = _mtp_spark_extra(_speculative_json_from_extra(cfg["extra_flags"]))
        if spec_extra.get("moe_backend") and not (cfg.get("mtp_moe_backend") or "").strip():
            cfg["mtp_moe_backend"] = str(spec_extra["moe_backend"])
        cfg["extra_flags"] = _strip_flag_from_extra(cfg["extra_flags"], "--speculative-config")

    # GB10 / Spark: Qwen FP8 dense has hit DeepGEMM issues — soft env when card silent
    if (
        detected.get("has_fp8")
        and not detected.get("is_mixed_nvfp4_fp8")
        and (cfg.get("quantization") or "").lower() == "fp8"
        and not any(e.startswith("VLLM_USE_DEEP_GEMM=") for e in cfg["docker_env"])
    ):
        cfg["docker_env"].append("VLLM_USE_DEEP_GEMM=0")
        rationale.append(
            "FP8 checkpoint on Spark → VLLM_USE_DEEP_GEMM=0 (avoids known GB10 DeepGEMM garbage)"
        )
        warnings.append(
            "Added VLLM_USE_DEEP_GEMM=0 for FP8 on GB10. Remove if your vLLM build is fine without it."
        )

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
    if overlay:
        label = overlay["label"]
    elif candidates:
        label = f"HF card: {candidates[0].section or 'vLLM recipe'}"
    elif from_website:
        label = "HF config (no serve recipe on card)"
    else:
        label = "Offline / incomplete"

    notes = None
    if overlay:
        notes = (
            f"{overlay['label']} — family overlay overrides the generic HF card recipe "
            f"({overlay['source']}). Topology: {topology.get('nodes', 1)} Spark(s) online."
        )
    elif from_website and candidates:
        notes = (
            f"Derived from the live Hugging Face card ({card_url}). "
            f"Picked the highest-scoring of {len(candidates)} vllm serve recipe(s) on the card."
        )
    elif from_website:
        notes = f"Fetched {card_url} but found no vllm serve blocks; used config.json + tags."
    else:
        notes = "Live HF card was not available; result may be incomplete."

    plan = cfg.get("topology_plan") or {}
    fits = bool(plan.get("fits", True))
    ok_load, load_msg = check_serve_loadability(
        mode=mode,
        weights_gib=weights_gib,
        node_ram_gib=_resolved_node_ram_gib(plan.get("node_ram_gib")),
        nodes_used=int(plan.get("nodes_needed") or 1),
        util=float(cfg.get("util") or WORKFLOW_UTIL),
        reserve_gib=float(plan.get("reserve_gib") or 15.0),
    )
    if fits and not ok_load:
        fits = False
        if load_msg:
            warnings.append(load_msg)
    serve_blocked = not fits
    if serve_blocked:
        confidence = "low"
        warnings.append(
            "SERVE BLOCKED: weights do not fit the online cluster — Start will refuse until "
            "you add nodes or pick a smaller checkpoint."
        )
    topology_out = {
        "nodes": topology.get("nodes", 1),
        "nodes_used": plan.get("nodes_needed", 1),
        "fabric_ok": topology.get("fabric_ok", False),
        "head_ip": (topology.get("head") or {}).get("qsfp_ip"),
        "worker_ips": [w.get("qsfp_ip") for w in (plan.get("workers") or []) if w.get("qsfp_ip")],
        "tensor_parallel_size": cfg.get("tensor_parallel_size") or 1,
        "pipeline_parallel_size": plan.get("pipeline_parallel_size", 1),
        "weights_gib": weights_gib,
        "per_node_weights_gib": plan.get("per_node_weights_gib"),
        "util_computed": plan.get("util_computed"),
        "fits": fits,
        "node_ram_gib": plan.get("node_ram_gib"),
        "overlay": overlay["family_key"] if overlay else None,
    }

    token_ok = False
    try:
        token_ok = bool(_hf_token()) and hf_token_usable()
    except Exception:
        token_ok = False
    if _hf_token() and not token_ok:
        warnings.append(
            "Stored HF token failed Hub auth (401). Fetches use anonymous access for public "
            "models. Re-login with `hf auth login` for gated models / higher rate limits."
        )

    return {
        "model": model,
        "engine": "vllm",
        "mode": "auto",
        "confidence": confidence,
        "label": label,
        "notes": notes,
        "card_url": card_url,
        "from_website": from_website,
        "hf_token_ok": token_ok,
        "serve_blocked": serve_blocked,
        "detected": detected,
        "config": cfg,
        "topology": topology_out,
        "sources": sources,
        "rationale": rationale,
        "warnings": warnings,
        "card_recipes": [
            {
                "score": c.score,
                "section": c.section,
                "raw": c.raw[:400],
                "selected": i == 0 and not _vendor_candidate_sibling_mismatch(c, model),
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


def looks_like_gguf(model: str, tags: list[str] | None = None) -> bool:
    """True when the id, path, or Hub tags point at a GGUF / llama.cpp checkpoint."""
    mid = (model or "").lower()
    if mid.endswith(".gguf") or "/gguf" in mid or "gguf" in mid.split("/")[-1]:
        return True
    for t in tags or []:
        if "gguf" in str(t).lower():
            return True
    return False


def recommend_llamacpp(model: str, *, fetch_remote: bool = True) -> dict[str, Any]:
    """Spark-optimal llama.cpp flags. Recommend-only — does not start llama-server.

    Pack from ggml DGX Spark benches + NVIDIA forum Spark guides:
    ``-ngl 99 --flash-attn on --no-mmap --jinja -ub 2048`` and a ctx sized
    to leftover UMA after weights + reserved 15 GiB.
    """
    model = (model or "").strip()
    if not model:
        raise ValueError("model is required")

    rationale: list[str] = []
    warnings: list[str] = []
    hf_config: Optional[dict] = None
    tags: list[str] = []
    if (
        fetch_remote
        and not Path(model).is_dir()
        and not model.startswith("/")
        and "/" in model
    ):
        remote = fetch_hf_card(model)
        hf_config = remote.get("config")
        api = remote.get("api") or {}
        tags = list(api.get("tags") or [])
        for u in remote.get("fetched") or []:
            rationale.append(f"HF fetch: {u}")

    hub_id, want_quant = _hf_repo_ref(model)
    blob = _hub_blob_measure(model)
    if blob and blob.get("gib") is not None:
        weights_gib = float(blob["gib"])
    else:
        weights_gib = estimate_weights_gib(hub_id, hf_config)
    spark_quant = (want_quant or (blob or {}).get("quant") or "UD-Q4_K_XL") if (
        "/" in hub_id and not hub_id.startswith("/") and not Path(hub_id).is_dir()
    ) else None
    use_mtp = bool((blob or {}).get("mtp")) or _gguf_siblings_have_mtp(
        hub_id, (blob or {}).get("siblings") or []
    )
    topology = _cluster_topology()
    plan = plan_placement(weights_gib, topology)
    ram = _resolved_node_ram_gib(plan.get("node_ram_gib"))
    w = float(weights_gib or 0.0)
    leftover = ram - w - _UMA_RESERVE_GIB
    ctx = 8192
    for cand in _CONTEXT_LADDER:
        # Unknown GQA/arch: budget ~0.25 GiB KV per 8k tokens (q8 cache, conservative).
        need = (float(cand) / 8192.0) * 0.25
        if leftover >= need + _RUNTIME_PAD_GIB:
            ctx = int(cand)
            break
    if leftover < 8.0:
        warnings.append(
            "Weights leave little UMA headroom — shrink --ctx-size if llama-server OOMs"
        )

    mtp_n = 3
    cfg = {
        "engine": "llamacpp",
        "model": hub_id,
        "ngl": 99,
        "flash_attn": "on",
        "no_mmap": True,
        "jinja": True,
        "ubatch": 2048,
        "ctx_size": ctx,
        "host": "0.0.0.0",
        "port": 8080,
        "hf_quant": spark_quant,
        "mtp": use_mtp,
        "mtp_num_tokens": mtp_n if use_mtp else 0,
        "extra_flags": "",
    }
    if spark_quant and "/" in hub_id and not hub_id.startswith("/"):
        model_arg = f"-hf {hub_id}:{spark_quant}"
        rationale.append(f"llama.cpp Hub ref: -hf {hub_id}:{spark_quant} (one Spark quant, not -m repo)")
    else:
        model_arg = f"-m {hub_id}"
    mtp_arg = ""
    if use_mtp:
        mtp_arg = f" --spec-type draft-mtp --spec-draft-n-max {mtp_n}"
        rationale.append(
            f"MTP GGUF detected → --spec-type draft-mtp --spec-draft-n-max {mtp_n}"
        )
    argv = (
        f"llama-server {model_arg} -ngl 99 --flash-attn on --no-mmap --jinja "
        f"-ub 2048 -c {ctx} --host 0.0.0.0 --port 8080{mtp_arg}"
    )
    rationale.append(
        "DGX Spark llama.cpp pack: -ngl 99, --flash-attn on, --no-mmap "
        "(GB10 mmap is slow), --jinja (tools). "
        "Refs: ggml-org/llama.cpp#16578, NVIDIA Spark Qwen3.5 llama.cpp guide."
    )
    rationale.append(f"ctx_size={ctx} from leftover UMA ({leftover:.1f} GiB after weights+reserve)")
    fits = bool(plan.get("fits", True))
    if weights_gib is None:
        fits = True
        warnings.append("Weight size unknown — ctx is conservative; verify the GGUF fits UMA")
    return {
        "model": hub_id,
        "engine": "llamacpp",
        "mode": "auto",
        "confidence": "medium" if weights_gib else "low",
        "label": "llama.cpp Spark",
        "notes": "Recommend-only. Start llama-server yourself (L.A.I.L does not spawn it yet).",
        "serve_blocked": not fits,
        "config": cfg,
        "argv": argv,
        "topology": {
            "nodes": topology.get("nodes", 1),
            "weights_gib": weights_gib,
            "node_ram_gib": ram,
            "fits": fits,
            "gguf_quant": spark_quant,
        },
        "sources": [
            {
                "kind": "vendor_doc",
                "ref": "https://github.com/ggml-org/llama.cpp/discussions/16578",
                "notes": "DGX Spark llama.cpp benches",
            }
        ],
        "rationale": rationale,
        "warnings": warnings,
        "detected": {"family": "gguf" if looks_like_gguf(hub_id, tags) else "unknown"},
    }


def recommend_sglang(model: str, *, fetch_remote: bool = True) -> dict[str, Any]:
    """Spark-optimal SGLang flags from the card / Unsloth docs. Recommend-only.

    Unsloth NVFP4 pages publish ``python -m sglang.launch_server`` with NEXTN
    speculative decoding. Hardware envelope sets ``--tp-size`` and
    ``--mem-fraction-static`` from live UMA when the vendor line is silent.
    """
    model = (model or "").strip()
    if not model:
        raise ValueError("model is required")

    rationale: list[str] = []
    warnings: list[str] = []
    sources: list[dict[str, str]] = []
    readme: Optional[str] = None
    hf_config: Optional[dict] = None
    if (
        fetch_remote
        and not Path(model).is_dir()
        and not model.startswith("/")
        and "/" in model
    ):
        remote = fetch_hf_card(model)
        readme = remote.get("readme")
        hf_config = remote.get("config")
        for u in remote.get("fetched") or []:
            sources.append({"kind": "huggingface", "ref": u, "notes": "HF card/config"})
        for e in remote.get("errors") or []:
            rationale.append(f"HF fetch note: {e}")
    if readme is None or hf_config is None:
        local = load_local_fallback(model)
        if readme is None and local.get("readme"):
            readme = local["readme"]
            sources.append({"kind": "hf_card_local_fallback", "ref": model, "notes": "offline README"})
        if hf_config is None and local.get("config"):
            hf_config = local["config"]

    detected = analyze_config(hf_config or {}, model)
    weights_gib = estimate_weights_gib(model, hf_config)
    topology = _cluster_topology()
    plan = plan_placement(weights_gib, topology)
    ram = _resolved_node_ram_gib(plan.get("node_ram_gib"))
    tp = max(1, int(plan.get("tensor_parallel_size") or 1))
    w = float(weights_gib or 0.0)
    per = float(plan.get("per_node_weights_gib") or w or 0.0)
    mem = recommended_gpu_util(ram, per, 0.0)

    candidates: list[ServeCandidate] = []
    if readme:
        candidates = extract_sglang_candidates(readme)
    # Same derived Unsloth / vLLM / GitHub URLs as vLLM — those pages ship SGLang too.
    if not candidates or candidates_recipe_poor(candidates):
        for rec in discover_recipe_urls(model, detected, readme or "")[:COOKBOOK_MAX_URLS]:
            text, err = fetch_cookbook_text(rec.url)
            if err or not text:
                continue
            extra = extract_sglang_candidates(text)
            if extra:
                sources.append(
                    {"kind": rec.kind, "ref": rec.url, "notes": f"SGLang from {rec.kind}"}
                )
                candidates.extend(extra)
    want_fam = str(detected.get("family") or "")
    filtered: list[ServeCandidate] = []
    for c in candidates:
        if want_fam and want_fam != "unknown" and _vendor_candidate_family_mismatch(c, want_fam):
            continue
        if _vendor_candidate_sibling_mismatch(c, model):
            continue
        filtered.append(c)
    filtered.sort(key=lambda c: c.score, reverse=True)
    candidates = filtered

    cfg: dict[str, Any] = {
        "engine": "sglang",
        "model": model,
        "model_path": model,
        "tp_size": tp,
        "mem_fraction_static": mem,
        "context_length": None,
        "trust_remote_code": True,
        "host": "0.0.0.0",
        "port": 30000,
        "quantization": "",
        "tool_call_parser": "",
        "reasoning_parser": "",
        "spec_algorithm": "",
        "spec_num_steps": None,
        "spec_eagle_topk": None,
        "spec_draft_tokens": None,
        "extra_flags": "",
    }
    if candidates and not _vendor_candidate_sibling_mismatch(candidates[0], model):
        best = candidates[0]
        for k, v in (best.config or {}).items():
            if k in ("engine", "model"):
                continue
            if k == "extra_flags":
                cfg["extra_flags"] = v or ""
            elif k == "model_path" and v:
                cfg["model_path"] = v
            elif v not in (None, "", False):
                cfg[k] = v
        rationale.append(
            f"Best SGLang recipe (score {best.score:.0f}"
            + (f" in «{best.section}»" if best.section else "")
            + f"): {(best.raw or '')[:160].replace(chr(10), ' ')}"
        )
        for r in best.reasons:
            rationale.append(f"  · {r}")
    else:
        rationale.append(
            "No parseable sglang.launch_server on card/vendor docs — Spark defaults "
            "(tp/mem from live UMA)"
        )

    # Always launch the id the operator pasted (vendor snippets name twins).
    cfg["model_path"] = model
    cfg["model"] = model
    # Placement owns TP on this cluster — never keep a vendor --tp-size 8 on 1 Spark.
    vendor_tp = cfg.get("tp_size")
    if not vendor_tp or int(vendor_tp) > tp:
        if vendor_tp and int(vendor_tp) > tp:
            rationale.append(
                f"Vendor --tp-size {vendor_tp} exceeds the online cluster ({tp}) — using placement TP"
            )
        cfg["tp_size"] = tp
    if cfg.get("mem_fraction_static") is None:
        cfg["mem_fraction_static"] = mem
        rationale.append(f"mem-fraction-static={mem} from leftover UMA after weights")
    # SGLang default port — do not inherit vLLM :8000 from a mixed cookbook line.
    if int(cfg.get("port") or 0) in (0, 8000):
        cfg["port"] = 30000
    native = detected.get("max_position_embeddings")
    if cfg.get("context_length") is None and isinstance(native, int) and native > 0:
        leftover = ram - per - _UMA_RESERVE_GIB
        ctx = 8192
        for cand in _CONTEXT_LADDER:
            if cand > native:
                continue
            need = (float(cand) / 8192.0) * 0.25
            if leftover >= need + _RUNTIME_PAD_GIB:
                ctx = int(cand)
                break
        cfg["context_length"] = min(int(native), ctx)

    # Never launch a second --model; SGLang takes --model-path only.
    if cfg.get("extra_flags"):
        cfg["extra_flags"] = _strip_flag_from_extra(cfg["extra_flags"], "--model")
        cfg["extra_flags"] = _scrub_unexpanded_shell_vars(cfg["extra_flags"], warnings)

    parts = [
        "python -m sglang.launch_server",
        "--model-path",
        str(cfg["model_path"] or model),
        "--host",
        str(cfg.get("host") or "0.0.0.0"),
        "--port",
        str(int(cfg.get("port") or 30000)),
        "--tp-size",
        str(int(cfg.get("tp_size") or 1)),
        "--mem-fraction-static",
        f"{float(cfg.get('mem_fraction_static') or mem):.2f}",
    ]
    if cfg.get("trust_remote_code"):
        parts.append("--trust-remote-code")
    if cfg.get("context_length"):
        parts += ["--context-length", str(int(cfg["context_length"]))]
    if cfg.get("quantization"):
        parts += ["--quantization", str(cfg["quantization"])]
    if cfg.get("tool_call_parser"):
        parts += ["--tool-call-parser", str(cfg["tool_call_parser"])]
    if cfg.get("reasoning_parser"):
        parts += ["--reasoning-parser", str(cfg["reasoning_parser"])]
    if cfg.get("spec_algorithm"):
        # Emit the Unsloth/card spelling; --speculative-algorithm is accepted on parse.
        parts += ["--speculative-algo", str(cfg["spec_algorithm"])]
        if cfg.get("spec_num_steps") is not None:
            parts += ["--speculative-num-steps", str(int(cfg["spec_num_steps"]))]
        if cfg.get("spec_eagle_topk") is not None:
            parts += ["--speculative-eagle-topk", str(int(cfg["spec_eagle_topk"]))]
        if cfg.get("spec_draft_tokens") is not None:
            parts += ["--speculative-num-draft-tokens", str(int(cfg["spec_draft_tokens"]))]
    extra = (cfg.get("extra_flags") or "").strip()
    if extra:
        parts.append(extra)
    argv = " ".join(parts)
    if "$" in argv:
        warnings.append("Stripped leftover $ from SGLang argv")
        argv = re.sub(r"\$\w+|\$\{\w+\}", "", argv)
        argv = re.sub(r"\s+", " ", argv).strip()

    fits = bool(plan.get("fits", True))
    ok_load, load_msg = check_serve_loadability(
        mode=None,
        weights_gib=weights_gib,
        node_ram_gib=ram,
        nodes_used=int(plan.get("nodes_needed") or tp or 1),
        util=float(cfg.get("mem_fraction_static") or mem or WORKFLOW_UTIL),
        reserve_gib=float(plan.get("reserve_gib") or 15.0),
    )
    if fits and not ok_load:
        fits = False
        if load_msg:
            warnings.append(load_msg)
    serve_blocked = not fits
    if serve_blocked:
        warnings.append(
            "SERVE BLOCKED: weights do not fit the online cluster — Start will refuse until "
            "you add nodes or pick a smaller checkpoint."
        )
    rationale.append(
        "SGLang Spark pack: python -m sglang.launch_server (not vllm serve). "
        "Refs: Unsloth NVFP4 SGLang tutorial (NEXTN)."
    )
    return {
        "model": model,
        "engine": "sglang",
        "mode": "auto",
        "confidence": "high" if candidates else ("medium" if hf_config else "low"),
        "label": "SGLang Spark",
        "notes": "Recommend-only. Start sglang yourself (L.A.I.L does not spawn it yet).",
        "serve_blocked": serve_blocked,
        "config": cfg,
        "argv": argv,
        "topology": {
            "nodes": topology.get("nodes", 1),
            "nodes_used": plan.get("nodes_needed", 1),
            "weights_gib": weights_gib,
            "node_ram_gib": ram,
            "fits": fits,
            "tensor_parallel_size": tp,
        },
        "sources": sources,
        "rationale": rationale,
        "warnings": warnings,
        "detected": detected,
        "card_recipes": [
            {
                "score": c.score,
                "section": c.section,
                "raw": (c.raw or "")[:400],
                "selected": i == 0 and not _vendor_candidate_sibling_mismatch(c, model),
                "reasons": c.reasons,
                "config": c.config,
            }
            for i, c in enumerate(candidates[:8])
        ],
    }
