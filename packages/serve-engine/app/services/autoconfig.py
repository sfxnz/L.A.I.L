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
) -> tuple[Optional[str], Optional[str]]:
    """GET text body. Follows redirects. Returns (body, error)."""
    headers = {"User-Agent": HF_UA, "Accept": "*/*"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout) as resp:
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
            out["mtp"] = "mtp" in cfg.lower()
            mm = re.search(r"num_speculative_tokens[\"']?\s*:\s*(\d+)", cfg)
            if mm:
                out["mtp_num_tokens"] = int(mm.group(1))
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
        "match": {"all": ["deepseek"], "any": ["v4", "dspark", "flash"]},
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


def _family_overlay(model: str, detected: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Match a model id against the overlay registry. Returns None when the card is
    authoritative (normal models). Data-driven so future models need no code change."""
    mid = (model or "").lower()
    for ov in _load_overlays():
        m = ov.get("match") or {}
        all_terms = [str(t).lower() for t in (m.get("all") or [])]
        any_terms = [str(t).lower() for t in (m.get("any") or [])]
        if all(t in mid for t in all_terms) and (not any_terms or any(t in mid for t in any_terms)):
            return ov
    return None


def _cluster_topology() -> dict[str, Any]:
    """Live cluster shape for serve planning. Never raises — falls back to single-node."""
    fallback = {
        "nodes": 1,
        "node_list": [],
        "head": None,
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
            return fallback
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
        return fallback


# ─── Placement engine (hardware-aware, N-node, model-agnostic) ────────────────

# Per-GPU arch hints for compile targets. Unknown skus fall back to GB10 (current lab).
_SKU_ARCH = {
    "gb10": {"cute_dsl_arch": "sm_121a", "torch_arch": "12.1a"},
    "gb200": {"cute_dsl_arch": "sm_100a", "torch_arch": "12.0a"},
    "gb300": {"cute_dsl_arch": "sm_103a", "torch_arch": "12.0a"},
}
_DEFAULT_NODE_RAM_GIB = 121.7  # GB10 UMA when a node has not been probed yet


def estimate_weights_gib(model: str, hf_config: Optional[dict]) -> Optional[float]:
    """Best-effort weight size for ANY model. Order: exact HF blob sum → config param
    estimate → None. Used by the placement engine to decide nodes_needed + util."""
    # 1) Exact: sum safetensors/bin blobs from the HF API (most accurate).
    try:
        body, err = _http_get(f"https://huggingface.co/api/models/{model}?blobs=true", timeout=20.0)
        if body and not err:
            d = json.loads(body)
            tot = 0
            for f in d.get("siblings") or []:
                n = f.get("rfilename", "")
                if n.endswith((".safetensors", ".bin", ".gguf")):
                    tot += f.get("size") or 0
            if tot > 0:
                return round(tot / (1024**3), 1)
    except Exception:
        pass
    # 2) Estimate from config: params × bytes/param (quant-aware).
    try:
        qc = (hf_config or {}).get("quantization_config") or {}
        nbits = 0
        for g in (qc.get("config_groups") or {}).values():
            if isinstance(g, dict):
                w = g.get("weights") or {}
                if isinstance(w, dict) and w.get("num_bits"):
                    nbits = int(w["num_bits"])
                    break
        hidden = (hf_config or {}).get("hidden_size")
        layers = (hf_config or {}).get("num_hidden_layers")
        if hidden and layers:
            # rough param count for dense-ish models: ~12 * layers * hidden^2 (+emb)
            params = 12 * layers * hidden * hidden
            bpp = (nbits / 8.0) if nbits in (4, 8) else 2.0  # fp4/fp8/bf16
            return round(params * bpp / (1024**3), 1)
    except Exception:
        pass
    return None


def _node_ram_gib(node: Optional[dict]) -> float:
    if node and isinstance(node.get("ram_gib"), (int, float)) and node["ram_gib"]:
        return float(node["ram_gib"])
    return _DEFAULT_NODE_RAM_GIB


def _gpu_arch_env(nodes: list[dict]) -> dict[str, str]:
    """Compile-target env for the cluster's GPU sku (future hardware: extend _SKU_ARCH)."""
    sku = ""
    for n in nodes:
        sku = (n.get("gpu_sku") or "").lower()
        if sku:
            break
    key = next((k for k in _SKU_ARCH if k in sku), "gb10")
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
    mode: str,
    overlay: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Decide nodes_needed / TP / PP / per-node util from real weights + probed hardware.

    One rule (UMA): per-node util ≈ (per-node weights + 15 GiB headroom) / node RAM,
    clamped to 0.40 … 0.85. Fits on minimal nodes; spreads (TP) only when a single node
    can't hold the weights. Overlay may pin util (its recipe is authoritative).
    """
    nodes = topology.get("node_list") or []
    n_avail = max(1, int(topology.get("nodes") or 1))
    head = topology.get("head") or (nodes[0] if nodes else None)
    node_ram = _node_ram_gib(head)
    reserve = 15.0  # OS + Hermes + runtime headroom per node (lab rule)

    # Usable weights capacity per node at the util ceiling.
    util_cap = 0.85 if mode == "workflow_max" else 0.40
    single_fit_gib = node_ram * util_cap - reserve

    if weights_gib and weights_gib > 0:
        nodes_needed = 1
        while nodes_needed < n_avail and (weights_gib / nodes_needed) > single_fit_gib:
            nodes_needed += 1
        fits = (weights_gib / nodes_needed) <= (node_ram * 0.85 - reserve)
        per_node_weights = weights_gib / nodes_needed
        util = per_node_weights and min(0.85, max(0.40, (per_node_weights + reserve) / node_ram))
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
        _sanitize_moe_backend_on_candidate(c, detected)
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
    flashinfer_unsafe = _flashinfer_b12x_unsafe_for_checkpoint(det)
    marlin_unsafe = _marlin_unsafe_for_checkpoint(det)
    if moe:
        if moe == "flashinfer_b12x" and flashinfer_unsafe:
            reasons.append(f"--moe-backend {moe} (will be stripped — unsafe for checkpoint)")
        elif moe == "marlin" and marlin_unsafe:
            reasons.append(
                f"--moe-backend {moe} (will be stripped — unsupported for this MoE on vLLM 0.25)"
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
    if det.get("quant_flag") and cfg.get("quantization") == det.get("quant_flag"):
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
            "PENALTY: moe_backend=marlin unsupported for this MoE on vLLM 0.25 "
            "(ValueError: not supported for unquantized MoE)"
        )

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


def _marlin_unsafe_for_checkpoint(detected: dict[str, Any]) -> bool:
    """True when --moe-backend marlin crashes on this checkpoint (vLLM 0.25.x).

    Observed on NVIDIA Qwen3.6-35B-A3B-NVFP4 (ModelOpt MoE):
      ValueError: moe_backend='marlin' is not supported for unquantized MoE.
      Expected one of ['triton', 'flashinfer_trtllm', 'flashinfer_cutlass', 'aiter'].
    Cards still ship marlin recipes; leave moe-backend empty (auto).
    """
    if not detected:
        # MoE/NVFP4 cards are the risk class — if unknown, still treat marlin as unsafe
        return True
    if detected.get("is_moe"):
        return True
    if detected.get("has_nvfp4") or detected.get("quant_flag") in ("modelopt", "compressed-tensors"):
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
    if moe == "flashinfer_b12x" and _flashinfer_b12x_unsafe_for_checkpoint(det):
        c.config["moe_backend"] = ""
        if not any("cleared moe_backend" in r for r in c.reasons):
            c.reasons.append(
                "cleared moe_backend=flashinfer_b12x on this recipe (unsafe for checkpoint)"
            )
    elif moe == "marlin" and _marlin_unsafe_for_checkpoint(det):
        c.config["moe_backend"] = ""
        if not any("cleared moe_backend" in r for r in c.reasons):
            c.reasons.append(
                "cleared moe_backend=marlin on this recipe "
                "(vLLM 0.25: marlin unsupported for this MoE path — use auto)"
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

    moe = (cfg.get("moe_backend") or "").strip()
    if moe == "marlin" and _marlin_unsafe_for_checkpoint(detected):
        cfg["moe_backend"] = ""
        warnings.append(
            "Card recipe used --moe-backend marlin, but vLLM 0.25 rejects marlin on this MoE "
            "path (ValueError: moe_backend='marlin' is not supported for unquantized MoE). "
            "Cleared moe-backend so vLLM auto-selects a supported backend."
        )
        rationale.append(
            "SAFETY (vLLM 0.25 > card flag): removed marlin MoE backend — use auto"
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
    should be stable: correct quant, safe moe auto, no speculative decode.
    """
    # MTP is opt-in — many cards include it; first boot should not.
    if cfg.get("mtp"):
        cfg["mtp"] = False
        cfg["mtp_num_tokens"] = 2
        if cfg.get("extra_flags"):
            cfg["extra_flags"] = _strip_flag_from_extra(cfg["extra_flags"], "--speculative-config")
        warnings.append(
            "Disabled MTP / speculative decode for first boot (card had it on). "
            "Re-enable MTP in the form after a healthy serve if you want it."
        )
        rationale.append("FIRST BOOT: MTP off (stable serve; re-enable later if needed)")

    # Empty moe = vLLM auto — preferred on Spark unless user forces a known-good backend
    moe = (cfg.get("moe_backend") or "").strip().lower()
    if moe in ("marlin",) or (moe == "flashinfer_b12x" and _flashinfer_b12x_unsafe_for_checkpoint(detected)):
        # already handled by checkpoint safety; ensure empty
        cfg["moe_backend"] = ""

    # Prefer empty moe for MoE first boot even if card set something exotic we didn't list
    if detected.get("is_moe") and moe and moe not in ("", "auto", "triton"):
        # Keep triton if card set it; clear unknown/risky backends
        if moe not in ("triton", "flashinfer_trtllm", "flashinfer_cutlass", "aiter"):
            cfg["moe_backend"] = ""
            if moe not in ("marlin", "flashinfer_b12x"):  # already warned
                warnings.append(
                    f"Cleared --moe-backend {moe} for first-boot MoE safety (vLLM auto)."
                )
                rationale.append(f"FIRST BOOT: moe-backend {moe!r} → empty (auto)")

    # Lab Safe: don't carry card util=0.85 into lab_safe form without clamp (envelope handles)
    # Workflow Max first boot on 35B MoE: 262k ctx is OK only with enough util headroom —
    # leave card max-len but note memory risk.
    if mode == "workflow_max" and isinstance(cfg.get("max_model_len"), int):
        if cfg["max_model_len"] >= 200000 and detected.get("is_moe"):
            rationale.append(
                f"Card max-model-len={cfg['max_model_len']} kept for Workflow Max — "
                "watch free UMA; drop to 65536 if load OOMs"
            )


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


def _apply_mode_envelope(cfg: dict[str, Any], mode: str, rationale: list[str], card_set_max_len: bool, card_set_util: bool = False) -> None:
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

    # Lab Safe clamps util, but never below a card/overlay-specified floor (large NVFP4 /
    # DSpark weights need the card's util just to load — clamping to 0.4 would brick the boot).
    if mode == "lab_safe" and cfg.get("util") is not None and cfg["util"] > SAFE_UTIL + 1e-9 and not card_set_util:
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

    # Model-family overlay (e.g. DeepSeek DSpark) + live cluster topology. The overlay
    # supplies the correct serve path when the card's generic recipe is wrong; topology
    # decides TP/nnodes/fabric from how many Sparks are actually online.
    overlay = _family_overlay(model, detected)
    topology = _cluster_topology()
    # Real weight size (exact HF blob sum → config estimate) drives the placement engine.
    weights_gib = estimate_weights_gib(model, hf_config)

    # ── Parse card for best vllm serve (scored with config.json knowledge) ──
    # When a family overlay matches, the card's generic recipe is wrong for this
    # checkpoint — skip card-candidate fill entirely and let the overlay drive.
    candidates: list[ServeCandidate] = []
    if readme and not overlay:
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

    # Gaps only: HF config.json / tags (still from the model on the hub)
    _fill_from_config_detection(cfg, detected, rationale)

    # config.json is ground truth for quant layout — fix card flags that crash
    _apply_checkpoint_safety(cfg, detected, warnings, rationale)

    # Model-family overlay wins over the card when the card's recipe is wrong for this
    # checkpoint's Spark path (DeepSeek DSpark: custom image + b12x + nvfp4_ds_mla).
    if overlay:
        overlay_cfg = overlay["config"]
        for k, v in overlay_cfg.items():
            if k == "docker_env":
                cfg["docker_env"] = _dedupe_env(list(cfg.get("docker_env") or []) + list(v))
            elif k == "extra_flags":
                cfg["extra_flags"] = ((cfg.get("extra_flags") or "") + " " + v).strip()
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

    # Mode envelope for util / defaults when card silent. util/max-len set by the card OR a
    # family overlay are authoritative — the envelope only fills gaps, never overrides them.
    card_set_util = cfg.get("util") is not None
    card_set_max_len = cfg.get("max_model_len") is not None
    _apply_mode_envelope(cfg, mode, rationale, card_set_max_len=card_set_max_len, card_set_util=card_set_util)

    max_pos = detected.get("max_position_embeddings")
    if isinstance(max_pos, int) and max_pos > 0 and cfg.get("max_model_len"):
        if cfg["max_model_len"] > max_pos:
            cfg["max_model_len"] = max_pos
            rationale.append(f"Capped max-model-len to config max_position_embeddings={max_pos}")

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
    cfg["docker_env"] = env_out

    # If MTP structured flag is on, strip duplicate --speculative-config from extras
    if cfg.get("mtp") and cfg.get("extra_flags"):
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
        "fits": plan.get("fits", True),
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
        "mode": mode,
        "confidence": confidence,
        "label": label,
        "notes": notes,
        "card_url": card_url,
        "from_website": from_website,
        "hf_token_ok": token_ok,
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
