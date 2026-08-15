"""Paths and defaults for Local AI Lab."""
from __future__ import annotations

import os
from pathlib import Path

HOME = Path.home()
# L.A.I.L monorepo: prefer LAIL_DATA_DIR / LOCAL_AI_LAB_ROOT, fall back to legacy path
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]  # packages/serve-engine/app -> repo root
APP_ROOT = Path(os.environ.get("LOCAL_AI_LAB_ROOT", os.environ.get("LAIL_ROOT", _DEFAULT_ROOT))).resolve()
DATA_DIR = Path(os.environ.get("LAIL_DATA_DIR", APP_ROOT / "data")).resolve()
RUNS_DIR = DATA_DIR / "runs"
DB_PATH = DATA_DIR / "lab.sqlite"

PIPELINE = HOME / "benchmarks" / "_pipeline"
SPARK_LAB = PIPELINE / "spark_lab.sh"
PROFILES_DIR = PIPELINE / "profiles"
BENCH_PREFILL = PIPELINE / "lib" / "bench_prefill_decode.py"
BENCH_CONCURRENCY = PIPELINE / "lib" / "bench_concurrency.py"
BENCH_WORKFLOW = HOME / "benchmarks" / "qwen36-27b-unsloth-nvfp4" / "bench_workflow.py"
VLLM_RUNTIME = HOME / "vllm-runtime"
GOLDEN_TOOLS = HOME / "lab" / "bin" / "golden_tools.py"

DEFAULT_BASE_URL = os.environ.get("LAB_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_PORT = int(os.environ.get("LAB_PORT", "8000"))
DEFAULT_IMAGE_SAFE = os.environ.get("SPARK_VLLM_IMAGE", "vllm/vllm-openai:v0.27.1")
DEFAULT_IMAGE_MAX = os.environ.get("LAB_VLLM_IMAGE_MAX", "vllm/vllm-openai:v0.27.1")

# Lab Safe envelope (matches spark_envelope.sh)
SAFE_UTIL = 0.4
SAFE_MAX_LEN = 65536
SAFE_MIN_AVAIL_GIB = 60
WORKFLOW_UTIL = 0.85
WORKFLOW_MAX_LEN = 262144

CONTAINER_SAFE = "spark-vllm"
CONTAINER_MAX = "spark-vllm-max"

MODEL_PRESETS: dict[str, dict] = {
    "nvidia/Qwen3.6-27B-NVFP4": {
        "architecture": "Qwen3",
        "param_count": "27B",
        "active_moe_params": None,
        "weights": {
            "dtype": "nvfp4",
            "quant_format": "compressed-tensors / modelopt",
            "group_size": None,
            "calibration": "modelopt",
        },
    },
    "nvidia/Qwen3.6-35B-A3B-NVFP4": {
        "architecture": "Qwen3 MoE",
        "param_count": "35B",
        "active_moe_params": "3B",
        "weights": {
            "dtype": "nvfp4",
            "quant_format": "compressed-tensors / modelopt",
            "group_size": None,
            "calibration": "modelopt",
        },
    },
    "unsloth/Qwen3.6-27B-NVFP4": {
        "architecture": "Qwen3_5MTP",
        "param_count": "27B",
        "active_moe_params": None,
        "weights": {
            "dtype": "nvfp4",
            "quant_format": "compressed-tensors",
            "group_size": None,
            "calibration": "unsloth",
        },
    },
    "unsloth/Qwen3.6-35B-A3B-NVFP4": {
        "architecture": "Qwen3 MoE + MTP",
        "param_count": "35B",
        "active_moe_params": "3B",
        "weights": {
            "dtype": "nvfp4",
            "quant_format": "compressed-tensors",
            "group_size": None,
            "calibration": "unsloth",
        },
    },
}

# Optional fill-in examples for the GUI (click to apply — never auto-applied by the backend).
SERVE_EXAMPLES: dict[str, dict] = {
    "unsloth-35b-spark": {
        "label": "Unsloth 35B NVFP4 (Spark — proven)",
        "model": "unsloth/Qwen3.6-35B-A3B-NVFP4",
        # Unsloth Aug 2026 Spark recipe: flashinfer_b12x + CUTE on vLLM ≥0.27.
        "quantization": "compressed-tensors",
        "kv_cache_dtype": "fp8",
        "moe_backend": "flashinfer_b12x",
        "trust_remote_code": True,
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
        "enable_auto_tool_choice": True,
        "max_num_seqs": 4,
        "docker_env": ["CUTE_DSL_ARCH=sm_121a"],
        "extra_flags": "",
        "mtp": False,
        "notes": "Unsloth Spark path on v0.27.1: --moe-backend flashinfer_b12x and CUTE_DSL_ARCH=sm_121a. Matches Auto-configure.",
    },
    "nvidia-27b": {
        "label": "NVIDIA 27B NVFP4",
        "model": "nvidia/Qwen3.6-27B-NVFP4",
        "quantization": "modelopt_mixed",
        "kv_cache_dtype": "fp8",
        "moe_backend": "",
        "trust_remote_code": True,
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
        "enable_auto_tool_choice": True,
        "max_num_seqs": "",
        "docker_env": [],
        "extra_flags": "",
        "mtp": False,
        "notes": "Dense ModelOpt MIXED_PRECISION — quantization=modelopt_mixed from config.json.",
    },
}

for d in (DATA_DIR, RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)
