---
pipeline_tag: text-generation
---

# NVIDIA-style dense NVFP4 card

## Usage

```
vllm serve nvidia/Example-27B-NVFP4 --port 8000 --quantization modelopt --max-model-len 262144 --reasoning-parser qwen3
```
