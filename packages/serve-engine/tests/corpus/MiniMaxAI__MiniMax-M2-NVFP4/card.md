---
library_name: transformers
tags:
- minimax
- moe
- nvfp4
---

# MiniMax-M2-NVFP4 (offline corpus fixture)

Community / lab NVFP4 path for 2× DGX Spark. Autoconfig overlay owns parsers and compile flags.

## vLLM

```bash
SAFETENSORS_FAST_GPU=1 vllm serve MiniMaxAI/MiniMax-M2-NVFP4 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 65536 \
  --compilation-config '{"cudagraph_mode":"none"}'
```
