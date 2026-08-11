---
library_name: transformers
---

# Test model card (mixed NVFP4+FP8 MoE)

## vLLM Run Instructions

Bare serve:

```
vllm serve example/Mixed-MoE-NVFP4
```

DGX Spark optimized path (card recommends flashinfer — unsafe for FP8 MoE layers):

```
export CUTE_DSL_ARCH=sm_121a
vllm serve example/Mixed-MoE-NVFP4 --moe-backend flashinfer_b12x
```

Optional MTP:

```
vllm serve example/Mixed-MoE-NVFP4 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

Demo only:

```
vllm serve example/Mixed-MoE-NVFP4 --trust-remote-code --dtype bfloat16 --max-model-len 4096
```
