---
library_name: transformers
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B-NVFP4

Unsloth NVFP4 of Qwen3.8-27B. Native VL. No vendor cookbook link on this card.

```shell
vllm serve unsloth/Qwen3.8-27B-NVFP4
```

Official 1M YaRN demo (do not apply for a normal serve):

```shell
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ... --hf-overrides '{"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}}}' --max-model-len 1000000
```
