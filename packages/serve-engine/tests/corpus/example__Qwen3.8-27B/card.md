---
pipeline_tag: image-text-to-text
license: apache-2.0
library_name: transformers
---

# Qwen3.8-27B

Dense 27B vision-language model. Native context 262144. Thinking on by default.

## Best Practices

For contexts above 262k, YaRN can extend to 1M. Do not apply this for a normal serve:

```shell
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ... --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' --max-model-len 1000000
```
