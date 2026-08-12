# Vendor vLLM cookbook (fixture body)

## DGX Spark

```shell
export CUTE_DSL_ARCH=sm_121a
vllm serve example/Cookbook-Model-NVFP4 \
  --quantization modelopt \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --max-model-len 65536 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```
