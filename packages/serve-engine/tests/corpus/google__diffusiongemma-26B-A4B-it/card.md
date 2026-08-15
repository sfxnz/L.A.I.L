# DiffusionGemma 26B A4B IT

NVIDIA DGX Spark playbook model. Use the Gemma vLLM container (not gemma4-cu130).

```bash
docker pull vllm/vllm-openai:gemma
```

```shell
vllm serve google/diffusiongemma-26B-A4B-it \
  --attention-backend TRITON_ATTN \
  --diffusion-config '{"canvas_length":256}' \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4
```
