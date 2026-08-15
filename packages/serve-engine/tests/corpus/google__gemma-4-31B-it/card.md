# Gemma 4 31B IT

NVIDIA DGX Spark playbook model. Use the Gemma 4 vLLM container.

```bash
docker pull vllm/vllm-openai:gemma4-cu130
```

```shell
vllm serve google/gemma-4-31B-it \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4
```
