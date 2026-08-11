# DeepSeek-V4-Flash (fixture)

Generic card recipe (wrong for GB10 — overlay must win):

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --data-parallel-size 8
```
