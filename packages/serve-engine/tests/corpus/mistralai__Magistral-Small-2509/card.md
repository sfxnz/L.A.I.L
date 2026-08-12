---
library_name: vllm
license: apache-2.0
tags:
- mistral
- magistral
- reasoning
---

# Magistral Small 2509 (offline corpus fixture)

Official-style mistral-common serve path. Overlay must own load/tokenizer flags once.

## vLLM

```bash
vllm serve mistralai/Magistral-Small-2509 \
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  --enable-auto-tool-choice \
  --tool-call-parser mistral \
  --reasoning-parser mistral \
  --max-model-len 65536
```
