---
pipeline_tag: text-generation
license: apache-2.0
library_name: transformers
---

# Qwen3.8-27B

Dense Qwen3.x 27B (synthetic fixture modeled on NVIDIA Qwen3.6-27B usage).
No family overlay. Quantization is ModelOpt MIXED — the repo id does not contain `nvfp4`.

## Usage

To serve this checkpoint with [vLLM](https://github.com/vllm-project/vllm), you can start the docker `vllm/vllm-openai:nightly` and run the sample command below:

```sh
vllm serve Qwen/Qwen3.8-27B --port 8000 --quantization modelopt --max-model-len 262144 --reasoning-parser qwen3
```
