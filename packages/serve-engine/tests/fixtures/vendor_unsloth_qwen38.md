# Qwen3.8 - How to Run Locally (fixture)

We added **FP8 KV cache** calibration for 2x longer context lengths.

#### **vLLM:**

Then to serve the 27B variant:

```shell
vllm serve unsloth/Qwen3.8-27B-NVFP4 --kv-cache-dtype fp8 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice --trust-remote-code
```

#### **SGLang:**

```shell
python -m sglang.launch_server --model-path unsloth/Qwen3.8-27B-NVFP4 --speculative-algo NEXTN \
     --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```
