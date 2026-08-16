# Run Unsloth Dynamic NVFP4 Guide (fixture)

Mixed vendor page: Spark tutorial is the 35B-A3B-Fast line; a 27B serve
also appears under the vLLM heading (mirrors https://unsloth.ai/docs/basics/nvfp4).

We added **FP8 KV cache** calibration for 2x longer context lengths.

### vLLM Tutorial

You can run all NVFP4 models in vLLM. If you have a DGX Spark you must use
`--moe-backend flashinfer_b12x` or you will get much slower inference.

To install vLLM in a separate venv:

```
uv venv unsloth-nvfp4-env --python 3.13
source unsloth-nvfp4-env/bin/activate
uv pip install "vllm>=0.25.0" "flashinfer-python>=0.6.13" "nvidia-cutlass-dsl>=4.5.2" \
    --torch-backend=auto
```

Then to serve the 35B Fast variant:

```
vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
```

Change `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` to the NVFP4 quant names!

### DGX Spark Tutorial

Then to serve in vLLM for DGX Spark:

```
export CUTE_DSL_ARCH=sm_121a
vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --moe-backend flashinfer_b12x
```

#### vLLM

Then to serve the 27B variant:

```
vllm serve unsloth/Qwen3.8-27B-NVFP4
```
