---
library_name: transformers
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
pipeline_tag: text-generation
language:
- en
- es
- fr
- de
- it
- ja
tags:
- nvidia
- pytorch
- nemotron-3.5
track_downloads: true
---

# NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4

<div align="center" style="line-height: 1;">
  <a href="https://developer.nvidia.com/nemotron" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://img.shields.io/badge/🏠Nemotron Developer Page-Learn More Here!-536af5?color=76B900&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://discord.gg/9xpKQtVvrk" target="_blank" style="margin: 2px;">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-NVIDIA%20AI%20Developer-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div style="text-align: center; line-height: 1;">
  <a href="https://openmdw.ai/license/1-1/" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-OpenMDW--1.1-f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

![](./accuracy_plot.png)

## Model Summary

| | |
|:---|:---|
| **Total Parameters** | 30B (3B active) |
| **Architecture** | MoE - Mamba-2 + MoE + Attention hybrid |
| **Context Length** | Up to 1M tokens |
| **Single-GPU Deployment** | 1× DGX Spark (GB10) or 1× H100 |
| **Supported Hardware** | NVIDIA Blackwell (DGX Spark / GB10, GB200, GeForce RTX 5090); NVIDIA Hopper (H100, H200); NVIDIA Ampere via W4A16 |
| **Supported Languages** | English (and coding languages), Spanish, French, German, Italian, Japanese |
| **Speculative Decoding** | DSpark for low-concurrency Data Centre and DGX Spark Workflows — [Read more below](#speculative-decoding-strategies), also provided are MTP (Multi-Token Prediction) and DFlash |
| **Recommended Sampling** | Temperature 1.0, Top_P 0.95 |
| **Best For** | Long-running autonomous agents, sub-agent workhorse deployments, and efficient local inference on personal hardware |
| **License** | [OpenMDW License Agreement, version 1.1](https://raw.githubusercontent.com/OpenMDW/OpenMDW/refs/heads/main/1.1/LICENSE.OpenMDW-1.1) |
| **Release Date** | August 11, 2026 |

## Model Overview

**Model Developer:** NVIDIA Corporation

**Model Dates:** December 2025 - May 2026

**Data Freshness:**

* The pre-training data has a cutoff date of September 2025.
* The post-training data has a cutoff date of May 2026.

### What is Nemotron?

NVIDIA Nemotron™ is a family of open models with open weights, training data, and recipes, delivering leading efficiency and accuracy for building specialized AI agents.

## Description

**NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4** is a large language model (LLM) trained by NVIDIA.

The model employs a hybrid **Mixture-of-Experts** architecture, utilizing interleaved Mamba-2 and MoE layers, along with select Attention layers. The Lightning 3.5 model is released alongside a number of speculative decoding methods for faster text generation. The model has **3B active parameters** and **30B parameters in total**.

This model is ready for commercial use.

## Quick Start

To get quickly started on DGX Spark (GB10) you can use the following command.

Grab the model:

```shell
export MODEL_CKPT=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
export DSPARK_CKPT=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark
```
 
Run it with vLLM — this recipe uses DSpark speculative decoding, tuned for DGX Spark. (vLLM Nightly: `vllm/vllm-openai:v0.27.1`)
 
```shell
vllm serve --model $MODEL_CKPT \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --speculative_config.num_speculative_tokens 3 \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 \
  --speculative_config.model $DSPARK_CKPT \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```

For more details on how to deploy and use the model — see the [Quick Start Guide](#quick-start-guide) below!

## License/Terms of Use

**Governing Download Terms:** Use of this model is governed by the [OpenMDW-1.1 model license](https://openmdw.ai/license/1-1/).

## Benchmarks

### Reasoning Benchmark Evaluations

We evaluated our model on the following benchmarks:

| Task | Nemotron-3.5-Lightning-30B-A3B-BF16 | Nemotron-3.5-Lightning-30B-A3B-NVFP4 |
| --- | --- | --- |
| **General Knowledge** | | |
| MMLU Pro | 81.94 | 81.62 |
| AA-Omniscience | 17.50 | 16.63 |
| **Reasoning** | | |
| GPQA Diamond (no tools) | 75.44 | 75.57 |
| HLE (text-only, no tools) | 11.72 | 10.47 |
| SciCode | 32.60 | 31.38 |
| **Coding & Agentic** | | |
| SWE-bench Verified | 51.56 | 52.80 |
| SWE-bench Multilingual | 39.33 | 36.47 |
| Terminal-Bench 2.1 | 24.58 | 23.46 |
| PinchBench | 85.37 | 83.43 |
| BrowseComp | 36.97 | 36.81 |
| τ³-bench (Banking) | 9.28 | 9.48 |
| GDPval-AA-V2 | 832 | 865 |
| **Instruction Following** | | |
| IFBench (loose) | 71.88 | 72.88 |
| **Long Context** | | |
| AA-LCR | 52.00 | 49.19 |

Accuracy numbers measured by NVIDIA under a consistent harness (NeMo Gym / Nemo Evaluator SDK); they may differ from vendors' self-reported numbers.

For reproducibility, the evaluation recipes, installation instructions, and commands for NVIDIA Nemotron 3.5 Lightning were collected and published in [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym/tree/main/nemotron_recipes/lightning-3.5/reproducibility.md). The reported results cover the release evaluation suite, including knowledge and reasoning, instruction following, coding, agentic, tool-use, and long-context. Most evaluations use NeMo Gym-native harnesses while a small subset, including SWE-Bench and Terminal-Bench, used [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) natively. The published recipes specify the benchmark-specific containers, prompts, inference parameters, parser configurations, and scoring settings used to produce the results. 

> These numbers were measured with and apply to the official [NVFP4 checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)

#### Agentic Coding Benchmarks

Additional harness-level coding-agent results for SWE-Bench Verified and Terminal-Bench 2.1 are shown below.

![Agentic Coding Benchmarks](./agentic_coding_benchmarks.png)

### Deployment Geography: Global

### Use Case

NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 is a general purpose reasoning and chat model intended to be used in English and coding languages. Other non-English languages (Spanish, French, German, Italian, Japanese) are also supported. Intended for developers designing AI Agent systems, chatbots, RAG systems, and other AI-powered applications. Also suitable for typical instruction-following tasks.

### Release Date

[Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) — 08/11/2026

## Model Architecture

- **Architecture Type:** Mixture-of-Experts Hybrid (Mamba + Transformer)
- **Network Architecture:** Nemotron-3-Lightning + Multi-Token Prediction (MTP)
- **Number of model parameters:** 30B Total / 3B Active

## Model Design

The model was pre-trained with over 20T tokens and supports up to 1M context length. The pre-training phase used an NVFP4 recipe. The model includes **Multi-Token Prediction (MTP)** layers, which predict multiple future tokens to provide richer training signals.

## Training Methodology

Stage 1: Pre-Training

* NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 model was pre-trained using an NVFP4 recipe with crawled and synthetic code, math, science, and general knowledge data.
* Software used for pre-training: [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

Stage 2: Continued Pre-Training for Multi-Token Prediction (MTP)

* The model underwent a continued pre-training phase to train its **Multi-Token Prediction (MTP)** layers. In this stage, MTP heads learn to predict multiple future tokens, providing richer training signals to the base model. This phase aligns the MTP layers with the base model's distribution.

Stage 3: Supervised Fine-Tuning

* The model was further fine-tuned on synthetic code, math, science, tool calling, instruction following, structured outputs, and general knowledge data. This stage incorporated data designed to support long-range retrieval and multi-document aggregation.

Stage 4: Reinforcement Learning

* The model underwent multi-environment reinforcement learning using GRPO (Group Relative Policy Optimization) across math, code, science, instruction following, multi-step tool use, multi-turn conversations, and structured output environments. It utilized an asynchronous RL architecture that decouples training from inference and leverages MTP to accelerate rollout generation.
* Software used for reinforcement learning: [NeMo RL](https://github.com/NVIDIA-NeMo/RL), [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)

Stage 5: Post-training Quantization (PTQ)

* We performed post-training quantization (PTQ) with Nvidia Model Optimizer using the following recipe: Four Over Six NVFP4 (a variant of static MSE calibration) W4A16 on routed and shared experts, FP8 per-tensor dynamic scales on mamba in_proj/out_proj and KV cache. We used a subset of the Nemotron Ultra validation set for calibration with 1000 samples at 32k token length.

NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 is a result of the above work.

## Input

- **Input Type(s):** Text
- **Input Format(s):** String
- **Input Parameters:** One-Dimensional (1D): Sequences
- **Other Properties Related to Input:** Maximum context length up to 1M tokens. Supported languages include English, Spanish, French, German, Italian, and Japanese.

## Output

- **Output Type(s):** Text
- **Output Format:** String
- **Output Parameters:** One-Dimensional (1D): Sequences
- **Other Properties Related to Output:** Maximum context length up to 1M tokens

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration

- **Runtime Engine(s):** PyTorch
- **Supported Hardware Microarchitecture Compatibility:** NVIDIA Blackwell; NVIDIA Hopper (NVFP4 / W4A16); NVIDIA Ampere (W4A16)
- **Preferred/Supported Operating System(s):** Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s)

* GA (08/11/2026)

## **Quick Start Guide**
 
All deployment snippets below assume:
 
```shell
export MODEL_CKPT=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
```
 
And for DSpark:
 
```shell
export DSPARK_CKPT=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark
```
 
### **Speculative Decoding Strategies**
 
Lightning 3.5 ships with two external draft models for speculative decoding as well as MTP (Multi-Token Prediction). While we currently recommend DSpark for all cases - your usecase may align with DFlash and MTP:
 
* **DSpark:** A semi-autoregressive speculative-decoding drafter that proposes a whole block of candidate tokens in a single forward pass from a parallel backbone. This is recommended for DGX Spark, as well as low-concurrency data centre deployments.
* **DFlash:** A speculative-decoding drafter that uses a lightweight block-diffusion model to generate an entire draft block in one forward pass.
* **MTP:** A modeling technique that trains the network to predict several future tokens at each position instead of only the next one.
 

### **vLLM** 

> For more indepth instructions on how to deploy through vLLM, head [here](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/vllm_cookbook.ipynb)
 
* vLLM Nightly: `vllm/vllm-openai:v0.27.1`
 
#### **1x DGX Spark (GB10)**
 
**Specdec method - DSpark:**
 
```shell
vllm serve --model $MODEL_CKPT \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --max-model-len 1048576 \
  --enable-prefix-caching \
  --speculative_config.num_speculative_tokens 3 \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 \
  --speculative_config.method dspark \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```
 
#### **1x H100**
 
For max throughput deployments, use the following configuration, no speculative decoding strategy is best for this serving configuration, and due to memory constraints the Mamba cache `dtype` is set as FP16:
 
```shell
vllm serve --model $MODEL_CKPT \
    --max-num-seqs 256 \
    --max-num-batched-tokens 16384 \
    --enable-prefix-caching \
    --async-scheduling \
    --mamba-backend flashinfer \
    --moe-backend humming \
    --linear-backend humming \
    --mamba-ssu-algorithm horizontal \
    --mamba-cache-mode align \
    --mamba-ssm-cache-dtype float16 \
    --enable-mamba-cache-stochastic-rounding \
    --mamba-cache-philox-rounds 5 \
    --reasoning-parser nemotron_v3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice
```
 
For interactive usage scenarios (achieving 40+ TPS/User) use a lower concurrency (<=128) with DSpark:
 
```shell
vllm serve --model $MODEL_CKPT \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    --async-scheduling \
    --speculative_config.model $DSPARK_CKPT \
    --speculative_config.num_speculative_tokens 3 \
    --mamba-ssu-algorithm horizontal \
    --mamba-backend flashinfer \
    --mamba-ssm-cache-dtype float16 \
    --enable-mamba-cache-stochastic-rounding \
    --mamba-cache-philox-rounds 5 \
    --reasoning-parser nemotron_v3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice
```
 
#### **8x H100**
 
For long-context, multi-GPU serving (TP8 with expert parallelism):
 
```shell
vllm serve --model $MODEL_CKPT \
    --mamba-backend flashinfer \
    --async-scheduling \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --enable-expert-parallel \
    --tensor-parallel-size 8 \
    --reasoning-parser nemotron_v3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice
```
 
#### **1x GB200**
 
```shell
vllm serve --model $MODEL_CKPT \
    --max-num-batched-tokens 10240 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --speculative_config.model $DSPARK_CKPT \
    --speculative_config.num_speculative_tokens 5 \
    --mamba-backend flashinfer \
    --reasoning-parser nemotron_v3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice
```
 
#### **W4A16 — Ampere**
 
The same checkpoint also serves via W4A16 kernels, extending coverage to Ampere-class GPUs:
 
```shell
vllm serve --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --moe-backend humming \
    --linear-backend humming \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --enable-prefix-caching \
    --async-scheduling \
    --quantization modelopt_fp4 \
    --mamba-backend flashinfer \
    --mamba-cache-mode align \
    --mamba-ssu-algorithm simple \
    --reasoning-parser nemotron_v3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice
```
 
* **Context Length:** The H100 and GB200 snippets above serve the model's full 1M-token context window by default. If you're memory-constrained — or want more KV-cache headroom at high concurrency — lower `--max-model-len` to match your workload.

### **TensorRT-LLM**

> For more indepth instructions on how to deploy through TensorRT-LLM, head [here](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/trtllm_cookbook.ipynb)

Container: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24`

#### **1x H100**

```shell
cat > nemotron-35-lightning-nvfp4-mtp.yaml << EOF
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  mamba_state_config:
     periodic_snapshot_interval: 8192
  free_gpu_memory_fraction: 0.8
  mamba_ssm_cache_dtype: float16
  mamba_ssm_stochastic_rounding: true
  mamba_ssm_philox_rounds: 5
moe_config:
   backend: MARLIN
nvfp4_gemm_config:
  allowed_backends: [marlin, cutlass, cublaslt, cuda_core]
cuda_graph_config:
    enable_padding: true
    max_batch_size: 8
speculative_config:
  decoding_type: MTP
  max_draft_len: 3
  allow_advanced_sampling: true
enable_chunked_prefill: true
num_postprocess_workers: 4
print_iter_log: true
stream_interval: 10
disable_overlap_scheduler: false
EOF

trtllm-serve \
$MODEL_CKPT \
--max_batch_size 8 \
--max_num_tokens 8192 \
--reasoning_parser nemotron-v3 \
--tool_parser qwen3_coder \
--config nemotron-35-lightning-nvfp4-mtp.yaml
```

* **Context length:** The command above serves the model's full 1M-token context window by default. If you're memory-constrained — or want more KV-cache headroom at higher concurrency — lower `--max_seq_len` to match your workload.

### **SGLang**

> For more indepth instructions on how to deploy through SGLang, head [here](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/sglang_cookbook.ipynb)

* Container: `lmsysorg/sglang:dev-nemotron3-5-lighting`

#### **1x H100**

```shell
sglang serve \
    --model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --max-running-requests 256 \
    --trust-remote-code \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.9 \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mamba-backend flashinfer \
    --mamba-radix-cache-strategy extra_buffer \
    --reasoning-parser nemotron_3 \
    --tool-call-parser qwen3_coder
```

* **Context length:** The command above serves the model's full 1M-token context window by default. If you're memory-constrained — or want more KV-cache headroom at higher concurrency — set `--context-length` to a smaller value.

### **API Client**

The examples below use the OpenAI-compatible client and work with any of the serving backends above. All backends serve on port `8000` (vLLM and TRT-LLM by default; SGLang via `$PORT=8000`), so the `base_url` works as-is. Recommended sampling settings are **Temperature 1.0** and **Top_P 0.95**.

The vLLM snippets above register the model as `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` via `--served-model-name`. For the other backends — or if you change that flag — copy the identifier returned by `GET /v1/models` into `MODEL` below.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
```

Lightning 3.5 exposes reasoning control through chat-template kwargs: thinking enabled (the default), and thinking disabled for direct answers.

<details>
<summary><b>Reasoning ON / OFF and streaming examples: Click to expand!</b></summary>

**Reasoning ON (default)**

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Write a haiku about GPUs"}],
    max_tokens=16000,
    temperature=1.0,
    top_p=0.95,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}}
)
print(response.choices[0].message.content)
```

**Reasoning OFF**

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "What is the capital of Japan?"}],
    max_tokens=16000,
    temperature=1.0,
    top_p=0.95,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)
print(response.choices[0].message.content)
```

**Streaming**

```python
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Explain speculative decoding in two sentences"}],
    max_tokens=16000,
    temperature=1.0,
    top_p=0.95,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

</details>

#### **Tool Calling**

The TRT-LLM snippet above already launches with the required parsers (`--reasoning_parser nemotron-v3 --tool_parser qwen3_coder`). For vLLM, add the following to any serve command above: 

```shell
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3
```

**NOTE:** For coding agents, add `extra_body={"chat_template_kwargs": {"force_nonempty_content": True}}` to the API call, as shown below.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "What's the weather in Santa Clara?"}],
    tools=tools,
    max_tokens=16000,
    temperature=1.0,
    top_p=0.95,
    extra_body={"chat_template_kwargs": {"force_nonempty_content": True}},
)
print(response.choices[0].message.tool_calls)
```

## Training, Testing, and Evaluation Datasets

# Training

**Data Modality:** Text
**Training Data Size:** More than 20 Trillion Tokens
**Dataset partition:** *Training [100%], testing [0%], validation [0%]*
**Time period for training data collection:** 2013 to December 2025
**Time period for testing data collection:** 2013 to December 2025
**Time period for validation data collection:** 2013 to December 2025
**Data Collection Method by dataset:** Hybrid: Automated, Manually-Collected, Synthetic
**Labeling Method by dataset:** Hybrid: Automated, Manually-Labeled, Synthetic

NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 is pre-trained on a large corpus of high-quality curated and synthetically-generated data. It is trained in the English language, as well as 19 other spoken languages and 43 programming languages. Our sources cover a variety of document types such as: webpages, dialogue, articles, and other written materials. The corpus spans domains including legal, math, science, finance, and more. We also include a small portion of question-answering, and alignment style data to improve model accuracy. The model was pre-trained for more than 20 trillion tokens.

The post-training corpus for NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 consists of high-quality curated and synthetically-generated data. Primary languages used for post-training include English, French, German, Italian, Japanese, Spanish, and Chinese.

These datasets, such as FinePDFs, EssentialWeb, HotpotQA, SQuAD, and HelpSteer3, do not collectively or exhaustively represent all demographic groups (and proportionally therein). For instance, these datasets do not contain explicit mentions of demographic classes such as age, gender, or ethnicity in 64-99% of samples, depending on the source. In the subset where such terms are present, document-based datasets (FinePDFs and EssentialWeb) contain representational skews, such as references to "male" outnumbering those to "female", and mentions of "White" as the most frequent among ethnic identifiers (comprising 43-44% of ethnicity mentions). To mitigate these imbalances, we recommend considering evaluation techniques such as bias audits, fine-tuning with demographically balanced datasets, and mitigation strategies like counterfactual data augmentation to align with the desired model behavior. This evaluation used a 3,000-sample subset per dataset, identified as the optimal threshold for maximizing embedder accuracy.

During post-training, we generate synthetic data by distilling trajectories, solutions, and translations from strong teacher models and agent systems, often grounded in real tasks or documents and aggressively filtered for quality. For math, code, and science, we start from curated problem sets and use open source permissive models such as GPT-OSS-120B to produce step-by-step reasoning traces, candidate solutions, best-of-n selection traces, and verified CUDA kernels. For long-context and science, we build synthetic QA and reasoning data by retrieving passages from long documents, generating MCQ/OpenQA questions and answers, and paraphrasing them into multiple prompt/response formats to ensure diversity. Across all pipelines we stack automated verification—compilers, numerical checks, language identification—to ensure our data is high quality.

For all domains, we apply a unified data filtering pipeline to ensure that only high-quality, license-compliant, and verifiable samples are used for post-training. We first discard malformed examples using structural checks (e.g., missing tool definitions when tool calls are present). We then aggressively filter reasoning traces exhibiting pathological repetition, such as repeated n-grams within a sliding window or across the entire trajectory, which we found to be a strong indicator of malformed or low-quality reasoning. Finally, based on internal audits of synthetically generated datasets, we observed that some teacher models occasionally produce reasoning traces and final responses that implicitly align with specific political entities or promote nationalistic narratives. To mitigate this, we apply targeted keyword- and regex-based filters and remove all trajectories matching such behavior.

Alongside the model, we release our final pre-training and post-training data, as outlined in this section. For ease of analysis, there is a sample set that is ungated. For all remaining code, math and multilingual data, gating and approval is required, and the dataset is permissively licensed for model training purposes.

<details>
  <summary>For Detailed Dataset Information: Click here!</summary>

#### **Base Pre-Training Corpus (Nemotron 3 Foundation)**

The foundation of the model is trained on the Nemotron 3 corpus, comprising the following datasets from the [Nemotron Pretraining Datasets collection](https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets):

| Dataset Collection | Token Counts | Description |
| :--- | :--- | :--- |
| **Nemotron-CC-v2** & **v2.1** | 9.1T | A massive collection of English web data filtered from Common Crawl, including 2.5T+ tokens of new organic, translated, and synthetically rephrased content. |
| **Nemotron-CC-Code-v1** | 427.9B | High-quality code tokens extracted from Common Crawl using the Lynx + LLM pipeline to preserve structure and equations. |
| **Nemotron-Pretraining-Code-v1** & **v2** & **v3** | 1.7T | Curated GitHub code references with multi-stage filtering, deduplication, and large-scale synthetic code data. |
| **Nemotron-CC-Math-v1** | 133.3B | High-quality math pre-training dataset preserving LaTeX formatting and mathematical structures. |
| **Nemotron-Pretraining-Specialized-v1** & **v1.1** & **v1.2** & **Nemotron-Pretraining-SFT-v1** | 660.0B | Synthetic datasets targeting specialized domains such as STEM reasoning and scientific coding. |
| **Nemotron-Pretraining-Legal-v1** | 4.3B | Synthetic datasets targeting the legal domain. |

### Public Datasets

| Dataset | Collection Period |
| :---- | :---- |
| [GSM8K](https://github.com/openai/grade-school-math) | 4/23/2025 |
| [CC-NEWS](https://commoncrawl.org/blog/news-dataset-available) | 4/23/2025 |
| [Common Crawl](https://commoncrawl.org/) | 4/23/2025 |
| [Wikimedia](https://dumps.wikimedia.org/) | 4/23/2025 |
| [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) | 4/23/2025 |
| [tigerbot-kaggle-leetcodesolutions-en-2k](https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k) | 4/23/2025 |
| [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | 4/23/2025 |
| [APIGen Function-Calling](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 4/23/2025 |
| [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | 4/23/2025 |
| [Open Textbook Library - CC BY-SA & GNU subset](https://open.umn.edu/opentextbooks/textbooks/) and [OpenStax - CC BY-SA subset](https://openstax.org/) | 4/23/2025 |
| [Advanced Reasoning Benchmark](https://github.com/TheDuckAI/arb), [tigerbot-kaggle-leetcodesolutions-en-2k](https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k), [PRM800K](https://github.com/openai/prm800k), and [SciBench](https://github.com/mandyyyyii/scibench) | 4/23/2025 |
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | 4/23/2025 |
| [Court Listener](https://www.courtlistener.com/help/api/bulk-data/) | Legacy Download |
| [peS2o](https://huggingface.co/datasets/allenai/peS2o) | Legacy Download |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | Legacy Download |
| [BioRxiv](https://www.biorxiv.org/tdm) | Legacy Download |
| [PMC Open Access Subset](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/) | Legacy Download |
| [OpenWebText2](https://openwebtext2.readthedocs.io/en/latest/) | Legacy Download |
| [Stack Exchange Data Dump](https://archive.org/details/stackexchange) | Legacy Download |
| [PubMed Abstracts](https://github.com/thoppe/The-Pile-PubMed) | Legacy Download |
| [NIH ExPorter](https://exporter.nih.gov/ExPORTER_Catalog.aspx) | Legacy Download |
| [arXiv](https://info.arxiv.org/help/bulk_data/index.html) | Legacy Download |
| [BigScience Workshop Datasets](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml#datasets) | Legacy Download |
| [Reddit Dataset](https://files.pushshift.io/reddit/) | Legacy Download |
| [SEC's Electronic Data Gathering, Analysis, and Retrieval (EDGAR)](https://www.sec.gov/search-filings) | Legacy Download |
| [Advanced Mathematical Problem Solving](https://github.com/hendrycks/math?tab=readme-ov-file) | Legacy Download |
| [MathPile](https://github.com/GAIR-NLP/MathPile/) | Legacy Download |
| [NuminaMath CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | Legacy Download |
| [PMC Article](https://pmc.ncbi.nlm.nih.gov/tools/textmining/) | Legacy Download |
| [FLAN](https://github.com/google-research/FLAN) | Legacy Download |
| [Advanced Reasoning Benchmark](https://github.com/TheDuckAI/arb) | Legacy Download |
| [SciBench](https://github.com/mandyyyyii/scibench) | Legacy Download |
| [WikiTableQuestions](https://huggingface.co/datasets/wikitablequestions) | Legacy Download |
| [FinQA](https://finqasite.github.io/) | Legacy Download |
| [Riddles](https://github.com/crawsome/riddles) | Legacy Download |
| [Problems in Elementary Mathematics for Home Study](https://archive.org/details/AntonovVygodskyNikitinSankinProblemsInElementaryMathematicsForHomeStudyMir1982) | Legacy Download |
| [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) | Legacy Download |
| [Cosmos QA](https://huggingface.co/datasets/allenai/cosmos_qa) | Legacy Download |
| [MCTest](https://huggingface.co/datasets/sagnikrayc/mctest) | Legacy Download |
| [AI2's Reasoning Challenge](https://huggingface.co/datasets/ai2_arc) | Legacy Download |
| [OpenBookQA](https://github.com/allenai/OpenBookQA) | Legacy Download |
| [MMLU Auxiliary Train](https://huggingface.co/datasets/cais/mmlu/viewer/all/auxiliary_train) | Legacy Download |
| [social-chemestry-101](https://huggingface.co/datasets/tasksource/social-chemestry-101) | Legacy Download |
| [Moral Stories](https://huggingface.co/datasets/demelin/moral_stories) | Legacy Download |
| [The Common Pile v0.1](https://huggingface.co/common-pile) | Legacy Download |
| [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | Legacy Download |
| [MegaMath](https://huggingface.co/datasets/LLM360/MegaMath) | Legacy Download |
| [MultiverseMathHard](https://huggingface.co/datasets/Nexusflow/MultiverseMathHard) | 10/2/2025 |
| [SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) | 10/2/2025 |
| [WorkBench](https://github.com/olly-styles/WorkBench/tree/main/data/raw) | 10/2/2025 |
| [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | 10/2/2025 |
| [OpenCodeReasoning-2](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2) | 10/2/2025 |
| [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) | 10/2/2025 |
| [opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2) | 10/2/2025 |
| [Big-Math-RL-Verified](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified) | 10/2/2025 |
| [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | 10/2/2025 |
| [simple-arithmetic-problems](https://huggingface.co/datasets/garrethlee/simple-arithmetic-problems) | 10/2/2025 |
| [arithmetic](https://huggingface.co/datasets/EleutherAI/arithmetic) | 10/2/2025 |
| [Skywork-OR1-RL-Data](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data) | 10/2/2025 |
| [FastChat](https://github.com/lm-sys/FastChat/blob/main/playground/data/dummy.json) | 10/2/2025 |
| [News Commentary](https://opus.nlpl.eu/News-Commentary.php) | 10/2/2025 |
| [Essential-Web](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0) | 10/2/2025 |
| [finepdfs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) | 10/2/2025 |
| [HotpotQA](https://huggingface.co/hotpot_qa/datasets) | 10/2/2025 |
| [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/) | 10/2/2025 |
| [NLTK Words Lists](https://www.nltk.org/nltk_data/) | 10/2/2025 |

### **Crawled and Scraped from Online Sources by NVIDIA**

The English Common Crawl data was downloaded from the Common Crawl Foundation (see their FAQ for details on their crawling) and includes the snapshots CC-MAIN-2013-20 through CC-MAIN-2025-13. The data was subsequently deduplicated and filtered in various ways described in the Nemotron-CC paper. Additionally, we extracted data for fifteen languages from the following three Common Crawl snapshots: CC-MAIN-2024-51, CC-MAIN-2025-08, CC-MAIN-2025-18. The fifteen languages included were Arabic, Chinese, Danish, Dutch, French, German, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Swedish, and Thai. As we did not have reliable multilingual model-based quality classifiers available, we applied just heuristic filtering instead—similar to what we did for lower quality English data in the Nemotron-CC pipeline, but selectively removing some filters for some languages that did not work well. Deduplication was done in the same way as for Nemotron-CC.

The GitHub Crawl was collected using the GitHub REST API and the Amazon S3 API. Each crawl was operated in accordance with the rate limits set by its respective source, either GitHub or S3. We collect raw source code and subsequently remove any having a license which does not exist in our permissive-license set.

| Dataset | Modality | Dataset Size | Collection Period | Collecting Organisation |
| :---- | :---- | :---- | :---- | :---- |
| English Common Crawl | Text | 3.36T | 4/8/2025 | NVIDIA Advanced Deep Learning Research |
| English Common Crawl 1.1 | Text | Not disclosed | 10/2/2025 | NVIDIA Advanced Deep Learning Research |
| Multilingual Common Crawl | Text | 812.7B | 5/1/2025 | NVIDIA Advanced Deep Learning Research |
| GitHub Crawl | Text | 747.4B | 4/29/2025 | NVIDIA Advanced Deep Learning Research |
| GitHub Crawl 1.1 | Text | 172.7B | 9/30/2025 | NVIDIA Advanced Deep Learning Research |

## Private Non-publicly Accessible Datasets of Third Parties

| Dataset | Model(s) used |
|---------|---------------|
| Global Regulation | Unknown |
| TAUS Translation Memory | Unknown |
| Scale HLE | Unknown |
| HackerRank Coding | Unknown |
| RL data for Search | Gemini 3; GPT-5 |
| Mercor SWE-AgentsV1 | Undisclosed |

## Private Non-publicly Accessible Datasets by NVIDIA

| Dataset | Model(s) used |
|---------|---------------|
| Simple Minesweeper | Undisclosed |
| Simple Sudoku | Undisclosed |
| Multitool Typewriter Hard | Undisclosed |
| Machine Translation of News Commentary and TAUS Translation Memory | Undisclosed |
| Machine Translation of STEM - | [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) |
| Competitive Coding RL data from Nemotron Cascade | Undisclosed |
| Long context RL | Undisclosed |
| Single-step SWE RL for patch generation | Undisclosed |
| OpenHands SWE | Undisclosed |

## NVIDIA-Sourced Synthetic Datasets (Pre-Training)

| Dataset | Modality | Dataset Size | Seed Dataset | Model(s) used for generation |
| :---- | :---- | :---- | :---- | :---- |
| Nemotron-Pretraining-Fact-Seeking | Text | 35.0B | [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) | [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) |
| Nemotron-Pretraining-Legal | Text | 4.3B | CommonPile (caselaw_access_project_filtered); California Code of Regulations; Judicial Ethics Opinions; [GLOBALCIT](https://globalcit.eu/); [CUAD](https://www.atticusprojectai.org/cuad); Nemotron Personas; [ToSDR Terms of Service Corpus](https://www.kaggle.com/datasets/sonu1607/tosdr-terms-of-service-corpus); CodeHima/TOS_Dataset; [ContractNLI](https://stanfordnlp.github.io/contract-nli/); CaseHOLD; Code of Federal Regulations; [Canadian Case Law](https://huggingface.co/datasets/a2aj/canadian-case-law) (subsets that allow commercial use) | [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) |
| Nemotron-Pretraining-Formal-Logic | Text | 128M | [Nemotron Personas](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) | [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) |
| Nemotron-Pretraining-Economics | Text | 73.4M | - | [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) |
| Nemotron-Pretraining-Multiple-Choice | Text | 1.6B | [MMLU Auxiliary Train](https://huggingface.co/datasets/cais/mmlu/viewer/all/auxiliary_train) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) |
| Nemotron-Pretraining-Code-Concepts | Text | 7.3B | - | [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b); [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Nemotron-Pretraining-Unconditional-Algorithmic | Text | 196.5M | - | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) |
| More Synthetic Tasks from DeepSeek-V3 and Qwen3-235B-A22B | Text | 1.1B | train splits of acp_bench; ai2_arc; babi; gsm8k; hendrycks_math; IFEval; MedText; mediqa_qa; mlqa; MMLU-Pro; mmlu-pro-plus; MMLU-ProX; nq_open; tinyGSM8k; truthful_qa; truthfulqa-multi; MATH-lighteval; mmlu; awesome-chatgpt-prompts; super_glue | [DeepSeek v3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) |
| Synthetic Tasks from DeepSeek-V3 and Qwen3-235B-A22B | Text | 6.7B | train splits of Into the Unknown; AI2 ARC (AI2 Reasoning Challenge); BLiMP (Benchmark of Linguistic Minimal Pairs); CommonSenseQA; GLUE; HeadQA; Hendrycks Ethics; Memo Trap; modus-tollens; NeQA; pattern-matching-suppression; mastermind_24_mcq_random; mastermind_24_mcq_close; quote-repetition; redefine-math; Repetitive Algebra; sig-figs; MMLU-Pro; MC-TACO; MedConceptsQA; MMLU_dataset; OpenbooksQA; PIQA (Physical Interaction Question Answering); SocialIQA; SuperGLUE; tinyAI2_arc; tinyMMLU; tinyWinogrande; TruthfulQA; WebQuestions; Winogrande; GPQA; MBPP | [DeepSeek v3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) |
| Synthetic Art of Problem Solving from DeepSeek-R1 | Text | 40B | [Art of Problem Solving](https://artofproblemsolving.com/company); [American Mathematics Competitions 8](https://artofproblemsolving.com/wiki/index.php/AMC_8_Problems_and_Solutions); [American Mathematics Competitions 10](https://artofproblemsolving.com/wiki/index.php/AMC_10_Problems_and_Solutions) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| Synthetic Moral Stories and Social Chemistry from Qwen3-235B-A22B-Thinking-2507 and Mixtral-8x22B-v0.1 | Text | 15.2M | [social-chemestry-101](https://huggingface.co/datasets/tasksource/social-chemestry-101); [Moral Stories](https://huggingface.co/datasets/demelin/moral_stories) | [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507); [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1) |
| Synthetic Moral Stories and Social Chemistry from Mixtral-8x22B-v0.1 | Text | 327M | [social-chemestry-101](https://huggingface.co/datasets/tasksource/social-chemestry-101); [Moral Stories](https://huggingface.co/datasets/demelin/moral_stories) | [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1) |
| Synthetic Social Sciences seeded with OpenStax from DeepSeek-V3, Mixtral-8x22B-v0.1, and Qwen2.5-72B | Text | 83.6M | [OpenStax - CC BY-SA subset](https://openstax.org/) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1); [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B) |
| Synthetic Health Sciences seeded with OpenStax from DeepSeek-V3, Mixtral-8x22B-v0.1, and Qwen2.5-72B | Text | 9.7M | [OpenStax - CC BY-SA subset](https://openstax.org/) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1); [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B) |
| Synthetic STEM seeded with OpenStax, Open Textbook Library, and GSM8K from DeepSeek-R1, DeepSeek-V3, DeepSeek-V3-0324, and Qwen2.5-72B | Text | 175M | [OpenStax - CC BY-SA subset](https://openstax.org/); [GSM8K](https://github.com/openai/grade-school-math); [Open Textbook Library - CC BY-SA & GNU subset](https://open.umn.edu/opentextbooks/textbooks/) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1), [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324); [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B) |
| [Nemotron-PrismMath](https://huggingface.co/datasets/nvidia/Nemotron-PrismMath) | Text | 4.6B | [Big-Math-RL-Verified](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified); [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | [Qwen2.5-0.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct); [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) |
| Synthetic Question Answering Data from Papers and Permissible Books from Qwen2.5-72B-Instruct | Text | 350M | [arXiv](https://info.arxiv.org/help/bulk_data/index.html); [National Institutes of Health ExPorter](https://www.nih.gov/); [BioRxiv](https://www.biorxiv.org/tdm); [PMC Article](https://pmc.ncbi.nlm.nih.gov/tools/textmining/); [USPTO Backgrounds](https://data.uspto.gov/apis/transition-guide/bdss#pats); [peS2o](https://huggingface.co/datasets/allenai/peS2o); Global Regulation; [CORE](https://core.ac.uk/documentation/dataset); [PG-19](https://github.com/google-deepmind/pg19); [DOAB CC BY & CC BY-SA subset](https://www.doabooks.org/en); [NDLTD](https://ndltd.org/thesis-resources/global-etd-search/) | [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Synthetic Rephrased [Math Data from Common Crawl](https://huggingface.co/datasets/nvidia/Nemotron-MIND) from phi-4 | Text | 73B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic Math Data from Common Crawl 4plus | Text | 52.3B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic Math Data from Common Crawl 3 | Text | 80.9B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic AGIEval seeded with AQUA-RAT, LogiQA, and AR-LSAT from DeepSeek-V3 and DeepSeek-V3-0324 | Text | 4.0B | [AQUA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat); [LogiQA](https://huggingface.co/datasets/lucasmccabe/logiqa); [AR-LSAT](https://github.com/zhongwanjun/AR-LSAT) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) |
| Synthetic AGIEval seeded with AQUA-RAT, LogiQA, and AR-LSAT from Qwen3-30B-A3B | Text | 4.2B | [AQUA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat); [LogiQA](https://huggingface.co/datasets/lucasmccabe/logiqa); [AR-LSAT](https://github.com/zhongwanjun/AR-LSAT) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Art of Problem Solving from Qwen2.5-32B-Instruct, Qwen2.5-Math-72B, Qwen2.5-Math-7B, and Qwen2.5-72B-Instruct | Text | Undisclosed | [Art of Problem Solving](https://artofproblemsolving.com/company); [American Mathematics Competitions 8](https://artofproblemsolving.com/wiki/index.php/AMC_8_Problems_and_Solutions); [American Mathematics Competitions 10](https://artofproblemsolving.com/wiki/index.php/AMC_10_Problems_and_Solutions); [GSM8K](https://github.com/openai/grade-school-math); [PRM800K](https://github.com/openai/prm800k) | [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct); [Qwen2.5-Math-72B](https://huggingface.co/Qwen/Qwen2.5-Math-72B); [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B); [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Synthetic MMLU Auxiliary Train from DeepSeek-R1 | Text | 0.5B | [MMLU Auxiliary Train](https://huggingface.co/datasets/cais/mmlu/viewer/all/auxiliary_train) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| Synthetic Long Context Continued Post-Training Data from Papers and Permissible Books from Qwen2.5-72B-Instruct | Text | Undisclosed | [arXiv](https://info.arxiv.org/help/bulk_data/index.html); [National Institutes of Health ExPorter](https://www.nih.gov/); [BioRxiv](https://www.biorxiv.org/tdm); [PMC Article](https://pmc.ncbi.nlm.nih.gov/tools/textmining/); [USPTO Backgrounds](https://data.uspto.gov/apis/transition-guide/bdss#pats); [peS2o](https://huggingface.co/datasets/allenai/peS2o); Global Regulation; [CORE](https://core.ac.uk/documentation/dataset); [PG-19](https://github.com/google-deepmind/pg19); [DOAB CC BY & CC BY-SA subset](https://www.doabooks.org/en); [NDLTD](https://ndltd.org/thesis-resources/global-etd-search/) | [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Synthetic Common Crawl from Qwen3-30B-A3B and Mistral-Nemo-12B-Instruct | Text | 415.8B | [Common Crawl](https://commoncrawl.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B); [Mistral-NeMo-12B-Instruct](https://huggingface.co/nvidia/Mistral-NeMo-12B-Instruct) |
| Synthetic Multilingual Data from Common Crawl from Qwen3-30B-A3B | Text | Undisclosed | [Common Crawl](https://commoncrawl.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Multilingual Data from Wikimedia from Qwen3-30B-A3B | Text | Undisclosed | [Wikimedia](https://dumps.wikimedia.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Math Data from Wikimedia from Nemotron-4-340B-Instruct | Text | Undisclosed | - | [Nemotron-4-340B-Instruct](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct) |
| Synthetic Common Crawl Code from phi-4 | Text | 427.9B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic Scientific Coding from Qwen3-235B-A22B | Text | 1.2B | [Wikimedia](https://dumps.wikimedia.org/) | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) |
| Tool Calling Data | Text | 26.2B | - | [Qwen3-235B-A22B-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507); [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Synthetic Essential-Web from QwQ-32B | Text | 28.1B | [Essential-Web](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0) | [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) |
| Translated Synthetic Crawl | Text | 389.9B | [Common Crawl](https://commoncrawl.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Translated Synthetic Wikipedia | Text | 7.9B | [Wikimedia](https://dumps.wikimedia.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Long Context from Qwen3-235B-A22B-Instruct-2507 | Text | Undisclosed | [CORE](https://core.ac.uk/documentation/dataset); [PG-19](https://github.com/google-deepmind/pg19); [DOAB CC BY & CC BY-SA subset](https://www.doabooks.org/en); [NDLTD](https://ndltd.org/thesis-resources/global-etd-search/) | [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) |
| Synthetic Search STEM OPENQ from DeepSeek-R1-0528 | Text | Undisclosed | - | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic MCQ from Qwen2.5-32B-Instruct and DeepSeek-R1-0528 | Text | Undisclosed | - | [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Offline Search MCQA HLE from DeepSeek-R1-0528 | Text | Undisclosed | - | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Offline Search MCQA GPQA from Qwen3-235B-A22B and DeepSeek-R1-0528 | Text | Undisclosed | - | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Human Preference from QwQ-32B, Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3-235B-A22B-Instruct-2507, Mistral-Small-3.1-24B-Instruct-2503, Mistral-Small-3.2-24B-Instruct-2506, MiniMax-M1-80k, MiniMax-M1-40k, Kimi-K2-Instruct, DeepSeek-V3-0324, DeepSeek-R1-0528 | Text | Undisclosed | - | [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B); [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B); [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507); [Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503); [Mistral-Small-3.2-24B-Instruct-2506](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506); [MiniMax-M1-80k](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k); [MiniMax-M1-40k](https://huggingface.co/MiniMaxAI/MiniMax-M1-40k); [Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct); [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic WildChat-1M and arena-human-preference-140k from DeepSeek-R1, gemma-2-2b-it, gemma-3-27b-it, gpt-oss-20b, gpt-oss-120b, Mistral-7B-Instruct-v0.3, Mixtral-8x22B-Instruct-v0.1, Nemotron-4-340B-Instruct, NVIDIA-Nemotron-Nano-9B-v2, Phi-4-mini-instruct, Phi-3-small-8k-instruct, Phi-3-medium-4k-instruct, Qwen3-235B-A22B, QwQ-32B | Text | Undisclosed | [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M); [arena-human-preference-140k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1); [gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it); [gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it); [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b); [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3); [Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1); [Nemotron-4-340B-Instruct](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct); [NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2); [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct); [Phi-3-small-8k-instruct](https://huggingface.co/microsoft/Phi-3-small-8k-instruct); [Phi-3-medium-4k-instruct](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct); [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) |
| Synthetic Code from Qwen3-32B | Text | Undisclosed | English Common Crawl; English Common Crawl 1.1 | [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) |
| Synthetic OpenCodeReasoning from DeepSeek-R1 | Text | Undisclosed | [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| Synthetic OpenCodeReasoning from DeepSeek-R1-0528 | Text | Undisclosed | [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic HackerRank Coding from DeepSeek-R1-0528 | Text | Undisclosed | HackerRank Coding Dataset | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic LIMO from DeepSeek-R1-0528 | Text | Undisclosed | [LIMO](https://huggingface.co/datasets/GAIR/LIMO) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic SCP from DeepSeek-R1-0528 | Text | Undisclosed | [SCP-116K](https://huggingface.co/datasets/EricLu/SCP-116K) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Stack Exchange from DeepSeek-R1-0528 | Text | Undisclosed | [Stack Exchange](https://archive.org/details/stackexchange) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Stack Exchange from gpt-oss-120b and Qwen2.5-32B-Instruct | Text | Undisclosed | [Stack Exchange](https://archive.org/details/stackexchange) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |
| Synthetic Stack Exchange from gpt-oss-120b | Text | Undisclosed | [Stack Exchange](https://archive.org/details/stackexchange) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Synthetic Art of Problem Solving from gpt-oss-120b and Qwen2.5-32B-Instruct | Text | Undisclosed | [Art of Problem Solving](https://artofproblemsolving.com/company); [American Mathematics Competitions 8](https://artofproblemsolving.com/wiki/index.php/AMC_8_Problems_and_Solutions); [American Mathematics Competitions 10](https://artofproblemsolving.com/wiki/index.php/AMC_10_Problems_and_Solutions) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |
| Synthetic Common Crawl from Qwen3-30B-A3B | Text | Undisclosed | [Common Crawl](https://commoncrawl.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Wikipedia from Qwen3-30B-A3B | Text | Undisclosed | [Wikimedia](https://dumps.wikimedia.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Essential-Web from Qwen3-30B-A3B and Qwen3-235B-A22B-Thinking-2507 | Text | Undisclosed | [Essential-Web](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B); [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) |
| Synthetic Essential-Web from gpt-oss-120b | Text | Undisclosed | [Essential-Web](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Synthetic Textbook Math from Qwen3-30B-A3B, Qwen3-235B-A22B, phi-4 | Text | Undisclosed | [Common Crawl](https://commoncrawl.org/); [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B); [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic Math and Code from DeepSeek-R1 and DeepSeek-R1-0528 | Text | Undisclosed | [Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K); [opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2); [TACO](https://huggingface.co/datasets/BAAI/TACO); [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning); [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning); [NuminaMath CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Math from gpt-oss-120b and Qwen2.5-32B-Instruct | Text | Undisclosed | - | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |
| Synthetic OpenMathReasoning from gpt-oss-120b and Qwen2.5-32B-Instruct | Text | Undisclosed | [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |
| Synthetic KernelBook from DeepSeek-R1-0528 | Text | Undisclosed | [KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Scale HLE from gpt-oss-120b | Text | Undisclosed | Scale HLE | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Synthetic CDQuestions from gpt-oss-120b | Text | Undisclosed | [CDQuestions](https://cdquestions.com/) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Synthetic GPQA from gpt-oss-120b and Qwen2.5-32B-Instruct | Text | Undisclosed | [Stack Exchange](https://archive.org/details/stackexchange) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |
| Synthetic Vedantu from gpt-oss-120b | Text | Undisclosed | [Vedantu](https://www.vedantu.com/) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Synthetic Search STEM MCQ from Qwen3-235B-A22B and DeepSeek-R1-0528 | Text | Undisclosed | - | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic OpenSTEM from Qwen2.5-32B-Instruct and DeepSeek-R1-0528 | Text | Undisclosed | - | [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic MCQ10 from DeepSeek-R1-0528 | Text | Undisclosed | - | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic MCQ4 from Qwen3-235B-A22B, DeepSeek-R1-0528, and Qwen3-235B-A22B-Instruct-2507 | Text | Undisclosed | - | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528); [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) |

## NVIDIA-Sourced Synthetic Datasets (Post-Training)

| Dataset | Modality | Dataset Size | Seed Dataset | Model(s) used for generation |
| :---- | :---- | :---- | :---- | :---- |
| Synthetic Competitive MATH Proofs from DeepSeek-V4-Pro | Text | Undisclosed | [AMC 8 Problems and Solutions, AMC 10 Problems and Solution, and AIME Problems and Solutions] | [deepseek-ai/DeepSeek-V4-Pro] |
| Synthetic Hermes Agent Reasoning Traces | Text | Undisclosed | [lambda/hermes-agent-reasoning-traces] | [hermes-agent-generator] |
| Synthetic Competitive Coding from DeepSeek-V4-Pro | Text | Undisclosed | [NVCompetitiveCodingV1] | [deepseek-ai/DeepSeek-V4-Pro] |
| Synthetic Competitive Science Reasoning from DeepSeek-V4-Pro | Text | Undisclosed | [AMC 8 Problems and Solutions, AMC 10 Problems and Solution, and AIME Problems and Solutions]; [EssentialAI/essential-web-v1.0]; [cdquestions.com]; [Pile-FreeLaw]; [Vedantu]; [askfilo]; [doubtnut]; [ICHO-IPH0 Dataset]; [LIMO dataset (Less is More for Reasoning)]; [AAPT]; [ChemData 700K]; [oMeBench]; [Flavor Analysis and Recognition Transformer]; [ChemCoTBench]; [Llama Nemotron Dataset] | [deepseek-ai/DeepSeek-V4-Pro] |
| Synthetic Competitive MATH CoT and TIR from Nemotron 5.5 | Text | Undisclosed | [Pile-FreeLaw]; [AMC 8 Problems and Solutions, AMC 10 Problems and Solution, and AIME Problems and Solutions] | [Nemotron 5.5] |
| Vendor Terminal Bench-like Tasks from Mercor | Text | Undisclosed | [Terminal bench like tasks curated by the vendor] | [Undisclosed - purchased dataset] |
| Turing Math Data Pack | Text | Undisclosed | [Turing Math Data Pack dataset] | [Undisclosed - purchased dataset] |
| Synthetic Holdout, Skywork, DAPO, and Turing Math from GPT-5.5 | Text | Undisclosed | [DocQA-RL-1.6K]; [DAPO-Math-17k] | [GPT-5.5] |
| Synthetic Long Context RL from QwenLong L1 and DocQA-RL-1.6K | Text | Undisclosed | [DocQA-RL-1.6K] | Undisclosed |
| Synthetic Competitive Coding Gym Tasks | Text | Undisclosed | [NVCompetitiveCodingV1.1] | Undisclosed |
| Synthetic Finance SEC Search Agent from GPT-OSS-120B and Qwen3 | Text | Undisclosed | [SEC filings from sec.gov] | [GPT-OSS-120B]; [Qwen3-235B-A22B-Instruct]; [Qwen3-4B-Instruct] |
| Synthetic Structured Outputs from Qwen3-30B-A3B-Instruct-2507, Qwen3-30B-A3B-Thinking-2507, Qwen3-235B-A22B-Instruct-2507, and Qwen3-235B-A22B-Thinking-2507 | Text | Undisclosed | [Nemotron-RL-agent-structured-outputs-v1] | [Qwen3-30B-A3B-Instruct-2507]; [Qwen3-235B-A22B-Instruct-2507] |
| Synthetic Long Context Equivalence Rule from Qwen3-235B-A22B-Thinking-2507 and DeepSeek-R1 | Text | Undisclosed | [Long-context SFT data] | [Qwen/Qwen3-235B-A22B-Thinking-2507]; [Deepseek-ai/DeepSeek-R1] |
| Synthetic Science RL Data Blend from Qwen2.5-32B | Text | Undisclosed | [doubtnut]; [Pile-FreeLaw]; [Llama Nemotron Dataset]; [askfilo]; [EssentialAI/essential-web-v1.0]; [Vedantu]; [auxiliary_train]; [cdquestions.com]; [AMC 8 Problems and Solutions, AMC 10 Problems and Solution, and AIME Problems and Solutions]; [AAPT]; [ICHO-IPH0 Dataset]; [LIMO dataset (Less is More for Reasoning)] | [Qwen2.5-32B] |
| Synthetic Abstention Data from Nemotron Super v3 | Text | Undisclosed | [Go abstention Dataset] | [nvidia/nvidia/nemotron-3-super-v3] |
| Synthetic Chemistry Data from Nemotron Super v3 | Text | Undisclosed | [ChemData 700K] | [nvidia/nvidia/nemotron-3-super-v3] |
| Synthetic Tool Call Schema for RL | Text | 469,983 | [UltraTool]; [ToolEyes]; [AutoTools]; [API-Bank]; [Nemotron-Personas-USA]; [Salesforce xLAM function-calling]; [Glaive function-calling-v2]; [Agent-Ark/Toucan-1.5M] | [DeepSeek-V3.2]; [GLM-4.6]; [gpt-oss-120b]; [Kimi-K2-Instruct] |
| Synthetic Freeform Text Formatting from GPT-OSS-120B | Text | Undisclosed | [In-house data] | [GPT OSS 120B - Apache 2.0] |
| Synthetic Citation Formatting from GPT-OSS-120B | Text | Undisclosed | [In-house data] | [GPT OSS 120B - Apache 2.0] |
| Droid Harness Pivot Vendor Data | Text | Undisclosed | [Droid Harness Pivot vendor data] | Undisclosed |
| Synthetic HotpotQA Training Data from Qwen3-235B | Text | Undisclosed | [HotpotQA] | [Qwen3-235B] |
| Synthetic Natural Language Math Proofs from Nemotron 5.5 | Text | Undisclosed | [AMC8, AMC10, and AIME problem sets hosted on Art of Problem Solving]; [Pile-StackExchange] | [Nemotron 5.5] |
| Synthetic Stack Overflow OpenQ | Text | Undisclosed | [Pile-FreeLaw] | Undisclosed |
| Chemistry Ether0 Vendor Data | Text | Undisclosed | [Chemistry ether0 vendor data] | Undisclosed |
| Synthetic Litmus-Bench Chemistry from ChEMBL | Text | Undisclosed | [ChEMBL]; [Nemo Gym RL dataset generated from ChEMBL with RDKit] | Undisclosed |
| Synthetic ZINC Chemistry from Nemotron Super v3 | Text | Undisclosed | [ZINC] | [Nemotron Super v3] |
| ARC-AGI Gym Environment | Text | Undisclosed | [ARC-AGI-2] | [ARC-AGI-2] |
| Synthetic Agentic Search Tool-Use from DeepSeek-V3.2 | Text | Undisclosed | [Mercor Data] | [DeepSeek-V3.2] |
| Synthetic Text-To-SQL | Text | 96,564 | [In-house Text-to-SQL data] | [gpt-oss-120b] |
| Dialog Memory Vendor Data | Text | Undisclosed | [Patronus external vendor agreement] | Undisclosed |
| Synthetic Indirect Prompt Injection from Nemotron Super v3 and Qwen3-Next-80B-A3B-Instruct | Text | Undisclosed | [In-house indirect prompt injection data] | [nvidia/nemotron-3-super-v3, qwen/qwen3-next-80b-a3b-instruct] |
| Synthetic Malicious Code and Agentic Security | Text | Undisclosed | [In-house malicious-code / agentic-security data] | Undisclosed |
| Synthetic Single-Step SWE Patch Selection | Text | Undisclosed | [SWE-Gym Dataset]; [SWE Bench Verified Benchmark] | [ground truth and task checks] |
| Synthetic Natural Language Math Final Answers from Nemotron 5.5 | Text | Undisclosed | [AMC8, AMC10, and AIME problem sets hosted on Art of Problem Solving]; [Pile-StackExchange] | [nemotron 5.5] |
| Synthetic Simple Math Prompts for Token Efficiency | Text | Undisclosed | [In-house simple math prompts] | Undisclosed |
| Synthetic Abstention Data from Nemotron Super v3 (CRAG) | Text | Undisclosed | [CRAG] | [nvidia/nvidia/nemotron-3-super-v3] |
| Synthetic Agentless SWE | Text | 242,536 | [SWE-Rebench-V2]; [SWEbench Training Set]; [R2E-Gym/R2E-Gym-Subset]; [SWE-Gym/SWE-Gym]; [SWE-Rebench] | [openai/gpt-oss-120b] |
| Synthetic Agentless SWE from DeepSeek-R1-0528 | Text | 209,976 | [SWE-Bench-Train]; [SWE-Fixer-Train]; [SWE-reBench]; [SWE-Smith] | [deepseek-ai/DeepSeek-R1-0528] |
| Synthetic Agentic CUDA Traces from GLM-4.7 | Text | 2,276 | [Internal CUDA task data] | [GLM-4.7] |
| Synthetic Math Proofs from DeepSeek-V3.2-Speciale | Text | 820,772 | [Nemotron-Math-Proofs-v1] | [SDG: DeepSeek-V3.2-Speciale]; [Filter: proof validation] |
| Synthetic Multilingual SFT from DeepSeek-V3 | Text | 1,245,284 | [Nano v3 SFT data] | [DeepSeek-V3] |
| Synthetic Agentic Code from gpt-oss-120b | Text | 109,086 | [NVAgenticCLIPrompts-v1]; [NVAgenticSkills-v1]; [NVAgenticCLIMultiTurnPrompts-v1] | [openai/gpt-oss-120b] |
| Synthetic Agentic CLI and Web Skills from gpt-oss-120b | Text | 27,418 | [NVAgenticCLIPrompts-v1]; [NVAgenticSkills-v1]; [NVAgenticCLIMultiTurnPrompts-v1]; [NVAgenticCLIPrompts-Web-v1] | [openai/gpt-oss-120b] |
| Synthetic Agentic Coding from gpt-oss-120b | Text | 160,531 | [NVAgenticCLIPrompts-v1]; [NVAgenticSkills-v1]; [NVAgenticCLIMultiTurnPrompts-v1]; [NVAgenticCLIPrompts-Web-v1] | [openai/gpt-oss-120b] |
| Synthetic OpenCode Agentic Tasks from gpt-oss-120b | Text | 614,773 | [NVAgenticCLIPrompts-v1]; [NVAgenticSkills-v1]; [NVAgenticCLIMultiTurnPrompts-v1]; [NVAgenticCLIPrompts-Web-v1] | [openai/gpt-oss-120b] |
| Synthetic SWE Unverified | Text | Undisclosed | [NVAgenticCLIPrompts-v1]; [NVAgenticSkills-v1]; [NVAgenticCLIMultiTurnPrompts-v1]; [NVAgenticCLIPrompts-Web-v1] | [gpt-oss-120b]; [Qwen/Qwen3-Coder-480B-A35B-Instruct]; [GLM-4.7-Flash] |
| Synthetic ARC-AGI Ultra Data | Text | 192,016 | [ARC-AGI-2]; [arc dataset collection] | [ARC-AGI-2] |
| Synthetic LiveCodeBench TIR from DeepSeek-R1-0528 | Text | 1,283,398 | [Nemotron-X training datasets] | [DeepSeek-R1-0528] |
| Synthetic Verilog and SystemVerilog Code from DeepSeek-R1-0528 and GPT-OSS-120B | Text | 1,233,247 | [Verilog/SystemVerilog seed code] | [SDR: DeepSeek R1 0528 and GPT-OSS-120B]; [Filtering: Claude 4 Sonnet] |
| Synthetic Aider Python Tasks from DeepSeek-R1-0528 | Text | 236,099 | [Exercism (GitHub Python)] | [Deepseek R1 0528] |
| Synthetic Chat Reasoning-Off Data from GLM-5 | Text | 646,738 | [lmarena-ai/repochat-arena-preference-4k user prompts] | [Multi-turn conversations generated by GLM-5 with best-of-4 selection via Qwen3-Nemotron-235B-A22B-GenRM] |
| Synthetic Chat Reasoning-On Data from GLM-5 | Text | 644,286 | [lmarena-ai/repochat-arena-preference-4k user prompts]; [lmarena-ai/arena-expert-5k user prompts]; [lmarena-ai/arena-human-preference-55k user prompts]; [lmarena-ai/arena-human-preference-100k user prompts]; [lmarena-ai/arena-human-preference-140k user prompts] | [Multi-turn conversations generated by GLM-5 with best-of-4 selection via Qwen3-Nemotron-235B-A22B-GenRM] |
| Synthetic Multilingual Safety from Riva-Translate-4B-Instruct-v1.1 | Text | 132,067 | [Safety SFT Data: Ultra] | [nvidia/Riva-Translate-4B-Instruct-v1.1] |
| Synthetic Science Reasoning Effort Medium | Text | 502,722 | [science-reasoning-effort-medium-v0] | Undisclosed |
| Synthetic Telecom Tool-Use Trajectories from gpt-oss-120b | Text | 12,455 | [Existing Tau2 telecom trajectories originally generated with DeepSeek V3.2] | [gpt-oss-120b] |
| Synthetic Terminal Bench Data from OpenReasoningv2 | Text | Undisclosed | [OpenCodeReasoningv2]; [OpenMathReasoning]; [nemo-swe-bench-repos]; [SWE-Rebench]; [SWE-Fixer-110K] | [OpenReasoningv2] |
| Synthetic Tulu Instruction Following from DeepSeek-R1-0528 | Text | 105,361 | [Nemotron-X training datasets] | [DeepSeek-R1-0528] |
| Synthetic Instruction Following from gpt-oss-120b | Text | 151,988 | [IFEval]; [IFEvalG] | [gpt-oss-120b] |
| Synthetic Instruction Following for RL | Text | Undisclosed | [WildChat-1M]; [LMSYS-340B-Eval Dataset]; [LMSYS-Chat-1M Prompts]; [IFEval]; [IFEvalG] | [Qwen/Qwen3-235B-A22B-Thinking-2507]; [gpt-oss-120b]; [Qwen3-235B-A22B-Instruct-2507] |
| Synthetic Identity Data from Qwen3-Next-80B-A3B-Instruct and Qwen3-235B-A22B-Instruct-2507 | Text | 25,992 | [Hand-written prompts] | [Qwen3-Next-80B-A3B-Instruct]; [Qwen3-235B-A22B-Instruct-2507] |
| Synthetic Terminus Ultra Agentic Reasoning Blend | Text | 96,881 | [ARC-AGI-2]; [OpenCodeReasoningv2]; [OpenMathReasoning]; [SWE-Fixer-110K]; [SWE-Rebench]; [SWE-Smith] | [DeepSeek-V3.2]; [Qwen3-235B-A22B-Thinking-2507]; [Ring-1T]; [Kimi-K2.5]; [GLM-4.7-FP8]; [Qwen3-Next-80B-A3B-Thinking]; [gpt-oss-120b]; [Ministral-3-14B-Reasoning-2512]; [LM-4.5-Air-FP8] |
| Synthetic STEM from Qwen3-235B-A22B-Thinking-2507 | Text | 1,174,694 | [IChO-IPhO-RL-v2]; [Physics-Big Dataset]; Scale HLE; [OpenMathReasoning]; [OpenCodeReasoning] | [Qwen3-235B-A22B-Thinking-2507] |
| Synthetic STEM from Qwen3-235B-A22B-Instruct-2507 and gpt-oss-120b | Text | Undisclosed | [arXiv]; [National Institutes of Health ExPorter]; [BioRxiv]; [PMC Article]; [USPTO Backgrounds]; [peS2o]; Global Regulation; [CORE]; [PG-19]; [DOAB CC BY & CC BY-SA subset]; [NDLTD] | [Qwen3-235B-A22B-Instruct-2507]; [gpt-oss-120b] |
| Translation Data from TAUS | Text | 1,618,055 | [TAUS proprietary dataset] | Undisclosed |
| Synthetic Art of Problem Solving and Stack Exchange from gpt-oss-120b, Qwen2.5-32B-Instruct, and Goedel-Prover-V2-32B | Text | 860,469 | [Nemotron-Math-Proofs-v1] | [Goedel-Prover-V2-32B] |
| Synthetic Art of Problem Solving and Stack Exchange from gpt-oss-120b | Text | 1,201,815 | [Upstream released math dataset]; [AoPS]; [StackOverflow / StackExchange] | [gpt-oss-120b] |
| Synthetic Multilingual Science and Code data from DeepSeek-R1, DeepSeek-R1-0528, Qwen2.5-32B-Instruct, and Qwen3-235B-A22B, translated with Qwen2.5-32B-Instruct and Qwen2.5-14B-Instruct | Text | Undisclosed | [Nano-V3 SFT Data (without tool call)] | [Qwen/Qwen2.5-14B-Instruct]; [Qwen/Qwen3-4B-Thinking-2507] |
| Synthetic Multilingual Science and Code data from DeepSeek-R1, DeepSeek-R1-0528, Qwen2.5-32B-Instruct, and Qwen3-235B-A22B, translated with Qwen2.5-32B-Instruct and Qwen2.5-14B-Instruct (Stack Exchange lineage) | Text | Undisclosed | [Stack Exchange]; [SCP-116K]; [LIMO]; [TACO]; Code Contest; Codeforces | [DeepSeek-R1]; [DeepSeek-R1-0528]; [Qwen2.5-32B-Instruct]; [Qwen3-235B-A22B] |
| Synthetic Search Graph Walk | Text | 6,977 | [Wikidata / Wikipedia KnowledgeBase] | [MiniMaxAI/MiniMax-M2] |
| Synthetic Agentic Diverse Domains | Text | 281,537 | [Handwritten prompts (synthetic; no external seed data used)] | [SDG model: deepseek-ai/DeepSeek-V3.2, deepseek-ai/DeepSeek-R1-0528, Qwen/Qwen3-235B-A22B-Thinking-2507, Qwen/Qwen3-32B]; [Filtering model: openai/gpt-oss-120b, Qwen/Qwen3-32B, Qwen/Qwen3-235B-A22B-Instruct-2507] |
| Synthetic Long Context from Qwen3-235B-A22B-Instruct-2507 | Text | Undisclosed | [Long-context SFT seed blend (pre-training blend + nano-v1 post-training data)]; [Long-context SFT data: lc_nothink 256k, MRCR 200k, RULER 256k]; [AALCR seed blend: SEC Filings, CC, Wikipedia, FinePDFs, ArXiv, Pile-NIH ExPorter, BioRxiv, PMC Article, USPTO Backgrounds, peS2o, Global Regulations, CORE, Gutenberg (PG-19), DOAB CC-BY, NDLTD, Amps, StackExchange, MathPile, Numinas] | [Qwen/Qwen3-235B-A22B-Thinking-2507]; [deepseek-ai/DeepSeek-R1]; [Qwen3-30B-A3B] |
| Synthetic Nemotron Math SFT from DeepSeek-V3.2-Speciale | Text | 1,900,553 | [Nemotron-Math-v2 (AOPS and StackExchange-math problems)] | [DeepSeek-V3.2-Speciale] |
| Synthetic Nemotron Math TIR from DeepSeek-V3.2 | Text | 1,789,258 | [Nemotron-Math-v2 (AOPS and StackExchange-math problems)] | [DeepSeek-V3.2] |
| Synthetic NemoCascade OCR Distillation from gpt-oss-120b | Text | 682,864 | [Nemotron-X training datasets] | [gpt-oss-120b] |
| Synthetic CUDA 100k | Text | 93,086 | [KernelBook]; [HuggingFace Transformers]; [FlashInfer] | [gpt-oss-120b]; [DeepSeek-R1-0528] |
| Synthetic Science MCQ and QA Diversity from GPT-OSS and Kimi-K2 | Text | 30,358 | [doubtnut]; [Pile-FreeLaw]; [Llama Nemotron Dataset]; [askfilo]; [EssentialAI/essential-web-v1.0]; [Vedantu]; [auxiliary_train]; [cdquestions.com]; [AMC 8 Problems and Solutions, AMC 10 Problems and Solution, and AIME Problems and Solutions]; [AAPT]; [ICHO-IPH0 Dataset]; [LIMO dataset (Less is More for Reasoning)] | [GPT-OSS]; [Kimi-K2] |
| Synthetic Science HLE with Python from GPT-OSS and Kimi-K2 | Text | 85,184 | [doubtnut]; [Pile-FreeLaw]; [Llama Nemotron Dataset]; [askfilo]; [EssentialAI/essential-web-v1.0]; [Vedantu]; [auxiliary_train]; [cdquestions.com]; [AMC 8 Problems and Solutions, AMC 10 Problems and Solution, and AIME Problems and Solutions]; [AAPT]; [ICHO-IPH0 Dataset]; [LIMO dataset (Less is More for Reasoning)] | [GPT-OSS]; [Kimi-K2] |
| Synthetic Science Search and Python from GPT-OSS and Kimi-K2 | Text | 6,179 | [doubtnut]; [Pile-FreeLaw]; [Llama Nemotron Dataset]; [askfilo]; [EssentialAI/essential-web-v1.0]; [Vedantu]; [auxiliary_train]; [cdquestions.com]; [AMC 8 Problems and Solutions, AMC 10 Problems and Solution, and AIME Problems and Solutions]; [AAPT]; [ICHO-IPH0 Dataset]; [LIMO dataset (Less is More for Reasoning)] | [GPT-OSS]; [Kimi-K2] |
| Synthetic Science Search from GPT-OSS and Kimi-K2 | Text | 32,554 | [doubtnut]; [Pile-FreeLaw]; [Llama Nemotron Dataset]; [askfilo]; [EssentialAI/essential-web-v1.0]; [Vedantu]; [auxiliary_train]; [cdquestions.com]; [AMC 8 Problems and Solutions, AMC 10 Problems and Solution, and AIME Problems and Solutions]; [AAPT]; [ICHO-IPH0 Dataset]; [LIMO dataset (Less is More for Reasoning)] | [GPT-OSS]; [Kimi-K2] |
| Synthetic Finance Reasoning from GPT-OSS-120B and Qwen3-235B-A22B-Instruct-2507 | Text | 326,700 | [SEC filings] | [GPT-OSS-120B, Qwen3-235B-A22B-Instruct-2507] |
| Synthetic Science Diversity MCQ from GPT-OSS and Kimi-K2 | Text | 532,942 | [doubtnut]; [Pile-FreeLaw]; [Llama Nemotron Dataset]; [askfilo]; [EssentialAI/essential-web-v1.0]; [Vedantu]; [auxiliary_train]; [cdquestions.com]; [AMC 8 Problems and Solutions, AMC 10 Problems and Solution, and AIME Problems and Solutions]; [AAPT]; [ICHO-IPH0 Dataset]; [LIMO dataset (Less is More for Reasoning)] | [GPT-OSS]; [Kimi-K2] |
| Synthetic Science Diversity OpenQ from GPT-OSS and Kimi-K2 | Text | 131,045 | [doubtnut]; [Pile-FreeLaw]; [Llama Nemotron Dataset]; [askfilo]; [EssentialAI/essential-web-v1.0]; [Vedantu]; [auxiliary_train]; [cdquestions.com]; [AMC 8 Problems and Solutions, AMC 10 Problems and Solution, and AIME Problems and Solutions]; [AAPT]; [ICHO-IPH0 Dataset]; [LIMO dataset (Less is More for Reasoning)] | [GPT-OSS]; [Kimi-K2] |
| Synthetic Science Reasoning No-Tool from GPT-OSS and Kimi-K2 | Text | 2,085,600 | [doubtnut]; [Pile-FreeLaw]; [Llama Nemotron Dataset]; [askfilo]; [EssentialAI/essential-web-v1.0]; [Vedantu]; [auxiliary_train]; [cdquestions.com]; [AMC 8 Problems and Solutions, AMC 10 Problems and Solution, and AIME Problems and Solutions]; [AAPT]; [ICHO-IPH0 Dataset]; [LIMO dataset (Less is More for Reasoning)] | [GPT-OSS]; [Kimi-K2] |
| Synthetic Text-To-SQL from gpt-oss-120b | Text | Undisclosed | [In-house Text-to-SQL data] | [gpt-oss-120b] |
| Synthetic Tool Call Schema for RL (extended) | Text | 707,967 | [UltraTool]; [ToolEyes]; [AutoTools]; [API-Bank]; [Nemotron-Personas-USA]; [Salesforce xLAM function-calling]; [Glaive function-calling-v2]; [Agent-Ark/Toucan-1.5M] | [DeepSeek-V3.2]; [GLM-4.6]; [gpt-oss-120b]; [Kimi-K2-Instruct] |
| Synthetic Safety from gemma-3-4b-it, Nemotron-Nano-9B-v2, and gpt-oss-120b | Text | 44,091 | [Safety SFT Data] | [google/gemma-3-4b-it]; [Nemotron-Nano-9B-v2]; [gpt-oss-120b] |
| Synthetic Safety from DeepSeek-R1-0528, gpt-oss-120b, DeepSeek-R1-Distill-Qwen-7B, and Mixtral-8x7B-v0.1 | Text | Undisclosed | [Nemotron Content Safety Dataset V2]; [Gretel Synthetic Safety Alignment Dataset]; [RedTeam-2K]; [Malicious Tasks]; [Nemotron-Personas-USA] | [DeepSeek-R1-0528]; [gpt-oss-120b]; [DeepSeek-R1-Distill-Qwen-7B]; [Qwen3-30B-A3B-Thinking-2507]; [Qwen3-235B-A22B-Instruct-2507]; [Mixtral-8x7B-v0.1] |
| Synthetic Tool Calling from Qwen3-235B-A22B-Thinking-2507 and Qwen3-Next-80B-A3B-Thinking | Text | Undisclosed | [ToolBench]; [glaive-function-calling-v2]; [APIGen Function-Calling]; [Nemotron-Personas-USA] | [Qwen3-235B-A22B-Thinking-2507]; [Qwen3-Next-80B-A3B-Thinking] |
| Synthetic Chat from gpt-oss-120b, Mixtral-8x22B-Instruct-v0.1, Qwen3-235B-A22B-Instruct-2507, and Qwen3-235B-A22B-Thinking-2507 | Text | Undisclosed | [C4]; [LMSYS-Chat-1M]; [ShareGPT]; [GSM8K]; [PRM800K]; [FinQA]; [WikiTableQuestions]; [Riddles]; [glaive-function-calling-v2]; [SciBench]; [tigerbot-kaggle-leetcodesolutions-en-2k]; [OpenBookQA]; [Advanced Reasoning Benchmark]; Software Heritage; [Khan Academy Math Keywords]; [WildChat-1M]; [Nemotron-Personas-USA] | [gpt-oss-120b]; [Mixtral-8x22B-Instruct-v0.1]; [Qwen3-235B-A22B-Instruct-2507]; [Qwen3-235B-A22B-Thinking-2507] |
| Synthetic Tool Use Interactive Agent from gpt-oss-120b, DeepSeek-R1-0528, Qwen3-32B, and Qwen3-235B-A22B-Thinking-2507 | Text | Undisclosed | NVIDIA Internal | [gpt-oss-120b]; [DeepSeek-R1-0528]; [Qwen3-32B]; [Qwen3-235B-A22B-Thinking-2507] |
| Synthetic DocFinQA and SWE-smith from Qwen3-Coder-480B-A35B-Instruct and Kimi-K2-Thinking | Text | Undisclosed | [DocFinQA]; [SWE-smith] | [Qwen3-Coder-480B-A35B-Instruct]; [Kimi-K2-Thinking] |
| Synthetic SWE-Gym from Qwen3-Coder-480B-A35B-Instruct | Text | Undisclosed | [SWE-Gym] | [Qwen3-Coder-480B-A35B-Instruct] |
| Synthetic SWE-Gym and R2E-Gym-Subset from Qwen3-Coder-480B-A35B-Instruct | Text | Undisclosed | [SWE-Gym]; [R2E-Gym-Subset] | [Qwen3-Coder-480B-A35B-Instruct] |
| Synthetic SWE-Gym and R2E-Gym-Subset from DeepSeek-R1-0528 | Text | Undisclosed | [SWE-Gym]; [R2E-Gym-Subset] | [DeepSeek-R1-0528] |
| Synthetic HelpSteer, LMSYS-Chat-1M, and Nemotron-Personas-USA from gpt-oss-120b, Qwen3-235B-A22B-Instruct-2507, and Qwen3-235B-A22B-Thinking-2507 | Text | Undisclosed | [HelpSteer2]; [HelpSteer3]; [LMSYS-Chat-1M]; [Nemotron-Personas-USA] | [gpt-oss-120b]; [Qwen3-235B-A22B-Instruct-2507]; [Qwen3-235B-A22B-Thinking-2507] |
| Synthetic Nemotron-Personas-USA from gpt-oss-120b and Qwen3-8B | Text | Undisclosed | [Nemotron-Personas-USA] | [gpt-oss-120b]; [Qwen3-8B] |
| Vendor Terminal Bench-like Tasks (Droid) | Text | Undisclosed | [Droid Harness Pivot vendor data] | Undisclosed |

## Language Distribution in Post-Training

For our post-training recipe, we focused on the following languages in addition to English: French, German, Italian, Japanese, Spanish, and Chinese. Those languages were represented in the form of multilingual reasoning and translation tasks.

</details>

## Testing Datasets:

**Data Collection Method by dataset** <br>
* Hybrid: Automated, Manually-Collected, Synthetic
**Labeling Method by dataset** <br>
* Hybrid: Automated, Manually-Labeled, Synthetic
**Properties:** This corpus comprises a mix of high-quality standard benchmarks and test suites for modern agentic AI. These benchmarks test model capabilities on tasks such as tool-calling and instruction following.

## Evaluation Datasets:

**Data Collection Method by dataset** <br>
* Hybrid: Automated, Manually-Collected, Synthetic
**Labeling Method by dataset** <br>
* Hybrid: Automated, Manually-Labeled, Synthetic
**Properties:** This corpus comprises a mix of high-quality standard benchmarks and test suites for modern agentic AI. These benchmarks test model capabilities on tasks such as tool-calling and instruction following.

## Inference

* **Acceleration Engine:** PyTorch
* **Test Hardware:**
    * NVIDIA Hopper
      - 1-8x H100
      - 1-8x H200
    * NVIDIA Blackwell
      - GB200
      - DGX Spark (GB10)
      - GeForce RTX 5090

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

We advise against circumvention of any provided safety guardrails contained in the Model without a substantially similar guardrail appropriate for your use case. For more details: [Safety](./safety.md) and [Explainability](./explainability.md) Subcards.

For more detailed information on ethical considerations for this model, please see the Model Card++ [Bias](./bias.md), and [Privacy](./privacy.md) Subcards.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).