---
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3.6-35B-A3B
license: apache-2.0
library_name: Model Optimizer
tags:
- nvidia
- ModelOpt
- Qwen3.6
- quantized
- FP4
- fp4
---

# Model Overview

## Description:
The NVIDIA Qwen3.6-35B-A3B-NVFP4 model is the quantized version of Alibaba's Qwen3.6-35B-A3B model, which is an auto-regressive language model that uses an optimized transformer architecture. For more information, please check [here](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The NVIDIA Qwen3.6-35B-A3B-NVFP4 model is quantized with [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

This model is ready for commercial/non-commercial use.  <br>

## Third-Party Community Consideration
This model is not owned or developed by NVIDIA. This model has been developed and built to a third-party’s requirements for this application and use case; see link to Non-NVIDIA [(Qwen3.6-35B-A3B) Model Card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) from Alibaba.

## References
NVIDIA Model Optimizer: https://github.com/NVIDIA/Model-Optimizer

### License/Terms of Use:
[Apache license 2.0](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE)

### Deployment Geography:
Global <br>

### Use Case: <br>
Developers looking to take off-the-shelf, pre-quantized models for deployment in AI Agent systems, chatbots, RAG systems, and other AI-powered applications. <br>

### Release Date:  <br>
Hugging Face on 05/28/2026 via https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4 <br>

## Model Architecture:
**Architecture Type:** Transformers  <br>
**Network Architecture:** Mixture-of-Experts (MoE) with Hybrid Attention <br>
**Number of Model Parameters:** 35B in total and 3B activated <br>

## Input:
**Input Type(s):** Text, Image, Video <br>
**Input Format(s):** String, Red, Green, Blue (RGB), Video (MP4/WebM) <br>
**Input Parameters:** One-Dimensional (1D), Two-Dimensional (2D), Three-Dimensional (3D) <br>
**Other Properties Related to Input:** Context length up to 262K <br>

## Output:
**Output Type(s):** Text <br>
**Output Format:** String <br>
**Output Parameters:** One-Dimensional(1D): Sequences <br>
**Other Properties Related to Output:** None <br>

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>

## Software Integration:
**Supported Runtime Engine(s):** <br>
* vLLM <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Hopper, NVIDIA Blackwell <br>

**Preferred Operating System(s):** <br>
* Linux <br>

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):
The model version is NVFP4 1.0 version and is Quantized with nvidia-modelopt v0.44.0 <br>

## Training and Evaluation Datasets:

## Calibration Dataset:
**Link:** [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail), [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) <br>
**Data Collection Method by dataset:** Automated. <br>
**Labeling Method by dataset:** Automated. <br>
**Properties:** The cnn_dailymail dataset is an English-language dataset containing just over 300k unique news articles as written by journalists at CNN and the Daily Mail. The Nemotron-Post-Training-Dataset-v2 is a post-training dataset curated by NVIDIA containing multi-turn conversations across diverse topics. <br>

## Training Dataset:
**Data Modality:** Undisclosed <br>
**Data Collection Method by dataset:** Undisclosed <br>
**Labeling Method by dataset:** Undisclosed <br>
**Data Size:**  Undisclosed <br>
**Properties:** Undisclosed

## Evaluation Dataset:
**Datasets:** MMLU Pro, GPQA Diamond, τ²-Bench Telecom,  MMMU Pro, SciCode, AIME 2025, AA-LCR, IFBench <br>
**Data Collection Method by dataset:** Hybrid: Automated, Human <br>
**Labeling Method by dataset:** Hybrid: Human, Automated <br>
**Properties:** We evaluated the model on text-based reasoning and coding benchmarks: MMLU Pro is a multi-task language understanding benchmark with challenging multiple-choice questions across diverse academic domains; GPQA Diamond contains 448 graduate-level multiple-choice questions written by domain experts in biology, physics, and chemistry; τ²-Bench Telecom evaluates agentic tool-use and policy-adherence capabilities in dual-control telecom customer-service scenarios where the model interacts with a simulated user and external tools to resolve account issues; MMMU Pro is the more challenging version of the Massive Multi-discipline Multimodal Understanding benchmark, measuring college-level multimodal reasoning across diverse disciplines with expanded answer choices and a vision-only input setting; SciCode evaluates scientific coding capabilities; AIME 2025 contains problems from the American Invitational Mathematics Examination; AA-LCR (Artificial Analysis Long Context Recall) evaluates a model's ability to accurately retrieve and recall information from long input contexts; IFBench is a benchmark for evaluating instruction-following capabilities across diverse and structured task constraints. <br>

## Inference:
**Acceleration Engine:** vLLM <br>
**Test Hardware:** NVIDIA GB300 <br>

## Post Training Quantization
This model was obtained by quantizing the weights of Qwen3.6-35B-A3B to NVFP4 data type, ready for inference with vLLM. Only the weights and activations of the linear operators within transformer blocks in MoE are quantized. This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 3.06x.

## Usage

To serve this checkpoint with [vLLM](https://github.com/vllm-project/vllm), you can start the docker `vllm/vllm-openai:nightly` and run the sample command below:

```sh
vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --port 8000 --quantization modelopt --max-model-len 262144 --reasoning-parser qwen3
```

For NVIDIA DGX Spark, we recommend using this `vllm serve` command:

```sh
vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4  \
   --host 0.0.0.0 \
   --port 8000 \
   --tensor-parallel-size 1 \
   --trust-remote-code \
   --kv-cache-dtype fp8  \
   --attention-backend flashinfer  \
   --moe-backend marlin \
   --gpu-memory-utilization 0.4  \
   --max-model-len 262144 \
   --max-num-seqs 4 \
   --max-num-batched-tokens 8192  \
   --enable-chunked-prefill \
   --async-scheduling \
   --enable-prefix-caching \
   --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'  \
   --load-format fastsafetensors \
   --reasoning-parser qwen3 \
   --tool-call-parser qwen3_xml \
   --enable-auto-tool-choice
```


## Evaluation
The accuracy benchmark results are presented in the table below:
<table>
  <tr>
   <td><strong>Precision</strong>
   </td>
   <td><strong>MMLU Pro</strong>
   </td>
   <td><strong>GPQA Diamond</strong>
   </td>
   <td><strong>τ²-Bench Telecom</strong>
   </td>
   <td><strong>SciCode</strong>
   </td>
   <td><strong>AIME 2025</strong>
   </td>
   <td><strong>AA-LCR</strong>
   </td>
   <td><strong>IFBench</strong>
   </td>
   <td><strong>MMMU PRO</strong>
   </td>
  </tr>
  <tr>
   <td>BF16
   </td>
   <td><strong>85.6</strong>
   </td>
   <td><strong>84.9</strong>
   </td>
   <td><strong>95.5</strong>
   </td>
   <td><strong>40.8</strong>
   </td>
   <td><strong>89.2</strong>
   </td>
   <td><strong>62.0</strong>
   </td>
   <td><strong>62.3</strong>
   </td>
   <td><strong>74.1</strong>
   </td>
  </tr>
  <tr>
   <td>NVFP4
   </td>
   <td><strong>85.0</strong>
   </td>
   <td><strong>84.8</strong>
   </td>
   <td><strong>94.7</strong>
   </td>
   <td><strong>40.6</strong>
   </td>
   <td><strong>88.8</strong>
   </td>
   <td><strong>62.0</strong>
   </td>
   <td><strong>62.8</strong>
   </td>
   <td><strong>74.5</strong>
   </td>
  </tr>
</table>

> Baseline: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).
> SciCode with temperature=0.6, top_p=0.95, max num tokens 131072; the others with temperature=1.0, top_p=0.95, max num tokens 131072

## Model Limitations:
The base model was trained on data that contains toxic language and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. The model may generate answers that may be inaccurate, omit key information, or include irrelevant or redundant text producing socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive.

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail).
