<h1 align="center">
<em>MiroTrain</em>: An Efficient and Algorithm-First Framework for Post-Training Large Agentic Models
</h1>

<p align="center">
<a href="https://huggingface.co/miromind-ai"><img src="https://img.shields.io/badge/-gery?style=social&label=%F0%9F%A4%97%20Huggingface" alt="HuggingFace" style="height: 20px;"></a>
<a href="https://x.com/miromind_ai"><img src="https://img.shields.io/badge/-grey?style=social&logo=x&label=MiroMindAI" alt="X" style="height: 20px;"></a>
<a href="https://www.xiaohongshu.com/user/profile/663098830000000003033edc"><img src="https://img.shields.io/badge/-grey?style=social&logo=red&label=RedNote" alt="小红书" style="height: 20px;"></a>
<a href="https://discord.gg/GPqEnkzQZd"><img src="https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord" alt="Discord" style="height: 20px;"></a>
<a href="https://github.com/user-attachments/assets/214ab129-a880-4882-8ae3-2702c0ed850b"><img src="https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat" alt="WeChat" style="height: 20px;"></a>
<a href="https://miromind.ai"><img src="https://img.shields.io/badge/-grey?style=social&logo=google-chrome&label=miromind.ai" alt="miromind.ai" style="height: 20px;"></a>
</p>

<p align="center">
<a href="#overview"><b>📖 Overview</b></a> | <a href="#installation"><b>🛠️ Installation</b></a> | <a href="#quick-start-on-single-node"><b>🚀 Quick Start</b></a> | <a href="docs/usage.md"><b>📚 Usage Guide</b></a> | <a href="#license"><b>📄 License</b></a>
</p>


## News

- **[2025/08]** Released MiroTrain-v0.1, supporting post-training for [MiroThinker](https://github.com/MiroMindAI/MiroThinker) using the [MiroVerse-v0.1](https://huggingface.co/datasets/miromind-ai/MiroVerse-v0.1) dataset.

## Overview 

**MiroTrain** is an efficient, algorithm-first framework for post-training large agentic models. Built on top of the open-source project [TorchTune](https://github.com/pytorch/torchtune), it delivers enhanced training recipes for **SFT** and **DPO**, supports post-training  of 32B-scale LLMs on agentic datasets on a single GPU node with 8×80GB GPUs, and enables seamless scaling of post-training workloads to **hundreds of GPUs**.



#### 🚀 Efficient

- **High-Performance Post-Training:** MiroTrain automatically leverages optimized operators such as [FlashAttention](https://github.com/Dao-AILab/flash-attention) and [Triton kernels](https://github.com/triton-lang/triton) to maximize training throughput.  It supports **streaming_pack**, which packs training samples on the fly without requiring dataset preprocessing.  

- **Best-in-Class Memory Efficiency:**  MiroTrain incorporates **Sequence Parallelism** and **CPU offloading**, enabling efficient post-training of models with large vocabulary sizes and long context lengths.

- **FSDPv2 Compatible:**  Fully compatible with [**FSDPv2**](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html), which adopts DTensor-based per-parameter sharding. 

#### ⚡ Algorithm-First

- **Customizable Post-Training Recipes:**  Provides easily hackable recipes for **SFT** and **DPO** workflows. The modular design makes it simple to adapt or extend recipes for new post-training methods.

- **Simple PyTorch-Based LLM Implementations:**  Clean and extensible model definitions allow for quick experimentation. Model architectures can be easily modified to integrate new features—such as support for **[Yarn-style RoPE scaling](https://arxiv.org/pdf/2309.00071)**.

- **HuggingFace Friendly:**  Fully compatible with HuggingFace datasets and model weights.  Fine-tuned checkpoints are saved in HuggingFace-compatible format and can be seamlessly loaded by [Transformers](https://github.com/huggingface/transformers), [vLLM](https://github.com/vllm-project/vllm), or [SGLang](https://github.com/sgl-project/sglang) for model serving.


For GRPO (Group Relative Policy Optimization) training, please refer to **[MiroRL](https://github.com/MiroMindAsia/mirorl)**: An MCP-first Reinforcement Learning Framework for Deep Research Agent


## Installation

MiroTrain is tested with the latest stable PyTorch releases (2.5, 2.6, and 2.7). We recommend using Python 3.10+ and CUDA 12.1+ for optimal performance.

#### 🐳 Docker Installation

For the fastest setup, we provide a pre-built Docker image with all dependencies pre-installed:

```bash
# Pull the Docker image
docker pull miromind/mirotrain:0.1.0-cuda12.6-pytorch2.6.0

# Run the container with GPU support
docker run --shm-size=8g --gpus all -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  miromind/mirotrain:0.1.0-cuda12.6-pytorch2.6.0
```

#### 🔧 Manual Installation

Create a Python Environment and install Pytorch based on your CUDA version. We recommend using conda to create a clean Python 3.10 environment. For other Pytorch and CUDA versions, please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).


```bash
conda create --name mirotrain-env python=3.10 -y
conda activate mirotrain-env
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install MiroTrain. Clone the repository and install MiroTrain:

```bash
git clone https://github.com/MiroMindAI/mirotrain
cd mirotrain
pip install ./torchtune
pip install .
```

## Quick Start on Single-Node

This guide demonstrates how to run MiroTrain on a single node with 8×80GB GPUs using Qwen3-32B as an example.

#### Download Model Weights

First, download the Qwen3-32B model weights from HuggingFace:

```bash
# Download Qwen3-32B model
tune download Qwen/Qwen3-32B \
  --output-dir /path/to/qwen3-32b \
  --hf-token <YOUR_HF_TOKEN>
```

#### SFT (Supervised Fine-Tuning)

Run supervised fine-tuning using the torchrun command:

```bash
cd recipes
torchrun \
  --nproc_per_node 8 \
  --nnodes 1 \
  sft_trainer.py \
  --config ./configs/qwen3/32B_full_sft.yaml
```

#### DPO (Direct Preference Optimization)

Run direct preference optimization using the torchrun command:

```bash
cd recipes
torchrun \
  --nproc_per_node 8 \
  --nnodes 1 \
  dpo_trainer.py \
  --config ./configs/qwen3/32B_full_dpo.yaml
```

## Usage Guide

[Usage Guide](docs/usage.md)


## Acknowledgements

- [TorchTune](https://github.com/pytorch/torchtune) for the excellent training framework and modular design
- [Liger-Kernel](https://github.com/fanshiqing/liger-kernel) for memory-efficient loss functions and training optimizations
- [Grouped GEMM](https://github.com/fanshiqing/grouped_gemm) for efficient grouped matrix operations in MoE model training
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) for high-performance attention implementations

## Citation

```bibtex
@misc{2025mirotrain,
    title={MiroTrain: An Efficient and Algorithm-First Framework for Post-Training Large Agentic Models},
    author={MiroMind AI Infra Team},
    howpublished = {\url{https://github.com/MiroMindAI/MiroTrain}},
    year={2025}
}
```

## License
This project is released under the [Apache License 2.0](LICENSE). Please also adhere to the Licenses of models and datasets being used.
