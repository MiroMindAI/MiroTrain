# MiroTrain Usage Guide

## 1. New Features

### 1.1 Model Support

#### ✅ Native Qwen3 Series Models

MiroTrain supports multiple original Qwen3 models. In your training configuration (YAML format), specify models as follows:

```yaml
model:
  _component_: mirotrain.models.qwen3.qwen3_0_6b_base
```

Available models include:

```yaml
mirotrain.models.qwen3.qwen3_0_6b_base
mirotrain.models.qwen3.qwen3_0_6b_instruct
mirotrain.models.qwen3.qwen3_1_7b_base
mirotrain.models.qwen3.qwen3_1_7b_instruct
mirotrain.models.qwen3.qwen3_4b_base
mirotrain.models.qwen3.qwen3_4b_instruct
mirotrain.models.qwen3.qwen3_8b_base
mirotrain.models.qwen3.qwen3_8b_instruct
mirotrain.models.qwen3.qwen3_14b_base
mirotrain.models.qwen3.qwen3_14b_instruct
mirotrain.models.qwen3.qwen3_32b
```

#### ✅ Qwen3 MoE Models

MiroTrain supports two Qwen3 MoE models with expert parallelism. Configure as follows:

```yaml
model:
  _component_: mirotrain.models.qwen3_moe.qwen3_235b_a22b
  router_aux_loss_coef: 0.001
  use_grouped_gemm: True
```

**Important**: Update the `checkpointer` configuration for MoE models:

```yaml
checkpointer:
  model_type: QWEN3_MoE
```

By default, GroupedGEMM operators are used to improve training efficiency. Reference configuration: `mirotrain/experiments/qwen3_moe/235B-A22B_full_simpleqa_hf.yaml`

Available MoE models:

```yaml
mirotrain.models.qwen3_moe.qwen3_30b_a3b
mirotrain.models.qwen3_moe.qwen3_235b_a22b
```

#### ✅ Yarn RoPE Extended Qwen3 Models

MiroTrain supports Qwen3 models with Yarn RoPE extension for ultra-long context. Configuration example:

```yaml
model:
  _component_: mirotrain.models.qwen3.qwen3_8b_instruct_yarn
  max_position_embeddings: 163_840
  rope_scaling:
    rope_type: yarn
    factor: 4.0
    original_max_position_embeddings: 40_960
```

Available Yarn RoPE extended models:

```yaml
mirotrain.models.qwen3.qwen3_0_6b_base_yarn
mirotrain.models.qwen3.qwen3_0_6b_instruct_yarn
mirotrain.models.qwen3.qwen3_1_7b_base_yarn
mirotrain.models.qwen3.qwen3_1_7b_instruct_yarn
mirotrain.models.qwen3.qwen3_4b_base_yarn
mirotrain.models.qwen3.qwen3_4b_instruct_yarn
mirotrain.models.qwen3.qwen3_8b_base_yarn
mirotrain.models.qwen3.qwen3_8b_instruct_yarn
mirotrain.models.qwen3.qwen3_14b_base_yarn
mirotrain.models.qwen3.qwen3_14b_instruct_yarn
mirotrain.models.qwen3.qwen3_32b_yarn
```

> **Note**: Yarn RoPE extension supports ultra-long context, suitable for long-text training and inference scenarios.

#### ✅ Qwen3-Specific Tokenizer: `Qwen3TokenizerAuto`

- Wraps HuggingFace AutoTokenizer, specifically designed for Qwen3 SFT training, compatible with LLaMA-Factory data processing results
- Supports Qwen format conversation tags (e.g., `<|im_start|>assistant\n`) and adds instruction fine-tuning label masking logic
- Supports left/right truncation and maximum length control
- Compatible with TorchTune's message-based data structure

Usage example:

```yaml
tokenizer:
  _component_: mirotrain.models.qwen3.qwen3_tokenizer_auto
  path: /pfs/training-data/hf/models/Qwen/Qwen3-14B
```

### 1.2 Dataset Support

#### ✅ Open Deep Research Agent Trace Dataset

MiroTrain supports direct loading of open_deep_research_agent_trace format datasets for multi-turn conversation training. Our open_deep_research_agent_trace dataset is to be opensourceed.

#### ✅ Packed Dataset Support

Enable packed mode to concatenate multiple samples into a single long sequence for improved training efficiency:

```yaml
dataset:
  packed: True
  packed_seq_len: 16384  # Usually a multiple of max_seq_len
```

**Important Notes:**

- When `packed: True` is enabled, `batch_size` must be set to `1`
- `packed_seq_len` represents the concatenated length of a single batch (e.g., 8192 or 16384)
- Maximum length of each sample is still controlled by `max_seq_len`
- DataLoader output format is `(1, packed_seq_len)`, i.e., one long sequence containing multiple samples

#### ✅ Streaming Packed Dataset Support

Enable streaming pack mode to dynamically concatenate multiple samples into a single long sequence during training without preprocessing, significantly reducing task startup time:

```yaml
dataset:
  packed: False            # Disable offline packed mode
  streaming_pack: True     # Enable online streaming pack mode
  packed_seq_len: 16384    # Usually a multiple of max_seq_len
```

**Important Notes:**

- **`streaming_pack` and `packed` are mutually exclusive** and cannot be both `True`
- When `streaming_pack: True` is enabled, samples will be concatenated into fixed-length long sequences on-demand during training, saving preprocessing time
- Each batch output dimension is `(1, packed_seq_len)`, consistent with packed mode
- `batch_size` must be set to `1` as each batch is actually a single sequence concatenated from multiple samples
- Maximum length of each original sample is still controlled by `max_seq_len`
- Training startup will include a brief warm-up process to estimate total steps per epoch (affecting learning rate scheduling)

This approach is suitable for large-scale dataset training tasks, especially in scenarios where startup time is critical or data is continuously updated.

### 1.3 Parallelism & Modules

#### ✅ Ulysses Sequence Parallelism

Enable Ulysses sequence parallelism to support longer sequence training:

```yaml
ulysses_sequence_parallel_size: 1  # Can be set to 1, 2, 4, 8, etc. (recommended to be divisible by GPU count)
```

**Usage Recommendations:**
- Set the value to the number of processes participating in sequence splitting in current parallel training (usually an integer factor of `data_parallel_size`)

#### ✅ MoE Expert Parallelism

Enable MoE expert parallelism:

```yaml
moe_expert_parallel_size: 1  # Can be set to 1, 2, 4, 8, etc. (recommended not to exceed 8)
```

#### ✅ LigerLinearCrossEntropyLoss Fusion Operator

Support for [LigerLinearCrossEntropyLoss](https://github.com/linkedin/Liger-Kernel) fusion operator, significantly optimizing GPU memory usage in the final layer during long sequence training. In traditional implementations, the final layer's `CrossEntropyLoss` typically has \$O(B \times S \times V)\$ memory overhead (where \$B\$ is batch size, \$S\$ is sequence length, \$V\$ is vocab size). Using the optimized implementation from Liger-Kernel, this overhead can be reduced to \$O(B \times S \times H)\$ (where \$H\$ is hidden size), greatly alleviating the GPU memory bottleneck in long sequence training.

Enable as follows:

```yaml
loss:
  # Default is torchtune.modules.loss.LinearCrossEntropyLoss
  _component_: mirotrain.modules.loss.LigerLinearCrossEntropyLoss
```

#### ✅ Cosine Schedule with Warmup Learning Rate Configuration

Support for `get_cosine_schedule_with_warmup` learning rate configuration, using `warmup_ratio` like LLaMA-Factory.

Enable as follows:

```yaml
lr_scheduler:
  _component_: mirotrain.training.get_cosine_schedule_with_warmup
  warmup_ratio: 0.1
  num_cycles: 0.5
```

### 1.4 Fault Tolerance and Recovery

#### ✅ Auto Resume Support

When `auto_resume` is enabled, step-based checkpoints are saved at specified intervals, recording not only model weights but also optimizer state, dataloader state, and all other training states. If training is interrupted, no configuration changes are needed. Simply restart the task and it will automatically find the nearest step-based checkpoint to resume training.

Enable as follows:

```yaml
auto_resume: True
checkpoint_steps_for_auto_resume: 50  # Default is 50 if not specified
```

**Important Notes:**

- Current step-based checkpoints are designed for automatic recovery after training interruption and are not human-readable, nor can they be directly used for evaluation
- When saving the latest step-based checkpoint, outdated old step-based checkpoints are automatically found and deleted, so there's no need to worry about manually cleaning disk space due to high save frequency
- To reduce the overhead of frequent checkpoint saving, the internal implementation uses PyTorch Distributed Checkpointing (DCP) asynchronous saving feature, with a background thread responsible for writing checkpoints to disk, greatly reducing the time that blocks the training process. However, only when you see the log message "Step-based checkpoint is saved asynchronously to ... successfully" can you confirm that the previous step-based checkpoint has been successfully persisted to disk
- When conducting performance tests, avoid the few steps after asynchronous saving of step-based checkpoints has started but not completely finished, as the sub-thread responsible for asynchronous writing occasionally competes with the main thread for GIL, slightly affecting training performance during these steps

---

## 2. Quick Start

### 2.1 Download Model Weights

```bash
# Download Qwen3 model
tune download Qwen/Qwen3-32B \
  --output-dir /path/to/qwen3-32b \
  --hf-token <YOUR_HF_TOKEN>
```

### 2.2 SFT Training

```bash
cd recipes
torchrun \
  --nproc_per_node 8 \
  --nnodes 1 \
  sft_trainer.py \
  --config ./configs/qwen3/32B_full_sft.yaml
```

### 2.3 DPO Training

```bash
cd recipes
torchrun \
  --nproc_per_node 8 \
  --nnodes 1 \
  dpo_trainer.py \
  --config ./configs/qwen3/32B_full_dpo.yaml
```

## 3. Multi-Node Distributed Training

For large-scale training across multiple nodes:

```bash
torchrun \
  --nproc_per_node $WORKER_GPU \
  --master_addr $WORKER_0_HOST \
  --node_rank $ROLE_INDEX \
  --master_port $WORKER_0_PORT \
  --nnodes $WORKER_NUM \
  sft_trainer.py \
  --config config.yaml
```

The variables `$WORKER_GPU`, `$WORKER_0_HOST`, `$ROLE_INDEX`, `$WORKER_0_PORT`, and `$WORKER_NUM` are injected as environment variables by the Kubernetes platform. In future versions, we will also support compatibility with Slurm.
