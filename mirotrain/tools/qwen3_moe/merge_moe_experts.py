# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""
This script is used to convert HF format weights to support Qwen3 MoE's GroupedGEMM and expert parallelism

The general operation is: merge the expert weight lists from the original HF format weights in order,
combining every experts_per_rank experts into one large expert
where experts_per_rank = num_experts // expert_parallel_size

Assuming num_experts=128, expert_parallel_size=8, experts_per_rank=16

Taking ep_rank=0 as an example, the weight key mapping relationship is:
["model.layers.{layer_idx}.mlp.experts.0.gate_proj.weight",
 "model.layers.{layer_idx}.mlp.experts.1.gate_proj.weight",
 "model.layers.{layer_idx}.mlp.experts.2.gate_proj.weight",
 ......
 "model.layers.{layer_idx}.mlp.experts.13.gate_proj.weight",
 "model.layers.{layer_idx}.mlp.experts.14.gate_proj.weight",
 "model.layers.{layer_idx}.mlp.experts.15.gate_proj.weight"] -> "model.layers.{layer_idx}.mlp.experts.ep_0.gate_proj.weight"

The weight shape mapping relationship is:
Taking gate_proj linear as an example, before merging each linear's shape is intermediate_dim * hidden_dim (transposed)
After merging, the large expert's gate_proj weight shape is experts_per_rank * hidden_dim * intermediate_dim
"""


import json
import os
import time
from typing import Dict

import torch
from safetensors.torch import load_file, save_file


MODEL_CONFIG = {
    "Qwen3-30B-A3B": {
        "num_layers": 48,
        "num_experts": 128,
        "hidden_dim": 2048,
        "intermediate_dim": 768,
        "num_files": 16,
    },
    "Qwen3-235B-A22B": {
        "num_layers": 94,
        "num_experts": 128,
        "hidden_dim": 4096,
        "intermediate_dim": 1536,
        "num_files": 94,
    },
}


def save_weights(
    merged_state_dict: Dict[str, torch.Tensor],
    output_dir: str,
    num_layers: int,
    num_files: int = 1,
):
    """
    Save the merged weight dictionary to multiple .safetensors files and record the filename for each key.

    Args:
        merged_state_dict (Dict[str, torch.Tensor]): The merged weight dictionary.
        output_dir (str): Directory to save weight files.
        num_layers (int): Number of model layers.
        num_files (int, optional): Number of files to save to, defaults to 1.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Calculate number of layers per file
    layers_per_file = num_layers // num_files

    # Initialize mapping to record keys to filenames
    weight_map = {}

    # Save weights to multiple files by layer segments
    for file_idx in range(num_files):
        start_layer = file_idx * layers_per_file
        end_layer = min((file_idx + 1) * layers_per_file, num_layers)

        # Create a sub-dictionary to store current file's weights
        current_state_dict = {}

        # Special handling for non-layer weights
        if file_idx == 0:
            key = "model.embed_tokens.weight"
            current_state_dict[key] = merged_state_dict[key]
            weight_map[key] = f"model-{file_idx + 1:05d}-of-{num_files:05d}.safetensors"
        if file_idx == num_files - 1:
            key = "model.norm.weight"
            current_state_dict[key] = merged_state_dict[key]
            weight_map[key] = f"model-{file_idx + 1:05d}-of-{num_files:05d}.safetensors"
            key = "lm_head.weight"
            current_state_dict[key] = merged_state_dict[key]
            weight_map[key] = f"model-{file_idx + 1:05d}-of-{num_files:05d}.safetensors"

        # Add current file's layer weights
        for layer_idx in range(start_layer, end_layer):
            for key in merged_state_dict:
                if f"model.layers.{layer_idx}." in key:
                    current_state_dict[key] = merged_state_dict[key]
                    weight_map[
                        key
                    ] = f"model-{file_idx + 1:05d}-of-{num_files:05d}.safetensors"

        # Construct output file path and save current file's weights
        output_path = os.path.join(
            output_dir, f"model-{file_idx + 1:05d}-of-{num_files:05d}.safetensors"
        )
        save_file(current_state_dict, output_path)
        print(
            f"Processed layers from {start_layer} to {end_layer} and saved to: {output_path}",
            flush=True,
        )

    # Save weight mapping to JSON file
    metadata = {
        "metadata": {
            "total_size": sum(
                [
                    weight.numel() * weight.element_size()
                    for weight in merged_state_dict.values()
                ]
            )
        },
        "weight_map": weight_map,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def merge_expert_weights(
    input_dir: str,
    output_dir: str,
    num_layers: int,
    num_experts: int,
    expert_parallel_size: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_files: int,
) -> None:
    """
    Load all weight files from the specified directory, merge expert weights for each layer, and save to a new directory.

    Args:
        input_dir (str): Directory containing weight files.
        output_dir (str): Directory to save merged weights.
        num_layers (int): Number of model layers.
        num_experts (int): Total number of experts.
        expert_parallel_size (int): Expert parallelism size.
        hidden_dim (int): Hidden layer dimension.
        intermediate_dim (int): Intermediate layer dimension.
        num_files (int): Number of files to store weights after merging experts.

    Raises:
        FileNotFoundError: If the input directory does not exist.
    """
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    # Ensure output directory exists, create if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize an empty state_dict to store all original weights
    origin_state_dict = {}
    # Initialize an empty state_dict to store weights after expert merging
    merged_state_dict = {}

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".safetensors"):
            # Construct complete file path
            input_path = os.path.join(input_dir, filename)

            # Load weight file
            state_dict = load_file(input_path)
            origin_state_dict.update(state_dict)

            for key, value in state_dict.items():
                # Add non-expert weights directly to merged_state_dict
                if "experts" not in key:
                    merged_state_dict[key] = value

    # Process expert weights for each layer
    experts_per_rank = num_experts // expert_parallel_size

    for layer_idx in range(num_layers):
        for rank in range(expert_parallel_size):
            expert_ids = list(
                range(rank * experts_per_rank, (rank + 1) * experts_per_rank)
            )

            # gate_proj
            gate_proj_merged_weight = torch.zeros(
                (experts_per_rank, hidden_dim, intermediate_dim), dtype=torch.bfloat16
            )

            for i, expert_id in enumerate(expert_ids):
                original_key = (
                    f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj.weight"
                )
                weight = origin_state_dict[original_key]
                # print(f"ht debug {weight.shape=}", flush=True)
                weight = weight.transpose(0, 1)
                gate_proj_merged_weight[i] = weight

            merged_key = (
                f"model.layers.{layer_idx}.mlp.experts.ep_{rank}.gate_proj.weight"
            )
            merged_state_dict[merged_key] = gate_proj_merged_weight

            # up_proj
            up_proj_merged_weight = torch.zeros(
                (experts_per_rank, hidden_dim, intermediate_dim), dtype=torch.bfloat16
            )

            for i, expert_id in enumerate(expert_ids):
                original_key = (
                    f"model.layers.{layer_idx}.mlp.experts.{expert_id}.up_proj.weight"
                )
                weight = origin_state_dict[original_key]
                weight = weight.transpose(0, 1)
                up_proj_merged_weight[i] = weight

            merged_key = (
                f"model.layers.{layer_idx}.mlp.experts.ep_{rank}.up_proj.weight"
            )
            merged_state_dict[merged_key] = up_proj_merged_weight

            # down_proj
            down_proj_merged_weight = torch.zeros(
                (experts_per_rank, intermediate_dim, hidden_dim), dtype=torch.bfloat16
            )

            for i, expert_id in enumerate(expert_ids):
                original_key = (
                    f"model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.weight"
                )
                weight = origin_state_dict[original_key]
                weight = weight.transpose(0, 1)
                down_proj_merged_weight[i] = weight

            merged_key = (
                f"model.layers.{layer_idx}.mlp.experts.ep_{rank}.down_proj.weight"
            )
            merged_state_dict[merged_key] = down_proj_merged_weight

    # print(f"ht debug {merged_state_dict.keys()=}", flush=True)

    # Save merged weight files
    save_weights(
        merged_state_dict=merged_state_dict,
        output_dir=output_dir,
        num_layers=num_layers,
        num_files=num_files,
    )


# Example usage
if __name__ == "__main__":
    # ckpt path
    input_dir = "/pfs/training-data/hf/models/Qwen/Qwen3-235B-A22B"
    output_dir = "/pfs/training-data/huangting/hf/Qwen3-235B-A22B-ep1"
    # model type
    model_type = "Qwen3-235B-A22B"  # or "Qwen3-30B-A3B"
    model_config = MODEL_CONFIG[model_type]
    # target expert parallel size
    expert_parallel_size = 1

    start_time = time.time()

    merge_expert_weights(
        input_dir=input_dir,
        output_dir=output_dir,
        expert_parallel_size=expert_parallel_size,
        **model_config,
    )

    end_time = time.time()
    print(
        f"Converting weights finished in total {end_time - start_time} seconds",
        flush=True,
    )
