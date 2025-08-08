# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""
This script is used to convert HF format weights, converting Qwen3 MoE weights that have been
trained using GroupedGEMM and expert parallelism back to the original HF weight format to
support subsequent training and inference.
This is the reverse operation of the tools/merge_moe_experts.py script.
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
    splited_state_dict: Dict[str, torch.Tensor],
    output_dir: str,
    num_layers: int,
    num_files: int = 1,
):
    """
    Save the weight dictionary after splitting experts to multiple .safetensors files and record the filename for each key.

    Args:
        splited_state_dict (Dict[str, torch.Tensor]): The weight dictionary after splitting experts.
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
            current_state_dict[key] = splited_state_dict[key]
            weight_map[key] = f"model-{file_idx + 1:05d}-of-{num_files:05d}.safetensors"
        if file_idx == num_files - 1:
            key = "model.norm.weight"
            current_state_dict[key] = splited_state_dict[key]
            weight_map[key] = f"model-{file_idx + 1:05d}-of-{num_files:05d}.safetensors"
            key = "lm_head.weight"
            current_state_dict[key] = splited_state_dict[key]
            weight_map[key] = f"model-{file_idx + 1:05d}-of-{num_files:05d}.safetensors"

        # Add current file's layer weights
        for layer_idx in range(start_layer, end_layer):
            for key in splited_state_dict:
                if f"model.layers.{layer_idx}." in key:
                    current_state_dict[key] = splited_state_dict[key]
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
                    for weight in splited_state_dict.values()
                ]
            )
        },
        "weight_map": weight_map,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def split_expert_weights(
    input_dir: str,
    output_dir: str,
    num_layers: int,
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_files: int,
) -> None:
    """
    Load all weight files from the specified directory and split expert weights.

    Args:
        input_dir (str): Directory containing weight files.
        output_dir (str): Directory to save split weights.
        num_layers (int): Number of model layers.
        num_experts (int): Total number of experts.
        hidden_dim (int): Hidden layer dimension.
        intermediate_dim (int): Intermediate layer dimension.
        num_files (int): Number of files to store weights after splitting experts.

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
    # Initialize an empty state_dict to store weights after expert splitting
    splited_state_dict = {}

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".safetensors"):
            # Construct complete file path
            input_path = os.path.join(input_dir, filename)

            # Load weight file
            state_dict = load_file(input_path)
            origin_state_dict.update(state_dict)

            for key, value in state_dict.items():
                # Add non-expert weights directly to splited_state_dict
                if "experts" not in key:
                    splited_state_dict[key] = value

    # Process expert weights for each layer
    for layer_idx in range(num_layers):
        original_key = f"model.layers.{layer_idx}.mlp.experts.ep_0.gate_proj.weight"
        gate_proj_weight = origin_state_dict[original_key]
        gate_proj_weight = gate_proj_weight.transpose(1, 2).clone()
        assert gate_proj_weight.size(0) == num_experts

        original_key = f"model.layers.{layer_idx}.mlp.experts.ep_0.up_proj.weight"
        up_proj_weight = origin_state_dict[original_key]
        up_proj_weight = up_proj_weight.transpose(1, 2).clone()
        assert up_proj_weight.size(0) == num_experts

        original_key = f"model.layers.{layer_idx}.mlp.experts.ep_0.down_proj.weight"
        down_proj_weight = origin_state_dict[original_key]
        down_proj_weight = down_proj_weight.transpose(1, 2).clone()
        assert down_proj_weight.size(0) == num_experts

        for expert_idx in range(num_experts):
            # gate_proj
            splited_key = (
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
            )
            splited_state_dict[splited_key] = gate_proj_weight[expert_idx].contiguous()

            # up_proj
            splited_key = (
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"
            )
            splited_state_dict[splited_key] = up_proj_weight[expert_idx].contiguous()

            # down_proj
            splited_key = (
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"
            )
            splited_state_dict[splited_key] = down_proj_weight[expert_idx].contiguous()

    # print(f"ht debug {splited_state_dict.keys()=}", flush=True)

    # Save merged weight files
    save_weights(
        splited_state_dict=splited_state_dict,
        output_dir=output_dir,
        num_layers=num_layers,
        num_files=num_files,
    )


# Example usage
if __name__ == "__main__":
    # ckpt path
    input_dir = (
        "/pfs/training-data/huangting/sd-torchtune/RUN/qwen3_moe_30B/full/epoch_0"
    )
    output_dir = "/pfs/training-data/huangting/hf/Qwen3-30B-A3B-splited-experts"
    # model type
    model_type = "Qwen3-30B-A3B"  # or "Qwen3-235B-A22B"
    model_config = MODEL_CONFIG[model_type]

    start_time = time.time()

    split_expert_weights(
        input_dir=input_dir,
        output_dir=output_dir,
        **model_config,
    )

    end_time = time.time()
    print(
        f"Converting weights finished in total {end_time - start_time} seconds",
        flush=True,
    )
