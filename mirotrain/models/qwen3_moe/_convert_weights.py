# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtune.models.qwen2._convert_weights

import torch
from mirotrain.models.convert_weights import get_mapped_key

from mirotrain.modules.moe import (
    get_expert_parallel_rank,
    get_expert_parallel_world_size,
)

# State dict key mappings from HF's format to TorchTune's format
_FROM_HF = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attn.q_proj.weight",
    "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attn.q_proj.bias",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attn.k_proj.weight",
    "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attn.k_proj.bias",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attn.v_proj.weight",
    "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attn.v_proj.bias",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attn.output_proj.weight",
    "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attn.q_norm.scale",
    "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attn.k_norm.scale",
    "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    "model.layers.{}.input_layernorm.weight": "layers.{}.sa_norm.scale",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.mlp_norm.scale",
    "model.norm.weight": "norm.scale",
    "lm_head.weight": "output.weight",
}

# Mapping for SequentialMLP (non-GroupedGEMM) experts
_Sequential_MLP = {
    "model.layers.{}.mlp.gate.weight": "layers.{}.mlp.gate.weight",
    "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.mlp.experts.{}.w1.weight",
    "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.mlp.experts.{}.w3.weight",
    "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.mlp.experts.{}.w2.weight",
}

# Mapping for GroupedGEMM MLP experts with dynamic expert parallel keys
_GroupedGEMM_MLP = {
    "model.layers.{}.mlp.gate.weight": "layers.{}.mlp.gate.wg.weight",
    f"model.layers.{{}}.mlp.experts.ep_{get_expert_parallel_rank()}.gate_proj.weight": "layers.{}.mlp.experts.gate_proj",
    f"model.layers.{{}}.mlp.experts.ep_{get_expert_parallel_rank()}.up_proj.weight": "layers.{}.mlp.experts.up_proj",
    f"model.layers.{{}}.mlp.experts.ep_{get_expert_parallel_rank()}.down_proj.weight": "layers.{}.mlp.experts.down_proj",
}

# Constants for tied embeddings
QWEN3_TIED_KEY = "lm_head.weight"
QWEN3_TUNE_EMBEDDING_KEY = "tok_embeddings.weight"


def _merge_expert_weights_online(
    state_dict: dict[str, torch.Tensor],
    num_layers: int,
    num_experts: int,
    expert_parallel_size: int,
    hidden_dim: int,
    intermediate_dim: int,
    use_grouped_gemm: bool,
) -> dict[str, torch.Tensor]:
    """
    Merge expert weights online, converting original HF format expert weights to support expert parallelism.

    This function merges experts per rank into larger experts, adjusting weight shapes and key mappings
    accordingly. It supports both single and multi-expert parallel configurations.

    Args:
        state_dict (dict[str, torch.Tensor]): Original HF format weight dictionary
        num_layers (int): Number of layers in the model
        num_experts (int): Total number of experts
        expert_parallel_size (int): Expert parallel size (number of ranks)
        hidden_dim (int): Hidden dimension of the model
        intermediate_dim (int): Intermediate dimension for experts
        use_grouped_gemm (bool): Whether to use GroupedGEMM for experts

    Returns:
        dict[str, torch.Tensor]: Merged weight dictionary with expert parallel format
    """
    if not use_grouped_gemm:
        # Return original weights if not using GroupedGEMM
        return state_dict

    merged_state_dict = {}
    experts_per_rank = num_experts // expert_parallel_size
    current_ep_rank = get_expert_parallel_rank()

    # Process non-expert weights (copy as-is)
    for key, value in state_dict.items():
        if "experts" not in key:
            merged_state_dict[key] = value

    # Process expert weights for each layer
    for layer_idx in range(num_layers):
        # Calculate expert IDs for current rank
        expert_ids = list(
            range(
                current_ep_rank * experts_per_rank,
                (current_ep_rank + 1) * experts_per_rank,
            )
        )

        # Get dtype from first weight for consistency
        first_weight_key = (
            f"model.layers.{layer_idx}.mlp.experts.{expert_ids[0]}.gate_proj.weight"
        )
        dtype = (
            state_dict[first_weight_key].dtype
            if first_weight_key in state_dict
            else torch.bfloat16
        )

        # Merge gate_proj weights
        gate_proj_merged_weight = torch.zeros(
            (experts_per_rank, hidden_dim, intermediate_dim), dtype=dtype
        )

        for i, expert_id in enumerate(expert_ids):
            original_key = (
                f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj.weight"
            )
            if original_key in state_dict:
                weight = state_dict[original_key]
                weight = weight.transpose(0, 1)  # Transpose to match target format
                gate_proj_merged_weight[i] = weight

        merged_key = f"model.layers.{layer_idx}.mlp.experts.ep_{current_ep_rank}.gate_proj.weight"
        merged_state_dict[merged_key] = gate_proj_merged_weight

        # Merge up_proj weights
        up_proj_merged_weight = torch.zeros(
            (experts_per_rank, hidden_dim, intermediate_dim), dtype=dtype
        )

        for i, expert_id in enumerate(expert_ids):
            original_key = (
                f"model.layers.{layer_idx}.mlp.experts.{expert_id}.up_proj.weight"
            )
            if original_key in state_dict:
                weight = state_dict[original_key]
                weight = weight.transpose(0, 1)
                up_proj_merged_weight[i] = weight

        merged_key = (
            f"model.layers.{layer_idx}.mlp.experts.ep_{current_ep_rank}.up_proj.weight"
        )
        merged_state_dict[merged_key] = up_proj_merged_weight

        # Merge down_proj weights
        down_proj_merged_weight = torch.zeros(
            (experts_per_rank, intermediate_dim, hidden_dim), dtype=dtype
        )

        for i, expert_id in enumerate(expert_ids):
            original_key = (
                f"model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.weight"
            )
            if original_key in state_dict:
                weight = state_dict[original_key]
                weight = weight.transpose(0, 1)
                down_proj_merged_weight[i] = weight

        merged_key = f"model.layers.{layer_idx}.mlp.experts.ep_{current_ep_rank}.down_proj.weight"
        merged_state_dict[merged_key] = down_proj_merged_weight

    return merged_state_dict


def qwen3_moe_hf_to_tune(
    state_dict: dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
    head_dim: int = None,
    tie_word_embeddings: bool = False,
    num_layers: int = None,
    num_experts: int = None,
    intermediate_dim: int = None,
    use_grouped_gemm: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Convert a state dict from HF's format to TorchTune's format for Qwen3 MoE models.

    This function supports online expert merging, eliminating the need for offline preprocessing
    with merge_moe_experts.py. It can handle both SequentialMLP and GroupedGEMM expert configurations.

    Args:
        state_dict (dict[str, torch.Tensor]): State dict in HF's format
        num_heads (int): Number of attention heads in the model
        num_kv_heads (int): Number of key/value heads in the model
        dim (int): Hidden dimension of the model
        head_dim (int, optional): Dimension of each attention head (calculated as dim // num_heads if None)
        tie_word_embeddings (bool): Whether input and output embeddings are tied
        num_layers (int, optional): Number of layers in the model (required for expert merging)
        num_experts (int, optional): Total number of experts (required for expert merging)
        intermediate_dim (int, optional): Intermediate dimension for experts (required for expert merging)
        use_grouped_gemm (bool): Whether to use GroupedGEMM for expert computation

    Returns:
        dict[str, torch.Tensor]: State dict in TorchTune's format
    """
    global _FROM_HF
    _from_hf_local = _FROM_HF.copy()

    # Get expert parallel size from distributed environment
    expert_parallel_size = get_expert_parallel_world_size()

    # Perform online expert merging if GroupedGEMM is enabled
    if use_grouped_gemm:
        state_dict = _merge_expert_weights_online(
            state_dict=state_dict,
            num_layers=num_layers,
            num_experts=num_experts,
            expert_parallel_size=expert_parallel_size,
            hidden_dim=dim,
            intermediate_dim=intermediate_dim,
            use_grouped_gemm=use_grouped_gemm,
        )

    # Update mapping dictionary based on expert configuration
    if use_grouped_gemm:
        _from_hf_local.update(_GroupedGEMM_MLP)
    else:
        _from_hf_local.update(_Sequential_MLP)

    # Convert state dict
    converted_state_dict = {}

    for key, value in state_dict.items():
        # Skip output projection weights if embeddings are tied
        if tie_word_embeddings and QWEN3_TIED_KEY in key:
            continue

        # Skip rotary embeddings
        if "rotary_emb.inv_freq" in key:
            continue

        # Skip experts not in current expert parallel rank
        if (
            use_grouped_gemm
            and "experts" in key
            and f"ep_{get_expert_parallel_rank()}." not in key
        ):
            continue

        # Use standard mapping for non-expert weights
        new_key = get_mapped_key(key, _from_hf_local)
        converted_state_dict[new_key] = value

    return converted_state_dict


def _split_expert_weights_online(
    state_dict: dict[str, torch.Tensor],
    num_layers: int,
    num_experts: int,
    expert_parallel_size: int,
    hidden_dim: int,
    intermediate_dim: int,
    use_grouped_gemm: bool,
) -> dict[str, torch.Tensor]:
    """
    Split expert weights online, converting GroupedGEMM format expert weights back to original HF format.

    This function splits merged experts back into individual experts, adjusting weight shapes
    and key mappings accordingly. It's the reverse operation of _merge_expert_weights_online.

    Note: In checkpoint saving, all expert weights have already been gathered across expert parallel ranks,
    so we process all experts, not just the current rank's experts.

    Args:
        state_dict (dict[str, torch.Tensor]): GroupedGEMM format weight dictionary (with all experts gathered)
        num_layers (int): Number of layers in the model
        num_experts (int): Total number of experts
        expert_parallel_size (int): Expert parallel size (number of ranks)
        hidden_dim (int): Hidden dimension of the model
        intermediate_dim (int): Intermediate dimension for experts
        use_grouped_gemm (bool): Whether to use GroupedGEMM for experts

    Returns:
        dict[str, torch.Tensor]: Split weight dictionary with original HF format
    """
    if not use_grouped_gemm:
        # Return original weights if not using GroupedGEMM
        return state_dict

    split_state_dict = {}

    # Process non-expert weights (copy as-is)
    for key, value in state_dict.items():
        if "experts" not in key:
            split_state_dict[key] = value

    # Process expert weights for each layer
    for layer_idx in range(num_layers):
        # Check if we have gathered expert weights (all experts in one tensor)
        # The gathered weights should have shape [num_experts, ...]
        gathered_gate_key = f"layers.{layer_idx}.mlp.experts.gate_proj"
        gathered_up_key = f"layers.{layer_idx}.mlp.experts.up_proj"
        gathered_down_key = f"layers.{layer_idx}.mlp.experts.down_proj"

        if gathered_gate_key in state_dict:
            # Get gathered weights and transpose back to original format
            gate_proj_gathered = state_dict[gathered_gate_key].transpose(1, 2).clone()
            up_proj_gathered = state_dict[gathered_up_key].transpose(1, 2).clone()
            down_proj_gathered = state_dict[gathered_down_key].transpose(1, 2).clone()

            # Verify we have all experts
            assert (
                gate_proj_gathered.size(0) == num_experts
            ), f"Expected {num_experts} experts, got {gate_proj_gathered.size(0)}"

            # Split gathered weights into individual experts
            for expert_id in range(num_experts):
                # gate_proj
                split_key = (
                    f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj.weight"
                )
                split_state_dict[split_key] = gate_proj_gathered[expert_id].contiguous()

                # up_proj
                split_key = (
                    f"model.layers.{layer_idx}.mlp.experts.{expert_id}.up_proj.weight"
                )
                split_state_dict[split_key] = up_proj_gathered[expert_id].contiguous()

                # down_proj
                split_key = (
                    f"model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.weight"
                )
                split_state_dict[split_key] = down_proj_gathered[expert_id].contiguous()

    return split_state_dict


def qwen3_moe_tune_to_hf(
    state_dict: dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
    head_dim: int = None,
    tie_word_embeddings: bool = False,
    num_layers: int = None,
    num_experts: int = None,
    intermediate_dim: int = None,
    use_grouped_gemm: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Convert a state dict from TorchTune's format to HF's format for Qwen3 MoE models.

    This function supports online expert splitting, eliminating the need for offline preprocessing
    with split_moe_experts.py. It can handle both SequentialMLP and GroupedGEMM expert configurations.

    Args:
        state_dict (dict[str, torch.Tensor]): State dict in TorchTune's format
        num_heads (int): Number of attention heads in the model
        num_kv_heads (int): Number of key/value heads in the model
        dim (int): Hidden dimension of the model
        head_dim (int, optional): Dimension of each attention head (calculated as dim // num_heads if None)
        tie_word_embeddings (bool): Whether input and output embeddings are tied
        num_layers (int, optional): Number of layers in the model (required for expert splitting)
        num_experts (int, optional): Total number of experts (required for expert splitting)
        intermediate_dim (int, optional): Intermediate dimension for experts (required for expert splitting)
        use_grouped_gemm (bool): Whether to use GroupedGEMM for expert computation

    Returns:
        dict[str, torch.Tensor]: State dict in HF's format
    """
    global _FROM_HF

    _from_hf_local = _FROM_HF.copy()

    # Get expert parallel size from distributed environment
    expert_parallel_size = get_expert_parallel_world_size()

    # Perform online expert splitting if GroupedGEMM is enabled
    if use_grouped_gemm:
        state_dict = _split_expert_weights_online(
            state_dict=state_dict,
            num_layers=num_layers,
            num_experts=num_experts,
            expert_parallel_size=expert_parallel_size,
            hidden_dim=dim,
            intermediate_dim=intermediate_dim,
            use_grouped_gemm=use_grouped_gemm,
        )

    _from_hf_local.update(_Sequential_MLP)

    if use_grouped_gemm:
        _from_hf_local.update(
            {"model.layers.{}.mlp.gate.weight": "layers.{}.mlp.gate.wg.weight"}
        )

    # Convert state dict
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _from_hf_local.items()}

    for key, value in state_dict.items():
        # If the key is already in HF format (from _split_expert_weights_online), use it directly
        if use_grouped_gemm and "mlp.experts." in key:
            converted_state_dict[key] = value
        else:
            # Otherwise, apply the mapping
            new_key = get_mapped_key(key, inverted_mapping_dict)
            converted_state_dict[new_key] = value

        # Handle tied embeddings
        if QWEN3_TUNE_EMBEDDING_KEY in key and tie_word_embeddings:
            # Copy input embeddings to output embeddings when tied
            converted_state_dict["lm_head.weight"] = value.detach().clone()

    return converted_state_dict
