# pydoclint: skip-file

# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""
The file has been adapted from the following files:
https://github.com/InternLM/InternEvo/blob/develop/internlm/model/moe/dropless_layer.py
"""

import math
from typing import Callable, Dict, Optional, Tuple

# To enable gemm permute optimizations on GPU:
#   python3 -m pip install --verbose git+https://github.com/fanshiqing/grouped_gemm@v1.1.4
try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .expert_parallel import (
    get_expert_parallel_group,
    get_expert_parallel_rank,
    get_expert_parallel_world_size,
)
from .experts import GroupedGEMMExperts
from .utils import all_to_all, gather_along_first_dim_expert_parallel, Silu

uniform_map: Dict[torch.device, Callable] = {}


def multiplicative_jitter(
    x: torch.Tensor, device: torch.device, epsilon: float = 1e-2
) -> torch.Tensor:
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.

    Args:
        x (torch.Tensor): Input tensor.
        device (torch.device): Device on which to create the random tensor.
        epsilon (float, optional): The range of the random number. Defaults to 1e-2.

    Returns:
        torch.Tensor: The jittered tensor.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 - epsilon, device=device),
            high=torch.tensor(1.0 + epsilon, device=device),
        ).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)


class TopKGate(Module):
    """
    The TopKGate routing module.

    Args:
        model_dim (int): Size of model embedding dimension.
        num_experts (int): Number of experts in model.
        topk (int, optional): Number of top experts to select. Defaults to 1.
        noisy_gate_policy (Optional[str], optional): Policy for noisy gating. Defaults to None.
        scoring_func (str, optional): Scoring function to use. Defaults to "softmax".
        jitter_eps (float, optional): Epsilon value for jittering. Defaults to 1e-2.
        aux_loss_by_layer (bool, optional): Whether to compute auxiliary loss by layer. Defaults to True.
    """

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        topk: int = 1,
        noisy_gate_policy: Optional[str] = None,
        scoring_func: str = "softmax",
        jitter_eps: float = 1e-2,
        aux_loss_by_layer: bool = True,
    ) -> None:
        super().__init__()

        # Deepspeed's mechanisms, always use fp32
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.k = topk
        self.jitter_eps = jitter_eps

        self.noisy_gate_policy = noisy_gate_policy
        self.aux_loss_by_layer = aux_loss_by_layer
        self.scoring_func = scoring_func

        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the linear layer.
        """
        from torch.nn import init

        init.kaiming_uniform_(self.wg.weight, a=math.sqrt(5))

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the TopKGate module.
        """
        # Input jittering
        if self.noisy_gate_policy == "Jitter" and self.training:
            inputs = multiplicative_jitter(
                inputs, epsilon=self.jitter_eps, device=inputs.device
            )
        logits = self.wg(inputs)
        if self.scoring_func == "sigmoid":
            gates = logits.sigmoid()
        elif self.scoring_func == "softmax":
            gates = F.softmax(logits, dim=1)
        else:
            raise NotImplementedError(
                f"Unsupported scoring function for MoE gating: {self.scoring_func}"
            )

        if not self.aux_loss_by_layer:
            return gates, logits
        return gates, None


# pydoclint: disable
class DroplessMoELayer(Module):
    """MoELayer module which implements MixtureOfExperts as described in Gshard_."""

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_experts: int,
        top_k: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.device] = None,
        activation_type: str = "swiglu",
        noisy_gate_policy: str = None,
        deterministic_mode: bool = False,
        moe_jitter_eps: float = 1e-2,
        normalize_expert_weights: bool = True,
        routed_scaling_factor: float = 1.0,
        scoring_func: str = "softmax",
        aux_loss_by_layer: bool = False,
        output_router_logits: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.ep_group = get_expert_parallel_group()
        self.ep_size = get_expert_parallel_world_size()
        self.ep_rank = get_expert_parallel_rank()
        self.num_local_experts = self.num_experts // self.ep_size
        assert noisy_gate_policy is None or noisy_gate_policy in [
            "None",
            "Jitter",
            "RSample",
        ], (
            "Unsupported noisy_gate_policy: " + noisy_gate_policy
        )
        assert (
            num_experts % self.ep_size == 0
        ), f"Number of experts ({num_experts}) should be divisible by expert parallel size ({self.ep_size})"

        self.gate = TopKGate(
            hidden_dim,
            self.num_experts,
            top_k,
            noisy_gate_policy,
            scoring_func,
            moe_jitter_eps,
            aux_loss_by_layer,
        )

        self.experts = GroupedGEMMExperts(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_local_experts=self.num_local_experts,
            activation=Silu,
        )

        local_expert_indices_offset = self.ep_rank * self.num_local_experts
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert (
            len(self.local_expert_indices) > 0
        ), "Expected at least one local expert index"
        self.topk = top_k
        self.deterministic_mode = deterministic_mode
        self.normalize_expert_weights = normalize_expert_weights
        self.routed_scaling_factor = routed_scaling_factor
        self.aux_loss_by_layer = aux_loss_by_layer
        self.output_router_logits = output_router_logits

        # AlltoAll token dispatch policy
        self.token_permutation_func = self.token_permutation_by_alltoall
        self.token_unpermutation_func = self.token_unpermutation_by_alltoall
        self.input_splits = None
        self.output_splits = None
        self.num_global_tokens_per_local_expert_cpu = None
        self.hidden_shape = None
        # A cuda stream synchronization is needed due to no blocking sync between host and device
        self.device_sync_point = "no_sync"
        input_chunk_idxs = torch.arange(
            self.num_experts, device=torch.cuda.current_device()
        )
        # [num_local_experts, ep_size]. Sort the input chunks by local experts.
        self.sort_input_by_local_experts = (
            input_chunk_idxs.reshape(-1, self.num_local_experts).T.ravel().tolist()
        )
        # [ep_size, num_local_experts]. Restore the output chunks by local experts.
        self.restore_output_by_local_experts = (
            input_chunk_idxs.reshape(self.num_local_experts, -1).T.ravel().tolist()
        )

    def forward(self, *inputs: Tensor) -> Tensor:
        self.hidden_shape = inputs[0].shape

        d_model = inputs[0].shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_inputs = inputs[0].reshape(-1, d_model)

        gates, logits = self.gate(reshaped_inputs)
        (
            expert_weights,
            indices,
            tokens_per_expert_before_capacity,
        ) = self.topk_softmax_with_capacity(gates)
        if self.aux_loss_by_layer:
            self.l_aux = self.load_balancing_loss(
                tokens_per_expert_before_capacity, gates
            )
        else:
            self.l_aux = None

        (dispatched_input, tokens_per_expert) = self.token_permutation_func(
            reshaped_inputs, expert_weights, indices, tokens_per_expert_before_capacity
        )

        expert_output = self.experts(
            dispatched_input, tokens_per_expert=tokens_per_expert
        )

        output = self.token_unpermutation_func(expert_output, expert_weights)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)

        # Note: 1. we need to relase self.l_aux and its compute graph; 2. we need self.l_aux to simplify code
        #   so we first use self.l_aux and then reset it.
        l_aux = self.l_aux
        self.l_aux = None
        if self.aux_loss_by_layer:
            return output, l_aux
        if self.output_router_logits:
            return output, logits
        return output

    def topk_softmax_with_capacity(self, gates):
        expert_weights, indices = torch.topk(gates, self.topk, dim=1)

        if self.normalize_expert_weights:
            expert_weights /= expert_weights.sum(dim=-1, keepdim=True)

        # histc(.) can be faster the bincount(.), but will cause non-deterministic behavior
        if self.deterministic_mode:
            num_local_tokens_per_expert = torch.bincount(
                indices.view(-1), minlength=self.num_experts
            )
        else:
            num_local_tokens_per_expert = torch.histc(
                indices, bins=self.num_experts, min=0, max=self.num_experts
            )

        expert_weights = expert_weights * self.routed_scaling_factor

        # shape: [num_token, topk]
        return expert_weights, indices, num_local_tokens_per_expert

    def preprocess(
        self,
        indices: torch.Tensor,
        expert_weight: torch.Tensor,
        tokens_per_expert_before_capacity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Preprocess token indices for AlltoAll communication and token permutation. This method computes
        the number of tokens assigned to each expert based on the input indices.
        It also initializes the necessary data structures for AlltoAll communication, such as input
        and output splits, and the mapping between global tokens and local experts.

        Args:
            indices (torch.Tensor): Tensor of indices mapping tokens to experts.
            expert_weight (torch.Tensor): Tensor of expert weights.
            tokens_per_expert_before_capacity (torch.Tensor): Tensor containing the number of tokens
                assigned to local expert before capacity.

        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """
        num_local_tokens_per_expert = tokens_per_expert_before_capacity

        if self.ep_size > 1 or self.num_local_experts > 1:
            # wait for input_splits and output_splits, or num_global_tokens_per_local_expert_cpu sync
            self.device_sync_point = "before_ep_alltoall"
        else:
            # wait for tokens_per_expert sync
            self.device_sync_point = "before_premute_finish"

        if self.ep_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall-v.
            # ===================================================
            self.input_splits = (
                num_local_tokens_per_expert.reshape(
                    self.ep_size, self.num_local_experts
                )
                .sum(axis=1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )
            # avoid allgather stuck in some case
            torch.cuda.current_stream().synchronize()
            num_global_tokens_per_expert = gather_along_first_dim_expert_parallel(
                num_local_tokens_per_expert
            ).reshape(self.ep_size, self.num_experts)
            num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                :, self.local_expert_indices
            ]

            self.output_splits = (
                num_global_tokens_per_local_expert.sum(axis=-1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )
            num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)
            # ===================================================
            # num_global_tokens_per_expert: [ep_size, num_experts]
            # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
            # num_tokens_per_local_expert: [num_local_experts]
            # ===================================================
        else:
            num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                self.num_experts
            )
            num_tokens_per_local_expert = num_local_tokens_per_expert

        num_tokens_per_local_expert = num_tokens_per_local_expert.to(
            torch.device("cpu"), non_blocking=True
        )

        if self.num_local_experts > 1 and self.ep_size > 1:
            # expert_ids_per_ep_rank = torch.remainder(
            #     torch.arange(self.num_experts, dtype=torch.int32, device=indices.device),
            #     self.num_local_experts,  # mpu.experts_per_rank(self.args),
            # )
            self.num_global_tokens_per_local_expert_cpu = (
                num_global_tokens_per_local_expert.view(-1, self.num_local_experts).to(
                    torch.device("cpu"), non_blocking=True
                )
            )

        return num_tokens_per_local_expert

    def sort_chunks_by_idxs(
        self, inputs: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor
    ):
        """Split and sort the input tensor based on the split_sizes and sorted indices."""
        inputs = torch.split(inputs, split_sizes.tolist(), dim=0)
        output = torch.cat([inputs[i] for i in sorted_idxs], dim=0)
        return output

    def token_permutation_by_alltoall(
        self,
        reshaped_inputs: torch.Tensor,
        expert_weights: torch.Tensor,
        indices: torch.Tensor,
        tokens_per_expert_before_capacity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to local experts using AlltoAll communication.

        Args:
            reshaped_inputs (torch.Tensor): Input token embeddings.
            expert_weights (torch.Tensor): Expert weights of tokens assigned to experts.
            indices (torch.Tensor): Indices of tokens assigned to experts.
            tokens_per_expert_before_capacity (torch.Tensor): Tensor containing the number of tokens
                assigned to local expert before capacity.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.

        Raises:
            ImportError: If grouped_gemm is not available.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        assert expert_weights.dim() == 2, "Expected 2D tensor for expert weights"
        assert indices.dim() == 2, "Expected 2D tensor for indices"
        tokens_per_expert = self.preprocess(
            indices, expert_weights, tokens_per_expert_before_capacity
        )

        # Permutation 1: input to AlltoAll input
        self.hiddden_shape_before_permute = reshaped_inputs.shape
        if self.device_sync_point == "before_ep_alltoall":
            torch.cuda.current_stream().synchronize()

        if grouped_gemm is None:
            raise ImportError(
                "Grouped GEMM is required for MoE operations. "
                "Please install it with: pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4"
            )
        (
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
        ) = grouped_gemm.ops.permute(reshaped_inputs, indices.to(torch.int32), None)

        # Perform expert parallel AlltoAll communication
        global_input_tokens, _ = all_to_all(
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
            self.ep_group,
        )

        # Permutation 2: Sort alltoall output by local experts when num_local_experts > 1.
        if self.num_local_experts > 1 and self.ep_size > 1:
            global_input_tokens = self.sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert_cpu.ravel(),
                self.sort_input_by_local_experts,
            )

        if self.device_sync_point == "before_premute_finish":
            torch.cuda.current_stream().synchronize()

        return global_input_tokens, tokens_per_expert

    def token_unpermutation_by_alltoall(
        self,
        hidden_states: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reverse the token permutation to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.
            expert_weights (torch.Tensor): Expert weights of tokens assigned to experts.

        Returns:
            Unpermuted token embeddings in the original order.

        Raises:
            ImportError: If grouped_gemm is not available.
        """

        # Unpermutation 2: expert output to AlltoAll input
        if self.num_local_experts > 1 and self.ep_size > 1:
            hidden_states = self.sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert_cpu.T.ravel(),
                self.restore_output_by_local_experts,
            )

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]

        permutated_local_input_tokens, _ = all_to_all(
            hidden_states, self.input_splits, self.output_splits, self.ep_group
        )

        # Unpermutation 1: AlltoAll output to output
        if grouped_gemm is None:
            raise ImportError(
                "Grouped GEMM is required for MoE operations. "
                "Please install it with: pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4"
            )
        output = grouped_gemm.ops.unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            expert_weights.to(torch.float32),
        )

        return output

    def load_balancing_loss(self, num_local_tokens_per_expert, gates):
        """Calculate the load balancing loss contribution."""
        assert len(gates.size()) == 2
        tokens, num_experts = gates.size()
        assert num_experts == self.num_experts
        assert len(num_local_tokens_per_expert.size()) == 1
        (num_experts,) = num_local_tokens_per_expert.size()
        assert num_experts == self.num_experts
        scale = self.num_experts / (tokens * self.topk)
        return scale * torch.dot(
            num_local_tokens_per_expert.to(gates.dtype), gates.mean(dim=0)
        )
