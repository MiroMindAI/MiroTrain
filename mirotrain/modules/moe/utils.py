# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from mirotrain.modules.moe.expert_parallel import get_expert_parallel_group
from torch import Tensor


def silu(w1_o, w2_o):
    return F.silu(w1_o) * w2_o


Silu = torch.jit.script(silu)


# Based on https://github.com/pytorch/pytorch/pull/40762
class AllToAll(torch.autograd.Function):
    """
    All to all communication
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        output_split_sizes=None,
        input_split_sizes=None,
        group: torch.distributed.ProcessGroup = None,
        async_op=False,
    ) -> Tensor:  # type: ignore

        ctx.input_shape = inputs.shape
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return inputs, None

        inputs = inputs.contiguous()
        out = (
            torch.empty_like(inputs)
            if output_split_sizes is None
            else inputs.new_empty(
                size=[sum(output_split_sizes)] + list(inputs.size()[1:])
            )
        )
        handle = torch.distributed.all_to_all_single(
            out,
            inputs,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
        )

        # if async_op=False, handle will be None
        return out, handle

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor, _) -> Tuple[None, Tensor]:
        if ctx.needs_input_grad[0]:
            # Bypass the function if we are using only 1 GPU.
            world_size = torch.distributed.get_world_size(group=ctx.group)
            if world_size == 1:
                return grad_output, None, None, None, None

            grad_output = grad_output.contiguous()
            out = torch.empty(
                ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype
            )
            torch.distributed.all_to_all_single(
                out,
                grad_output,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
            )
            return out, None, None, None, None
        return None, None, None, None, None


def all_to_all(
    x, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False
):
    return AllToAll.apply(x, output_split_sizes, input_split_sizes, group, async_op)


class MoEGather(torch.autograd.Function):
    """Gather the input tensor based on the map tensor."""

    @staticmethod
    def forward(ctx, input_, map_):
        """Gather the input tensor based on the map tensor."""
        ctx.input_size = input_.size()
        ctx.map = map_
        return torch.gather(input_, 0, map_)

    @staticmethod
    def backward(ctx, grad_output):
        """Scatter the grad_output tensor based on the map tensor."""
        input_size = ctx.input_size
        map_ = ctx.map

        output = torch.zeros(
            input_size, dtype=grad_output.dtype, device=torch.cuda.current_device()
        )
        output.scatter_add_(0, map_, grad_output)
        return output, None, None


class MoEScatter(torch.autograd.Function):
    """Scatter the input tensor based on the map tensor."""

    @staticmethod
    def forward(ctx, input_, map_, output_size=None):
        """Scatter the input tensor based on the map tensor."""
        ctx.map = map_
        if output_size is not None:
            output = torch.zeros(output_size, dtype=input_.dtype, device=input_.device)
        else:
            output = torch.zeros_like(input_)

        output.scatter_add_(0, map_, input_)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Gather the grad_output tensor based on the map tensor."""
        map_ = ctx.map
        grad_input = torch.gather(grad_output, 0, map_)
        return grad_input, None, None, None


def _gather_along_first_dim_moe(input_):
    """Gather tensors and concatenate along the first dimension."""
    group = get_expert_parallel_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
    )
    torch.distributed._all_gather_base(output, input_.contiguous(), group=group)

    return output


def _reduce_scatter_along_first_dim_moe(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    group = get_expert_parallel_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0
    dim_size[0] = dim_size[0] // world_size
    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
    )
    torch.distributed._reduce_scatter_base(output, input_.contiguous(), group=group)
    return output


class _GatherFromSequenceParallelRegionToMOE(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""  # TODO

    @staticmethod
    def symbolic(graph, input_):  # pylint: disable=W0613
        """Symbolic function for tracing."""
        return _gather_along_first_dim_moe(input_)

    @staticmethod
    def forward(ctx, input_):  # pylint: disable=W0613
        """Forward function."""
        return _gather_along_first_dim_moe(input_)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=W0613
        """Backward function."""
        return _reduce_scatter_along_first_dim_moe(grad_output), None


class _ReduceScatterToSequenceParallelRegionFromMOE(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):  # pylint: disable=W0613
        """Symbolic function for tracing."""
        return _reduce_scatter_along_first_dim_moe(input_)

    @staticmethod
    def forward(ctx, input_):  # pylint: disable=W0613
        """Forward function."""
        return _reduce_scatter_along_first_dim_moe(input_)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=W0613
        """Backward function."""
        return _gather_along_first_dim_moe(grad_output), None


def gather_from_parallel_region_to_moe(input_):
    """Wrapper for autograd function"""
    return _GatherFromSequenceParallelRegionToMOE.apply(input_)


def reduce_scatter_to_parallel_region_from_moe(input_):
    """Wrapper for autograd function"""
    return _ReduceScatterToSequenceParallelRegionFromMOE.apply(input_)


def gather_along_first_dim_expert_parallel(input_):
    """Gather tensors and concatenate along the first dimension."""
    group = get_expert_parallel_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
    )

    # torch.distributed._all_gather_base(output, input_.contiguous(), group=group)
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)

    return output


# Adapted from https://github.com/huggingface/transformers/blob/81799d8b556b3c810ed314187674bc439c0582b4/src/\
# transformers/models/qwen3_moe/modeling_qwen3_moe.py#L596
def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[torch.Tensor, Tuple[torch.Tensor], None]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts (Optional[int]):
            Number of experts
        top_k (int):
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (Optional[torch.Tensor]):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    # Check if all values in attention_mask are 0
    if attention_mask is not None and torch.all(attention_mask == 0):
        return torch.tensor(0.0, device=gate_logits[0].device, dtype=torch.float32)

    if not isinstance(gate_logits, tuple):
        return 0

    compute_device = gate_logits[0].device
    overall_loss = 0.0

    # Process each layer separately to reduce memory usage
    for layer_gate in gate_logits:
        layer_gate = layer_gate.to(compute_device)

        routing_weights = torch.nn.functional.softmax(layer_gate, dim=-1)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

        if attention_mask is None:
            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.mean(routing_weights, dim=0)
        else:
            batch_size, sequence_length = attention_mask.shape

            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
            expert_attention_mask = (
                attention_mask[None, :, :, None, None]
                .expand((1, batch_size, sequence_length, top_k, num_experts))
                .reshape(-1, top_k, num_experts)
                .to(compute_device)
            )

            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.sum(
                expert_mask.float() * expert_attention_mask, dim=0
            ) / torch.sum(expert_attention_mask, dim=0)

            # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
            router_per_expert_attention_mask = (
                attention_mask[None, :, :, None]
                .expand((1, batch_size, sequence_length, num_experts))
                .reshape(-1, num_experts)
                .to(compute_device)
            )

            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.sum(
                routing_weights * router_per_expert_attention_mask, dim=0
            ) / torch.sum(router_per_expert_attention_mask, dim=0)

        layer_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
        overall_loss += layer_loss

    return overall_loss * num_experts


# Adapted from https://github.com/huggingface/transformers/blob/81799d8b556b3c810ed314187674bc439c0582b4/src/\
# transformers/models/qwen3_moe/modeling_qwen3_moe.py#L596
def load_balancing_loss_func_allinone(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[torch.Tensor, Tuple[torch.Tensor], None]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts (Optional[int]):
            Number of experts
        top_k (int):
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (Optional[torch.Tensor]):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    # Check if all values in attention_mask are 0
    if attention_mask is not None and torch.all(attention_mask == 0):
        return torch.tensor(0.0, device=gate_logits[0].device, dtype=torch.float32)

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
        )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
            batch_size * sequence_length
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand(
                (num_hidden_layers, batch_size, sequence_length, top_k, num_experts)
            )
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts
