# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtune.training._distributed

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from mirotrain.modules.moe import (
    get_expert_parallel_group,
    get_expert_parallel_rank,
    get_expert_parallel_world_size,
    set_expert_parallel_group,
)
from mirotrain.modules.ulysses import set_ulysses_sequence_parallel_group
from torch import nn

from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torchao.dtypes.nf4tensor import NF4Tensor
from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import get_adapter_state_dict


@dataclass
class ParallelDims:
    dp_replicate: int = 1
    dp_shard: int = 1
    tp: int = 1
    ulysses_sp: int = 1
    moe_ep: int = 1
    world_size: int = 1

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, tp, ulysses_sp = (
            self.dp_replicate,
            self.dp_shard,
            self.tp,
            self.ulysses_sp,
        )
        if tp > 1:
            assert ulysses_sp == 1, "ulysses_sp is not compatible with TP now"

        for d in (dp_replicate, tp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"

        assert dp_shard == -1 or dp_shard >= 1, " dp_shard must -1 or >=1."
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * tp)
        assert dp_shard >= 1

        assert dp_replicate * dp_shard * tp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"tp({tp}) != WORLD_SIZE({self.world_size})"
        )
        assert (
            dp_replicate * dp_shard % ulysses_sp == 0
        ), "dp_degree must be divided by ulysses_sp"

        assert (
            self.world_size % self.moe_ep == 0
        ), "world_size must be divided by moe_ep size"

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.dp_replicate, self.dp_shard, self.tp],
            ["dp_replicate", "dp_shard", "tp"],
        ):
            if d > 1:
                dims.append(d)
                names.append(name)

        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        # Create all the submesh here to ensure all required process groups are
        # initialized:
        # Mesh for data loading (no communication on this mesh)
        dp_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
        if self.dp_shard_enabled:
            dp_mesh_dim_names.append("dp_shard")

        if dp_mesh_dim_names != []:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")

        # set up ulysses_device_mesh and process group with init_device_mesh
        real_dp_size = self.dp_replicate * self.dp_shard // self.ulysses_sp
        ulysses_device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(real_dp_size, self.ulysses_sp),
            mesh_dim_names=("ulysses_dp", "ulysses_sp"),
        )
        set_ulysses_sequence_parallel_group(
            ulysses_device_mesh["ulysses_sp"].get_group()
        )

        # set up moe_device_mesh and process group with init_device_mesh
        moe_edp_size = self.world_size // self.moe_ep
        moe_device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(moe_edp_size, self.moe_ep),
            mesh_dim_names=("moe_edp", "moe_ep"),
        )
        set_expert_parallel_group(moe_device_mesh["moe_ep"].get_group())

        return mesh, ulysses_device_mesh, moe_device_mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def tp_enabled(self):
        return self.tp > 1

    @cached_property
    def non_data_parallel_size(self):
        # update below as more parallelism options are implemented
        return self.tp


def _gather(input_, group, world_size=1, rank=0, dim=-1):
    # skip if only one rank involved
    if world_size == 1:
        return input_

    # all gather
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    dist.all_gather(tensor_list, input_, group=group)

    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


def gather_cpu_state_dict(
    model: "FSDPModule",  # noqa
    is_rank_zero: bool,
    device: Optional[torch.device] = None,
    adapter_weights_only: bool = False,
) -> Dict[str, Any]:
    """
    Converting sharded state dict into a full state dict on CPU
    Returning non-empty result only on rank0 to avoid peaking CPU memory
    Currenltly we can used distributed state dict API to process model without NF4Tensor. Otherwise, we need to
    manually gather any NF4 tensors until all-gather is supported in the NF4Tensor subclass
    TODO: add support for NF4Tensor at distributed state dict API

    Args:
        model (FSDPModule): Model to generate fully qualified names for cpu_state_dict
        is_rank_zero (bool): flag to check if the process is on rank 0
        device (Optional[torch.device]): device to use for sharded tensors. Default: None
        adapter_weights_only (bool): flag to check if only trainable parameters should be returned. Default: False

    Returns:
        Dict[str, Any]: State dict on CPU
    """
    # TODO: Disabling DSD as it has issues. Add back changes in #2138 once DSD issue is fixed.
    cpu_state_dict = {}
    sharded_sd = model.state_dict()
    ep_size = get_expert_parallel_world_size()
    ep_rank = get_expert_parallel_rank()
    ep_group = get_expert_parallel_group()
    for param_name, param in sharded_sd.items():
        if param.is_cpu:
            # Move back to device if offloaded to CPU
            param = param.to(device)
        if hasattr(param, "_local_tensor"):
            if isinstance(param._local_tensor, NF4Tensor):
                param = _gather_nf4_tensor(param)
            else:
                # Gather DTensor
                param = param.full_tensor()
        if isinstance(param, NF4Tensor):
            # upcasting NF4 to original dtype
            param = param.to(param.dtype)
        if is_rank_zero:
            cpu_state_dict[param_name] = param.cpu()
        torch.distributed.barrier()

        # merge experts in expert parallel group
        if "experts" in param_name and ep_size > 1:
            _total_experts = _gather(param, ep_group, ep_size, ep_rank, dim=0).cpu()
            param = None
            if is_rank_zero:
                cpu_state_dict.update({param_name: _total_experts})
        torch.distributed.barrier()

    if adapter_weights_only:
        cpu_state_dict = get_adapter_state_dict(cpu_state_dict, device=None)
    return cpu_state_dict


def shard_model(
    model: TransformerDecoder,
    shard_conditions: List[Callable[[str, nn.Module], bool]],
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
    dp_mesh: Optional[DeviceMesh] = None,
    ep_mesh: Optional[DeviceMesh] = None,
) -> None:
    """
    Utility to shard a model with FSDP using the PyTorch Distributed fully_shard API.

    This method will over the model's named modules from the bottom-up and apply shard modules
    based on whether they meet any of the criteria from shard_conditions.

    Args:
        model (TransformerDecoder): Model to shard with FSDP.
        shard_conditions (List[Callable[[str, nn.Module], bool]]): A list of functions to determine
            which modules to shard with FSDP. Each function should take module name (relative to root)
            and the module itself, returning True if FSDP should shard the module and False otherwise.
            If any of shard_conditions return True for a given module, it will be sharded by FSDP.
        cpu_offload (bool): If set to True, FSDP will offload parameters, gradients, and optimizer
            states to CPU.
        reshard_after_forward (bool): Whether to reshard parameters and buffers after
            the forward pass. Setting this to True corresponds to the FULL_SHARD sharding strategy
            from FSDP1, while setting it to False corresponds to the SHARD_GRAD_OP sharding strategy.
        dp_mesh (Optional[DeviceMesh]): Device mesh to use for FSDP sharding under mutliple parallelism.
            Default to None.
        ep_mesh (Optional[DeviceMesh]): Device mesh to use for FSDP sharding under expert parallelism.
            Default to None.
    """
    # fsdp_kwargs = {"reshard_after_forward": reshard_after_forward, "mesh": dp_mesh}
    # if cpu_offload:
    #     fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    # if get_expert_parallel_world_size() > 1:
    #     for layer_id, layer in enumerate(model.layers):
    #         fully_shard(
    #             layer.mlp.experts,
    #             mesh=ep_mesh["moe_edp"],
    #             reshard_after_forward=reshard_after_forward,
    #         )

    # # Shard the model with FSDP, iterating in reverse to start with
    # # lowest-level modules first
    # num_layers_sharded = 0
    # for n, m in reversed(list(model.named_modules())):
    #     if any([shard_condition(n, m) for shard_condition in shard_conditions]):
    #         fully_shard(m, **fsdp_kwargs)
    #         num_layers_sharded += 1

    # if num_layers_sharded == 0:
    #     raise ValueError(
    #         "No layer modules were sharded. Please check if shard conditions are working as expected."
    #     )

    # # Finally shard the entire model to account for any stragglers
    # fully_shard(model, **fsdp_kwargs)

    dp_shard_kwargs = {"reshard_after_forward": reshard_after_forward, "mesh": dp_mesh}
    expert_shard_kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": ep_mesh["moe_edp"],
    }

    if cpu_offload:
        dp_shard_kwargs["offload_policy"] = CPUOffloadPolicy()
        expert_shard_kwargs["offload_policy"] = CPUOffloadPolicy()

    for layer_id, layer in enumerate(model.layers):
        if get_expert_parallel_world_size() > 1:
            fully_shard(layer.mlp.experts, **expert_shard_kwargs)
        fully_shard(layer, **dp_shard_kwargs)

    if get_expert_parallel_world_size() == 1:
        fully_shard(model, mesh=dp_mesh, reshard_after_forward=True)
