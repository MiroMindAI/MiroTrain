# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtune.training._grad_scaler

from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.nn.utils.clip_grad import _no_grad, _tensor_or_tensors
from torch.utils._foreach_utils import _device_has_foreach_support, _has_foreach_support


@_no_grad
def scale_grads_(
    parameters: _tensor_or_tensors,
    scaler: torch.Tensor,
    foreach: Optional[bool] = None,
) -> None:
    r"""Scale gradients of iterable parameters.

    This function is equivalent to :func:`torch.mul_` applied to each parameter.
    Gradients are modified in-place, multiplying by specified scaler.

    Args:
        parameters (_tensor_or_tensors): an iterable of Tensors or a
            single Tensor that will have gradients scaled
        scaler (torch.Tensor): multiplier to scale gradients
        foreach (Optional[bool]): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
    Returns:
        None
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    _scale_grad_(parameters, scaler, foreach)


def _group_tensors_by_device_and_mesh(
    tensors: List[torch.Tensor],
) -> Dict[torch.device, Dict[str, List[Tensor]]]:
    """
    Group tensors by device and mesh type for MoE expert parallel training.

    This function separates expert parameters (using moe_edp mesh) from dense parameters
    to enable different gradient processing strategies for MoE models.

    Args:
        tensors (List[torch.Tensor]): List of tensors to group

    Returns:
        Dict[torch.device, Dict[str, List[Tensor]]]: Dictionary mapping device -> mesh_type -> list of tensors
        mesh_type can be "experts" (for MoE expert parameters) or "default" (for dense parameters)
    """
    # Use defaultdict to avoid repeated dict key checks
    ret: Dict[torch.device, Dict[str, List[Tensor]]] = defaultdict(
        lambda: {"default": [], "experts": []}
    )

    for tensor in tensors:
        device = tensor.device
        mesh_type = "default"  # Default to dense parameters

        # Check if tensor is an expert parameter (has moe_edp mesh)
        try:
            if hasattr(tensor, "device_mesh") and tensor.device_mesh is not None:
                if "moe_edp" in tensor.device_mesh.mesh_dim_names:
                    mesh_type = "experts"
        except (AttributeError, KeyError):
            # Fallback to default if any error occurs
            pass

        ret[device][mesh_type].append(tensor)

    return ret


@_no_grad
def _scale_grad_(
    parameters: _tensor_or_tensors,
    scaler: torch.Tensor,
    foreach: Optional[bool] = None,
) -> None:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return
    grouped_grads = _group_tensors_by_device_and_mesh(grads)

    for device, device_grads in grouped_grads.items():
        for mesh_type, device_mesh_grads in device_grads.items():
            if len(device_mesh_grads) <= 0:
                continue
            if (
                foreach is None and _has_foreach_support(device_mesh_grads, device)
            ) or (foreach and _device_has_foreach_support(device)):
                # Separate DTensors from regular tensors to avoid mixed tensor type issues
                dtensor_grads = [g for g in device_mesh_grads if isinstance(g, DTensor)]
                regular_grads = [
                    g for g in device_mesh_grads if not isinstance(g, DTensor)
                ]

                # Handle DTensors and regular tensors separately
                if dtensor_grads:
                    torch._foreach_mul_(dtensor_grads, scaler.to(device))
                if regular_grads:
                    torch._foreach_mul_(regular_grads, scaler.to(device))
            elif foreach:
                raise RuntimeError(
                    f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
                )
            else:
                scaler_device = scaler.to(device)
                for g in device_mesh_grads:
                    if isinstance(g, DTensor):
                        g[:] = g * scaler_device
                    else:
                        g.mul_(scaler_device)
