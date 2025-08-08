# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import functools
from typing import Iterable, List, Optional, Union

import torch

from mirotrain.modules.moe import get_expert_parallel_group
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.utils._foreach_utils import _device_has_foreach_support, _has_foreach_support

from ._grad_scaler import _group_tensors_by_device_and_mesh


__all__ = ["clip_grad_norm_"]


_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def _no_grad(func):
    """
    This wrapper is needed to avoid a circular import when using @torch.no_grad on the exposed functions
    clip_grad_norm_ and clip_grad_value_ themselves.
    """

    def _no_grad_wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    functools.update_wrapper(_no_grad_wrapper, func)
    return _no_grad_wrapper


# Adapted from torch.nn.utils.clip_grad_norm_
@_no_grad
def clip_grad_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    The norm is computed over the norms of the individual gradients of all parameters,
    as if the norms of the individual gradients were concatenated into a single vector.
    Gradients are modified in-place.

    Args:
        parameters (_tensor_or_tensors): An iterable of Tensors or a single Tensor
            that will have gradients normalized.
        max_norm (float): Max norm of the gradients.
        norm_type (float, optional): Type of the used p-norm. Can be 'inf' for infinity norm. Defaults to 2.0.
        error_if_nonfinite (bool, optional): If True, an error is thrown if the total norm of the gradients
            from `parameters` is nan, inf, or -inf. Defaults to False.
        foreach (Optional[bool], optional): Use the faster foreach-based implementation.
            If None, use the foreach implementation for CUDA and CPU native tensors and silently fall back to
                the slow implementation for other device types. Defaults to None.

    Returns:
        torch.Tensor: Total norm of the parameter gradients (viewed as a single vector).

    Raises:
        RuntimeError: If `foreach=True` is passed but the foreach API is not supported on the device.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    first_device = grads[0].device
    grouped_grads = _group_tensors_by_device_and_mesh(grads)

    total_norm = []
    for device, device_grads in grouped_grads.items():  # type: ignore[assignment]
        for mesh, device_mesh_grads in device_grads.items():
            if len(device_mesh_grads) <= 0:
                continue

            norms: List[Tensor] = []
            # Separate DTensors from regular tensors to avoid mixed tensor type issues
            dtensor_grads = [g for g in device_mesh_grads if isinstance(g, DTensor)]
            regular_grads = [g for g in device_mesh_grads if not isinstance(g, DTensor)]

            if (
                foreach is None and _has_foreach_support(device_mesh_grads, device)
            ) or (foreach and _device_has_foreach_support(device)):
                # Handle DTensors and regular tensors separately
                if dtensor_grads:
                    dtensor_norms = torch._foreach_norm(dtensor_grads, norm_type)
                    norms.extend(dtensor_norms)
                if regular_grads:
                    regular_norms = torch._foreach_norm(regular_grads, norm_type)
                    norms.extend(regular_norms)
            elif foreach:
                raise RuntimeError(
                    f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
                )
            else:
                norms.extend(
                    [torch.linalg.vector_norm(g, norm_type) for g in device_mesh_grads]
                )

            _group_norm = torch.linalg.vector_norm(
                torch.stack([norm.to(first_device) for norm in norms]), norm_type
            )

            # reduce grad norm in weight shard FSDP group
            _group_norm = _group_norm.full_tensor()

            # reduce grad norm in expert parallel group
            if mesh == "experts":
                torch.distributed.all_reduce(
                    _group_norm, group=get_expert_parallel_group()
                )

            total_norm.append(_group_norm)
    total_norm = sum(total_norm)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for device, device_grads in grouped_grads.items():  # type: ignore[assignment]
        for _, device_mesh_grads in device_grads.items():
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
                    torch._foreach_mul_(dtensor_grads, clip_coef_clamped.to(device))
                if regular_grads:
                    torch._foreach_mul_(regular_grads, clip_coef_clamped.to(device))
            elif foreach:
                raise RuntimeError(
                    f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
                )
            else:
                clip_coef_clamped_device = clip_coef_clamped.to(device)
                for g in device_mesh_grads:
                    g.mul_(clip_coef_clamped_device)

    return total_norm
