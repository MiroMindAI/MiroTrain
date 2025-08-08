# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch.distributed as dist

_EXPERT_PARALLEL_GROUP = None


def set_expert_parallel_group(group: dist.ProcessGroup):
    """
    Set expert parallel process group.
    """
    global _EXPERT_PARALLEL_GROUP
    _EXPERT_PARALLEL_GROUP = group


def get_expert_parallel_group() -> Optional[dist.ProcessGroup]:
    """
    Get expert parallel process group.
    """
    global _EXPERT_PARALLEL_GROUP
    return _EXPERT_PARALLEL_GROUP


def get_expert_parallel_world_size(group: dist.ProcessGroup = None) -> int:
    """
    Get expert parallel world size.
    """
    group = get_expert_parallel_group() if group is None else group
    return dist.get_world_size(group) if group else 1


def get_expert_parallel_rank(group: dist.ProcessGroup = None) -> int:
    """
    Get expert parallel rank.
    """
    group = get_expert_parallel_group() if group is None else group
    return dist.get_rank(group) if group else 0
