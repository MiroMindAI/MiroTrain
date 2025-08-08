# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from torchtune.training.checkpointing._utils import ModelType as original_ModelType

STEP_KEY = "global_step"

ModelType = Enum(
    "ModelType",
    {
        **{member.name: member.value for member in original_ModelType},
        "QWEN3": "qwen3",
        "QWEN3_MoE": "qwen3_moe",
    },
)
