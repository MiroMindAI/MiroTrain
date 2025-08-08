# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""
YARN model builders for Qwen3 models.

This module provides YARN-extended versions of Qwen3 model builders that support
longer context lengths through RoPE scaling techniques.
"""

from ._checkpointing_utils import patch_copy_files
from ._component_builders import qwen3
from ._model_builders import (
    qwen3_0_6b_base,
    qwen3_0_6b_instruct,
    qwen3_14b_base,
    qwen3_14b_instruct,
    qwen3_1_7b_base,
    qwen3_1_7b_instruct,
    qwen3_32b,
    qwen3_4b_base,
    qwen3_4b_instruct,
    qwen3_8b_base,
    qwen3_8b_instruct,
)
from ._positional_embeddings import RopeScaling
from ._validate import validate_yarn_cfg


def _create_yarn_model_builder(original_model_builder):
    """
    Create a YARN-extended version of a model builder.

    This function takes an existing model builder, extracts its call args, and creates a new version that supports YARN.

    Args:
        original_model_builder: The original model builder function to extract call args from

    Returns:
        A new model builder function that accepts max_position_embeddings and rope_scaling
    """

    # Store reference to original qwen3 component builder
    original_qwen3_builder = original_model_builder.__globals__["qwen3"]

    # Temporarily replace qwen3 to intercept call args
    original_model_builder.__globals__["qwen3"] = lambda **kwargs: kwargs

    # Execute original model builder to capture the call args it would use
    captured_kwargs = original_model_builder()

    # Restore the original qwen3 component builder
    original_model_builder.__globals__["qwen3"] = original_qwen3_builder

    def yarn_model_builder(max_position_embeddings: int, rope_scaling: RopeScaling):
        """
        Build a YARN-extended Qwen3 model with custom position embeddings and RoPE scaling.

        Args:
            max_position_embeddings: Maximum sequence length the model will be run with
            rope_scaling: RoPE scaling configuration for extended context

        Returns:
            A Qwen3 model configured with YARN
        """

        validate_yarn_cfg(max_position_embeddings, rope_scaling)
        patch_copy_files(max_position_embeddings, rope_scaling)

        # Configure YARN-specific parameters
        captured_kwargs["max_seq_len"] = max_position_embeddings
        captured_kwargs["rope_scaling"] = rope_scaling

        # Build the actual model with YARN
        model = qwen3(**captured_kwargs)

        return model

    return yarn_model_builder


# Create YARN-extended versions of all Qwen3 model builders
(
    qwen3_0_6b_base_yarn,
    qwen3_0_6b_instruct_yarn,
    qwen3_1_7b_base_yarn,
    qwen3_1_7b_instruct_yarn,
    qwen3_4b_base_yarn,
    qwen3_4b_instruct_yarn,
    qwen3_8b_base_yarn,
    qwen3_8b_instruct_yarn,
    qwen3_14b_base_yarn,
    qwen3_14b_instruct_yarn,
    qwen3_32b_yarn,
) = map(
    _create_yarn_model_builder,
    [
        qwen3_0_6b_base,
        qwen3_0_6b_instruct,
        qwen3_1_7b_base,
        qwen3_1_7b_instruct,
        qwen3_4b_base,
        qwen3_4b_instruct,
        qwen3_8b_base,
        qwen3_8b_instruct,
        qwen3_14b_base,
        qwen3_14b_instruct,
        qwen3_32b,
    ],
)
