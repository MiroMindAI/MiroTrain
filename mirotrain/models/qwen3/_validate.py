# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

from ._positional_embeddings import RopeScaling


def validate_yarn_cfg(max_position_embeddings: int, rope_scaling: RopeScaling):
    """
    Validate YARN-specific configs
    """

    rope_type = rope_scaling.get("rope_type")
    if rope_type is None:
        raise KeyError("'rope_type' must be specified in 'rope_scaling' configuration.")
    if rope_type != "yarn":
        raise ValueError(
            f"Unsupported 'rope_type': {rope_type}. "
            "Currently, only 'yarn' is supported for Qwen patching."
        )

    factor = rope_scaling.get("factor")
    if factor is None:
        raise KeyError("'factor' must be specified in 'rope_scaling' for YARN.")

    original_max_position_embeddings = rope_scaling.get(
        "original_max_position_embeddings"
    )
    if original_max_position_embeddings is None:
        raise KeyError(
            "'original_max_position_embeddings' must be specified in 'rope_scaling' for YARN."
        )

    if max_position_embeddings is None:
        raise KeyError(
            "'max_position_embeddings' must be specified in model_extra when rope_scaling is enabled."
        )

    if max_position_embeddings != int(original_max_position_embeddings * factor):
        raise ValueError(
            f"Configured 'max_position_embeddings' ({max_position_embeddings}) is not equal to "
            f"'original_max_position_embeddings' ({original_max_position_embeddings}) * "
            f"'rope_scaling.factor' ({factor})."
        )
