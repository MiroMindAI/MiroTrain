# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import json
from functools import wraps
from pathlib import Path

from ._positional_embeddings import RopeScaling


def patch_copy_files(max_position_embeddings: int, rope_scaling: RopeScaling):
    import mirotrain.training.checkpointing._checkpointer as _checkpointer

    original_copy_files = _checkpointer.copy_files

    @wraps(original_copy_files)
    def patched_copy_files(
        input_dir: str | Path, output_dir: str | Path, *args, **kwargs
    ):
        """
        Find the config.json in output_dir and update it with custom model_extra parameters.

        Modifications applied to config.json:
            - max_position_embeddings: Updated if specified in model_extra
            - rope_scaling: Added/replaced if specified in model_extra
        """

        original_copy_files(input_dir, output_dir, *args, **kwargs)

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        config_path = output_dir / "config.json"
        if not config_path.exists():
            return

        with open(config_path, "r") as f:
            config = json.load(f)

        config["max_position_embeddings"] = max_position_embeddings
        config["rope_scaling"] = rope_scaling

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")

        return patched_copy_files

    _checkpointer.copy_files = patched_copy_files
