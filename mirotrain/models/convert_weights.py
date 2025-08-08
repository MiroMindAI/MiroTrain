# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtune.models.convert_weights

import re

from typing import Dict


def get_mapped_key(key: str, mapping_dict: Dict[str, str]) -> str:
    try:
        # Checks if there is a layer # in the key
        if any(k.isdigit() for k in key.split(".")):
            # Replace layer number with "{}" to create key for lookup
            abstract_key = re.sub(r"(\.\d+)", ".{}", key)
            # search all layer number
            layer_nums = re.findall(r"\d+", key)
            new_key = mapping_dict[abstract_key]
            new_key = new_key.format(*layer_nums)
        else:
            new_key = mapping_dict[key]
    except KeyError as e:
        raise Exception(
            f'Error converting the state dict. Found unexpected key: "{key}". '
            "Please make sure you're loading a checkpoint with the right format. "
        ) from e

    return new_key
