# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""
author: lei.lei@shanda.com
time: 2025/05/29 10:38
description: gevent style monkey patching:
1. explicit: user calls `mirotrain.monkey.patch_common()`.
2. declarative: module under `mirotrain` uses `__targets__` and `__implements__` to inform WHAT to patch.
3. silent error: when patching failed, gives a warning and continue. no exception will be raised.
"""
import logging

from ._state import is_anything_patched, is_module_patched, is_object_patched
from .api import patch_by_source_module_fqn

__all__ = [
    "patch_common",
    "is_object_patched",
    "is_module_patched",
    "is_anything_patched",
]

logger = logging.getLogger("mirotrain")


def patch_common():
    """TODO: add other patchese here"""
    patch_by_source_module_fqn("mirotrain.data._messages")
    patch_by_source_module_fqn("mirotrain.datasets._packed")
    patch_by_source_module_fqn("mirotrain.training.checkpointing._checkpointer")
