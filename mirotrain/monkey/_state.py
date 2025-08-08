# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-

# This file is adapted from
# https://github.com/gevent/gevent/blob/73025a8837b3bff19c106e877fa2374889c59dd3/src/gevent/monkey/_state.py

"""
State management and query functions for tracking and discovering what
has been patched.
"""
import importlib
from types import ModuleType
from typing import Any, Sequence

# maps module name -> {attribute name: original item}
# e.g. "time" -> {"sleep": built-in function sleep}
# NOT A PUBLIC API. However, third-party monkey-patchers may be using
# it? TODO: Provide better API for them.
saved: dict[str, dict[str, Any]] = {}


def is_module_patched(mod_name: str):
    """
    Check if a module has been replaced with a cooperative version.

    :param str mod_name: The name of the standard library module,
        e.g., ``'socket'``.

    """
    return mod_name in saved


def is_object_patched(mod_name: str, item_name: str):
    """
    Check if an object in a module has been replaced with a
    cooperative version.

    :param str mod_name: The name of the standard library module,
        e.g., ``'socket'``.
    :param str item_name: The name of the attribute in the module,
        e.g., ``'create_connection'``.

    """
    return is_module_patched(mod_name) and item_name in saved[mod_name]


def is_anything_patched():
    """
    Check if this module has done any patching in the current process.
    This is currently only used in gevent tests.

    Not currently a documented, public API, because I'm not convinced
    it is 100% reliable in the event of third-party patch functions that
    don't use ``saved``.

    .. versionadded:: 21.1.0
    """
    return bool(saved)


def _get_original(name: str, items: Sequence[str]):
    d = saved.get(name, {})
    values = []
    module = None
    for item in items:
        if item in d:
            values.append(d[item])
        else:
            if module is None:
                # Quoted from https://docs.python.org/3/library/importlib.html:
                # import_module() returns the specified package or module (e.g. pkg.mod),
                # while __import__() returns the top-level package or module.
                module = importlib.import_module(name)
            values.append(getattr(module, item))
    return values


def _save(module: ModuleType, attr_name: str, item: Any):
    saved.setdefault(module.__name__, {}).setdefault(attr_name, item)
