# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-

# This file is adapted from https://github.com/gevent/gevent/blob/73025a8837b3bff19c106e877fa2374889c59dd3/src/gevent/monkey/api.py

"""
Higher level functions that comprise parts of
the public monkey patching API.
"""

from importlib import import_module
from types import ModuleType
from typing import Any, Sequence


def get_original(mod_name: str | Sequence[str], item_name: str):
    """
    Retrieve the original object from a module.

    If the object has not been patched, then that object will still be
    retrieved.

    :param str|sequence mod_name: The name of the standard library module,
        e.g., ``'socket'``. Can also be a sequence of standard library
        modules giving alternate names to try, e.g., ``('thread', '_thread')``;
        the first importable module will supply all *item_name* items.
    :param str|sequence item_name: A string or sequence of strings naming the
        attribute(s) on the module ``mod_name`` to return.

    :return: The original value if a string was given for
             ``item_name`` or a sequence of original values if a
             sequence was passed.
    """
    from ._state import _get_original

    mod_names = [mod_name] if isinstance(mod_name, str) else mod_name
    if isinstance(item_name, str):
        item_names = [item_name]
        unpack = True
    else:
        item_names = item_name
        unpack = False

    for mod in mod_names:
        try:
            result = _get_original(mod, item_names)
        except ImportError:
            if mod is mod_names[-1]:
                raise
        else:
            return result[0] if unpack else result


_NONE = object()


def patch_item(module: ModuleType, attr: str, newitem: Any):
    from ._state import _save

    olditem = getattr(module, attr, _NONE)
    if olditem is not _NONE:
        _save(module, attr, olditem)
    setattr(module, attr, newitem)


def remove_item(module: ModuleType, attr: str):
    from ._state import _save

    olditem = getattr(module, attr, _NONE)
    if olditem is _NONE:
        return
    _save(module, attr, olditem)

    delattr(module, attr)


def patch_module(
    target_module: ModuleType, source_module: ModuleType, items: list[str]
):
    """raises no exception."""
    for attr in items:
        patch_item(target_module, attr, getattr(source_module, attr))


def patch_by_source_module_fqn(source_module_fqn: str):
    """
    Replace **attributes** in **target_module** with the attributes of the
    same name in **source_module**.

    Core logic:
    ```python
    source_module: ModuleType = importlib.import_module(module_fqn)
    target_modules: list[ModuleType] = [importlib.import_module(x) for x in source_module.__targets__]
    attributes: list[str] = source_module.__implements__
    ```

    May raises:
    - _BadModule: if the source module cannot be imported.
    - _BadTarget: if the source module does not have a valid `__targets__` attribute.
    - _BadImplements: if the source module does not have a valid `__implements__` attribute.

    """
    from ._errors import _BadImplements, _BadModule, _BadTargets

    try:
        source_module = import_module(source_module_fqn)
    except ImportError as e:
        raise _BadModule(source_module_fqn) from e

    target_module_names = None
    try:
        target_module_names = source_module.__targets__
    except AttributeError as e:
        raise _BadTargets("No __targets__ is set", source_module_fqn) from e

    if not isinstance(target_module_names, (list, tuple)):
        raise _BadTargets("__targets__ is not iterable", source_module_fqn)

    target_modules = []
    for target in target_module_names:
        try:
            target_module = import_module(target)
            target_modules.append(target_module)
        except ImportError as e:
            raise _BadTargets(f"failed to import {target}", source_module_fqn) from e

    target_attr_list = None
    try:
        target_attr_list = source_module.__implements__
    except AttributeError as e:
        raise _BadImplements("No __implements__ is set", source_module_fqn) from e

    if not isinstance(target_attr_list, (list, tuple)):
        raise _BadImplements("__implements__ is not iterable", source_module_fqn)

    for attr in target_attr_list:
        if not isinstance(attr, str):
            raise _BadImplements(f"{attr} is not a string", source_module_fqn)
        if not hasattr(source_module, attr):
            raise _BadImplements(f"{attr} is not found", source_module_fqn)

    for target_module in target_modules:
        patch_module(target_module, source_module, target_attr_list)
