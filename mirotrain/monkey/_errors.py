# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-

# This file is adapted from
# https://github.com/gevent/gevent/blob/73025a8837b3bff19c106e877fa2374889c59dd3/src/gevent/monkey/_errors.py

"""
Exception classes and errors that this package may raise.
"""


class _BadModule(ImportError):
    """
    Raised when a module is not importable.
    """

    def __init__(self, name):
        ImportError.__init__(self, "Module %r is not importable" % (name,))


class _BadTargets(AttributeError):
    """
    Raised when ``__targets__`` is incorrect.
    """

    def __init__(self, module, target):
        AttributeError.__init__(
            self,
            "Module %r has a bad or missing value %r for __targets__"
            % (
                target,
                module,
            ),
        )


class _BadImplements(AttributeError):
    """
    Raised when ``__implements__`` is incorrect.
    """

    def __init__(self, module, implements):
        AttributeError.__init__(
            self,
            "Module %r has a bad or missing value %r for __implements__"
            % (
                implements,
                module,
            ),
        )
