#!/usr/bin/env python3

from contextlib import contextmanager
import os


_TENSORRT_COMPAT = False
_TRUE_VALUES = {"1", "true", "yes", "on"}


def set_tensorrt_compat(enabled: bool = True):
    global _TENSORRT_COMPAT
    _TENSORRT_COMPAT = bool(enabled)


def tensorrt_compat_enabled() -> bool:
    if _TENSORRT_COMPAT:
        return True
    return os.environ.get("DOMI_TENSORRT_COMPAT", "").lower() in _TRUE_VALUES


@contextmanager
def tensorrt_compat(enabled: bool = True):
    global _TENSORRT_COMPAT
    old = _TENSORRT_COMPAT
    _TENSORRT_COMPAT = bool(enabled)
    try:
        yield
    finally:
        _TENSORRT_COMPAT = old
