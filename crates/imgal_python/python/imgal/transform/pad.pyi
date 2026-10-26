"""
Image padding functions.

This module provides *n*-dimensional image padding functions.
"""

from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Any

def constant_pad(data: Any, value: float, pad_config: Sequence[int], direction: int | None = None, threads: int | None = None) -> Any:
    """
    Pad an n-dimensional image with a constant value.

    Pads an n-dimensional image with a constant value symmetrically or
    asymmetrically, along each axis. Symmetric padding increases each axis
    length by `2 * pad`, where `pad` is the value specified in `pad_config` for
    that axis. Asymmetric padding increases each axis length by `pad`, adding
    the specified number of elements at the end of the axis.

    Args:
        data: The input n-dimensional image to be padded.
        value: The constant value to use for padding.
        pad_config: A slice specifying the pad width for each axis of `data`.
        direction: A `u8` value to indicate which direction to pad. There are
            three valid pad directions:
             - 0: End (right or bottom)
             - 1: Start (left or top)
             - 2: Symmetric (both sides)
            If `None`, then `direction = 2` (symmetric padding).
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        A new constant value padded image containing the input data.

    Errors:
        If `len(pad_config) != data.ndim`.
    """

def reflect_pad(data: Any, pad_config: Sequence[int], direction: int | None = None, threads: int | None = None) -> Any:
    """
    Pad an n-dimensional image with reflected values.

    Pads an n-dimensional image with reflected values symmetrically or
    asymmetrically, along each axis. Symmetric padding increases each axis
    length by `2 * pad`, where `pad` is the value specified in `pad_config` for
    that axis. Asymmetric padding increases each axis length by `pad`, adding
    the specified number of elements at the end of the axis.

    Args:
        data: The input n-dimensional image to be padded.
        pad_config: A slice specifying the pad width for each axis of `data`.
        direction: A `u8` value to indicate which direction to pad. There are
            three valid pad directions:
             - 0: End (right or bottom)
             - 1: Start (left or top)
             - 2: Symmetric (both sides)
            If `None`, then `direction = 2` (symmetric padding).
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        A new reflected value padded image containing the input data.

    Errors:
        If `len(pad_config) != data.ndim`.
    """

def zero_pad(data: Any, pad_config: Sequence[int], direction: int | None = None, threads: int | None = None) -> Any:
    """
    Pad an n-dimensional image with reflected values.

    Pads an n-dimensional image with reflected values symmetrically or
    asymmetrically, along each axis. Symmetric padding increases each axis
    length by `2 * pad`, where `pad` is the value specified in `pad_config` for
    that axis. Asymmetric padding increases each axis length by `pad`, adding
    the specified number of elements at the end of the axis.

    Args:
        data: The input n-dimensional image to be padded.
        pad_config: A slice specifying the pad width for each axis of `data`.
        direction: A `u8` value to indicate which direction to pad. There are
            three valid pad directions:
             - 0: End (right or bottom)
             - 1: Start (left or top)
             - 2: Symmetric (both sides)
            If `None`, then `direction = 2` (symmetric padding).
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        A new reflected value padded image containing the input data.

    Errors:
        If `len(pad_config) != data.ndim`.
    """

def __getattr__(name: str) -> Incomplete: ...
