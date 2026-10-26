"""
Image projection functions.

This module provides *n*-dimensional image projection functions.
"""

from _typeshed import Incomplete
from typing import Any

def sum_project(data: Any, axis: int | None = None, threads: int | None = None) -> Any:
    """
    Project an n-dimensional image by summing along a specified axis.

    Computes the sum projection of an n-dimensional image along the specified
    axis. Each output element is the sum of all values along the corresponding
    lane of the projection axis. The resulting image has one fewer dimension
    than the input image.

    Args:
        data: The input n-dimensional image.
        axis: The axis to sum project along. If `None` then the last axis is
            used.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The sum projected image.

    Errors:
        If `axis` is greater than or equal to the number of dimensions.
    """

def __getattr__(name: str) -> Incomplete: ...
