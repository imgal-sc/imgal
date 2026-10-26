"""
Manual image threshold functions.

This module provides manual image threshold functions for creating masks.
"""

from _typeshed import Incomplete
from typing import Any

def manual_mask(data: Any, threshold: float, threads: int | None = None) -> Incomplete:
    """
    Create a boolean mask from a threshold value.

    Creates a threshold mask (as a boolean array) from the input image at the
    given threshold value.

    Args:
        data: The input n-dimensional image.
        threshold: The image pixel threshold value.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        A boolean image of the same shape as the input image with pixels that
        are greater than the threshold value set as `true` and pixels that are
        below the threshold value set as `false`.
    """

def __getattr__(name: str) -> Incomplete: ...
