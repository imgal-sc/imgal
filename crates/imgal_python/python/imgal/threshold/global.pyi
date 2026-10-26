"""
Global image threshold functions.

This module provides global image threshold functions for calculating
threshold values and creating masks.
"""

from _typeshed import Incomplete
from typing import Any

def otsu_mask(data: Any, bins: int | None = None, threads: int | None = None) -> Incomplete:
    """
    Create a boolean mask using Otsu's method.

    Creates a boolean mask using Nobuyuki Otsu's automatic threshold method. The
    Otsu threshold value used to create the mask is calculated by maximizing the
    between-class variance of the assumed bimodal distribution in the image
    histogram.

    Args:
        data: The input n-dimensional image.
        bins: The number of bins to use to construct the image histogram for
            Otsu's method. If `None`, then `bins = 256`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        A boolean image of the same shape as the input image with pixels that
        are greater than the computed Otsu threshold value set as `true` and
        pixels that are below the Otsu threshold value set as `false`.

    Errors:
        If `data` is empty or `bins == 0`.

    Reference:
        <https://doi.org/10.1109/TSMC.1979.4310076>
    """

def otsu_value(data: Any, bins: int | None = None, threads: int | None = None) -> float:
    """
    Compute an image threshold with Otsu's method.

    Calculates an image threshold value using Nobuyuki Otsu's automatic image
    threshold method. The Otsu threshold value is calculated by maximizing the
    between-class variance of the assumed bimodal distribution in the image
    histogram.

    Args:
        data: The input n-dimensional image.
        bins: The number of bins to use to construct the image histogram for
            Otsu's method. If `None`, the `bins = 256`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The Otsu threshold value.

    Errors:
        If `data` is empty or `bins == 0`.

    Reference:
        <https://doi.org/10.1109/TSMC.1979.4310076>
    """

def __getattr__(name: str) -> Incomplete: ...
