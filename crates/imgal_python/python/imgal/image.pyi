"""
Basic image processing functions.

This module provides *n*-dimensional basic image functions such as
histograms and image normalization.
"""

from _typeshed import Incomplete
from typing import Any

def histogram(data: Any, bins: int | None = None, threads: int | None = None) -> Incomplete:
    """
    Create an image histogram from an n-dimensional image.

    Creates a 1D image histogram from an n-dimensional image.

    Args:
        data: The input n-dimensional image.
        bins: The number of bins to use for the image histogram. If `None`, then
            `bins = 256`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The image histogram of the input n-dimensional image of size `bins`.
        Each element represents the count of values falling into the
        corresponding bin.

    Errors:
        If the input data array is empty or `bins == 0`.
    """

def histogram_bin_midpoint(index: int, min: float, max: float, bins: int) -> float:
    """
    Compute the histogram bin midpoint value from a bin index.

    Computes the midpoint value of an image histogram bin at the given index.
    The midpoint value is the center value of the bin range.

    Args:
        index: The histogram bin index.
        min: The minimum value of the source data used to construct the
            histogram.
        max: The maximum value of the source data used to construct the
            histogram.
        bins: The number of bins in the histogram.

    Returns:
         The midpoint bin value of the specified index.

    Errors:
         If `bins == 0`.
    """

def histogram_bin_range(index: int, min: float, max: float, bins: int) -> tuple[float, float]:
    """
    Compute the histogram bin value range from a bin index.

    Computes the start and end values (*i.e.* the range) for a specified
    histogram bin index.

    Args:
        index: The histogram bin index.
        min: The minimum value of the source data used to construct the
            histogram.
        max: The maximum value of the source data used to construct the
      histogram.
            bins: The number of bins in the histogram.

    Returns:
        A tuple containing the start and end values representing the value range
        of the specified bin index.

    Errors:
        If `bins == 0`.
    """

def percentile_normalize(data: Any, min: float, max: float, clip: bool | None = None, axis: int | None = None, epsilon: float | None = None, threads: int | None = None) -> Incomplete:
    """
    Normalize an n-dimensional image using percentile-based minimum and maximum.

    Performs percentile-based normalization of an input n-dimensional image with
    minimum and maximum percentage within the range of `0.0` to `100.0`.

    Args:
        data: The input n-dimensional image to normalize.
        min: The minimum normalization percentile in the range `0.0` to `100.0`.
        max: The maximum normalization percentile in the range `0.0` to `100.0`.
        clip: Boolean to indicate whether to clamp the normalized values to the
            range `0.0` to `100.0`. If `None`, then `clip = false`.
        axis: The axis to comppute percentiles idependently along. Each subview
            along this axis normalized with the independent percentiles. If
            `None`, then the input `data` is flattened.
        epsilon: A small positive value to avoid division by zero. If `None`,
            then `epsilon = 1e-20`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The percentile normalized n-dimensional image.

    Errors:
        If `min` and/or `max` are outside of range `0.0` to `1.0`.
    """

def __getattr__(name: str) -> Incomplete: ...
