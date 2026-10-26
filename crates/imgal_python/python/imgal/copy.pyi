"""
Copy *n*-dimensional image data.

This module provides *n*-dimensional image copying functions where data can
be duplicated, copied into pre-existing containers (*i.e.* arrays) and
copied into 1D flat arrays.
"""

from typing import Any

__all__ = ["copy_into", "copy_into_flat", "duplicate"]

def copy_into(data_a: Any, data_b: Any, threads: int | None = None) -> None:
    """
    Copy n-dimensional image data into an exisiting array.

    Copies an input image into an exisiting array with the same shape and type.

    Args:
        data_a: The input n-dimensional image to copy data from.
        data_b: The input n-dimensional image to copy data to.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Errors:
        If `data_a.shape != data_b.shape`.
    """

def copy_into_flat(data: Any, threads: int | None = None) -> Any:
    """
    Copy an n-dimensional image into a flat 1D array.

    Copies an input n-dimensional image into a flat 1D array.

    Args:
        data: The input n-dimensional image to flatten.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        A flat 1D array of the input data.
    """

def duplicate(data: Any, threads: int | None = None) -> Any:
    """
    Duplicate an n-dimensional image.

    Duplicates a input n-dimensional image by allocating a new array and copying
    elements into it.

    Args:
        data: The input n-dimensional image to duplicate.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        A duplicate of the input image.
    """
