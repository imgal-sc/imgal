"""
Image filtering functions.

This module provides *n*-dimensional image filtering functions using various
techniques like convolution.
"""

from _typeshed import Incomplete
from collections.abc import Sequence

__all__ = ["fft_convolve_1d", "fft_deconvolve_1d"]

def fft_convolve_1d(data_a: Sequence[float], data_b: Sequence[float], threads: int | None = None) -> Incomplete:
    """
    Convolve two 1D signals using the Fast Fourier Transform (FFT).

    Computes the convolution of two discrete signals (`data_a` and `data_b`) by
    transforming them into the frequency domain, multiplying them, and then
    transforming the result back into a signal. This function uses "same-length"
    trimming with the first parameter `data_a`. This means that the returned
    convolution's array length will have the same length as `data_a`.

    Args:
        data_a: The first input signal to FFT convolve. Returned convolution
            arrays will be "same-length" trimmed to `data_a`'s length.
        data_b: The second input signal to FFT convolve.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
           The FFT convolved result of the same length as input signal `data_a`.
    """

def fft_deconvolve_1d(data_a: Sequence[float], data_b: Sequence[float], epsilon: float | None = None, threads: int | None = None) -> Incomplete:
    """
    Deconvolve two 1D signals using the Fast Fourier Transform (FFT).

    Computes the deconvolution of two discrete signals (`data_a` and `data_b`)
    by transforming them into the frequency domain, dividing them, and then
    transforming the result back into a signal. This function uses "same-length"
    trimming with the first parameter `data_a`. This means that the returned
    deconvolution's array length will have the same length as `data_a`.

    Args:
        data_a: The first input signal to FFT deconvolve. Returned deconvolution
            arrays will be "same-length" trimmed to `data_a`'s length.
        data_b: The second input singal to FFT deconvolve.
        epsilon: An epsilon value to prevent division by zero errors (default =
            `1e-8`).
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The FFT deconvolved result of the same length as input signal `data_a`.
    """
