"""
Probability distribution functions.

This module provides functions for creating normalized distributions and
computing cumulative distribution values.
"""

from _typeshed import Incomplete

__all__ = ["inverse_normal_cdf", "normalized_gaussian"]

def inverse_normal_cdf(p: float) -> float:
    """
    Compute the quantile of a probability using the inverse normal cumulative
    distribution function.

    Computes the quantile (*z-score*) corresponding to a given cumulative
    probabililty `prob` using Peter Acklam's rational approximation algorithm.
    Acklam's algorithm has a relative error of less than `1.15e-9`.

    Args:
        prob: The probability value in the range of `0.0` to `1.0`.

    Returns:
        The quantile (z-score) corresponding to the given probability `prob`.

    Errors:
        If `prob < 0.0` or `prob > 1.0`.

    Reference:
        <https://home.online.no/~pjacklam/notes/invnorm/>
    """

def normalized_gaussian(sigma: float, bins: int, width: float, center: float, threads: int | None = None) -> Incomplete:
    """
    Create a normalized Gaussian distribution over a specified range.

    Creates a discrete Gaussian distribution by sampling the continuous Gaussian
    probability density function at evenly spaced points across a given range.
    The resulting distribution is normalized so that all values sum to `1.0`.
    This function implements the Gaussian probability density function:

    ```text
    f(x) = exp(-((x - μ)² / (2σ²)))
    ```

    Where:
    - `x` is the position along the range.
    - `μ` is the center (mean).
    - `σ` is the sigma (standard deviation).

    Args:
        sigma: The standard deviation of the Gaussian distribution (*i.e.* the
            width).
        bins: The number of discrete points to sample the Gaussian distribution.
        width: The total width of the sampling range.
        center: The mean (center) of the Gaussian distribution (*i.e.* the
            peak).
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The normalized Gaussian distribution.
    """
