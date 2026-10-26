"""
Instrument simulation functions.

This module provides instrument related simulation functions (*e.g* an
instrument response function (IRF)).
"""

from _typeshed import Incomplete

def gaussian_irf_1d(bins: int, time_range: float, irf_center: float, irf_width: float, threads: int | None = None) -> Incomplete:
    """
    Create a 1D Gaussian instrument response function (IRF).

    Creates a Gaussian IRF by converting "full width at half maximum" (FWHM)
    parameters into a normalized Gaussian distribution. The FWHM is converted to
    standard deviation using the relationship:

    ```text
    σ = FWHM / (2 × √(2 × ln(2)))
    ```

    Where `ln(2) ≈ 0.693147` is the natural logarithm of `2`.

    Args:
        bins: The number of discrete points to sample the Gaussian distribution.
        time_range: The total time range over which to simulate the IRF.
        irf_center: The temporal position of the IRF peak within the time range.
        irf_width: The full width at half maximum (FWHM) of the IRF.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The simulated 1D IRF curve array.
    """

def __getattr__(name: str) -> Incomplete: ...
