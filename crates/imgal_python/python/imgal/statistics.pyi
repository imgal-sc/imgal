"""
Sorting, correlation and other image statistics.

This module provides various image related statistics
(*e.g.* min, max, etc...), sorting and correlation functions.
"""

from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Any

__all__ = ["effective_sample_size", "kahan_sum", "linear_percentile", "max", "min", "min_max", "pearson", "sum", "weighted_kendall_tau_b", "weighted_merge_sort_mut"]

def effective_sample_size(weights: Sequence[float]) -> float:
    """
    Compute the effective sample size (ESS) of a weighted sample set.

    Computes the effective sample size (ESS) of a weighted sample set. Only the
    weights of the associated sample set are needed. The ESS is defined as:

    ```text
    ESS = (Σ wᵢ)² / Σ (wᵢ²)
    ```

    Args:
        weights: A slice of non-negative weights where each element represents
            the weight of an associated sample.

    Returns:
        The effective number of independent samples.
    """

def kahan_sum(data: Any) -> float:
    """
    Compute the sum of an n-dimensional image using Kahan compensated summation.

    Computes the Kahan sum of an n-dimensional image. The Kahan compensated
    summation algorithm corrects for floating-point rounding errors and
    precision loss at each step of the summation. To compensate for
    floating-point error an error residual is subtracted from each value per
    iteration.

    Args:
        data: The input n-dimensional image.

    Returns:
        The Kahan sum.

    Errors:
        If `data` is empty.
    """

def linear_percentile(data: Any, p: float, axis: int | None = None, epsilon: float | None = None, threads: int | None = None) -> Incomplete:
    """
    Compute the linear percentile over an n-dimensional image.

    Calculates percentiles using linear interpolation between data points. The
    computation can be performed either on the entire array (flattened) or along
    a specified axis. The linear percentile is computed using:

    ```text
    h = (n - 1) × p
    j = ⌊h⌋
    γ = h - j
    percentile = (1 - γ) × v[j] + γ × v[j+1]
    ```

    Where:
    - `n` is the array length.
    - `p` is the percentile in range `0` to `100`.
    - `v[j]` is the value at index `j`.
    - `⌊h⌋` is the floor function of `h`.

    When `γ` is close to zero (within `epsilon`), the result is simply `v[j]`,
    avoiding unnecessary interpolation.

    Args:
        data: An n-dimensional image.
        percentile: The percentile value in thae range `0.0` to `100.0`. Values
            out side this range will be clamped.
        axis: The axis to compute percentiles along. If `None`, the input `data`
            is flattened and a single percentile value is returned.
        epsilon: The tolerance value used to decide the if the fractional index
            is an integer. If `None`, then `epsilon = 1e-12`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The linear percentile of the input data. If `axis` is `None`, the result
        shape is `(1,)` and contains a single percentile value of the flattened
        input `data`. If `axis` is a valid axis value, the result has the same
        shape as `data` with `axis` removed and contains the percentiles
        calculated along `axis`.

    Errors:
        If `axis >= data.ndim`.
    """

def max(data: Any, threads: int | None = None) -> float:
    """
    Find the maximum value in an n-dimensional image.

    Iterates through all elements of an n-dimensional image to determine the
    maximum value.

    Args:
        data: The input n-dimensional image.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The maximum value in the input n-dimensional image.

    Errors:
        If `data` is empty.
    """

def min(data: Any, threads: int | None = None) -> float:
    """
    Find the minimum value in an n-dimensional image.

    Iterates through all elements of an n-dimensional image to determine the
    minimum value.

    Args:
        data: The input n-dimensional image.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The minimum value in the input n-dimensional image.

    Errors:
        If `data` is empty.
    """

def min_max(data: Any, threads: int | None = None) -> tuple[float, float]:
    """
    Find the minimum and maximum values in an n-dimensional image.

    Iterates through all elements of an n-dimensional image to determine the
    minimum and maximum values.

    Args:
        data: The input n-dimensional image.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        A tuple containing the minimum and maximum values (*i.e.* (min, max)) in
        the given n-dimensional image.

    Errors:
        If `data` is empty.
    """

def pearson(data_a: Sequence[float], data_b: Sequence[float], threads: int | None = None) -> float:
    """
    Compute the Pearson correlation coefficient between two 1D arrays.

    Computes the Pearson correlation coefficient, a measure of linear
    correlation between two sets of 1D data.

    Pearson's correlation coefficient is computed as:

    ```text
    r = Σ[(aᵢ - mean(a)) × (bᵢ - mean(b))] / √[Σ(aᵢ - mean(a))² × Σ(bᵢ - mean(b))²]
    ```

    Args:
        data_a: The first array for correlation analysis.
        data_a: The second array for correlation analysis.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        Pearson's correlatoin coefficient ranging between `-1.0` (perfect
        negative correlation), `0.0` (no correlation), and `1.0` (perfect
        positive correlation).

    Errors:
        If `len(data_a) != len(data_b)`. If `len(data_a)` or `len(data_b)` is <=
        2.
    """

def sum(data: Any, threads: int | None = None) -> float:
    """
    Compute the sum of an n-dimensional image.

    Computes the sum of numerical values in an n-dimensional image.

    Args:
        data: The input n-dimensional image.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The sum.
    """

def weighted_kendall_tau_b(data_a: Sequence[float], data_b: Sequence[float], weights: Sequence[float]) -> float:
    """
    Compute the weighted Kendall's Tau-b rank correlation coefficient.

    Calculates a weighted Kendall's Tau-b rank correlation coefficient between
    two datasets. This implementation uses a weighted merge sort to count
    discordant pairs (inversions), and applies tie corrections for both
    variables to compute the final Tau-b coefficient. Here the weighted
    observations contribute unequally to the final correlation coefficient.

    The weighted Kendall's Tau-b is calculated using:

    ```text
    τ_b = (C - D) / √[(n₀ - n₁)(n₀ - n₂)]
    ```

    Where:
    - `C` = number of weighted concordant pairs
    - `D` = number of weighted discordant pairs
    - `n₀` = total weighted pairs = `(Σwᵢ)² - Σwᵢ²`
    - `n₁` = weighted tie correction for first variable
    - `n₂` = weighted tie correction for second variable

    Args:
        data_a: The first dataset for correlation analysis.
        data_b: The second dataset for correlation analysis.
        weights: The associated weights for each observation pait. Must be the
            same length as both input datasets.

    Returns:
        The weighted Kendall's Tau-b correlation coefficient, ranging between
        `-1.0` (perfect negative correlation), `0.0` (no correlation) and `1.0`
        (perfect positive correlation).

    Errors:
        If `len(data_a) != len(data_b)`.
    """

def weighted_merge_sort_mut(data: Any, weights: Incomplete) -> float:
    """
    Sort 1D arrays of values and their associated weights.

    Performs a bottom up merge sort on the input 1D data array along with it's
    associated weights. Both the `data` and `weights` arrays are *mutated*
    during the sorting. The output of this function is a weighted inversion
    count.

    Args:
        data: The input 1D data array.
        weights: The input 1D weights array.

    Returns:
        The number of swaps needed to sort the input array.

    Errors:
        If `len(data) != len(weights)`.

    Reference:
        <https://doi.org/10.1109/TIP.2019.2909194>
    """
