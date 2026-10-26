"""
Numerical integration functions.

This module provides numerical integration functions that work on either 1D
or *n*-dimensinal data.
"""

from collections.abc import Sequence

__all__ = ["composite_simpson", "midpoint", "simpson"]

def composite_simpson(x: Sequence[float], delta_x: float | None = None, threads: int | None = None) -> float:
    """
    Integrate a curve with Simpson's 1/3 rule and the trapezoid rule.

    Approximates the definite integral using Simpson's 1/3 rule and
    the trapezoid rule (for odd number of subintervals) with pre-computed
    x-values:

    ```text
    ∫(f(x)dx) ≈ (Δx/3) * [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 2f(xₙ₋₂) + 4f(xₙ₋₁) + f(xₙ)]
    ```

    Where `n` is the number of evenly spaced points in the data. If there is an
    odd number of subintervals, the final subinterval is integrated using the
    trapezoid rule:

    ```text
    ∫(f(x)dx) ≈ (Δx/2) * [f(x₀) + f(x₁)]
    ```

    Args:
        x: The 1-dimensional data to integrate.
        delta_x: The width between data points. If `None`, then `delta_x = 1.0`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The computed integral.
    """

def midpoint(x: Sequence[float], delta_x: float | None = None, threads: int | None = None) -> float:
    """
    Integrate a curve with the midpoint rule.

    Approximates the definite integral using the midpoint rule
    with pre-computed x-values:

    ```text
    ∫f(x) dx ≈ Δx * [f(x₁) + f(x₂) + ... + f(xₙ)]
    ```

    Args:
        x: The n-dimensional array to integrate.
        delta_x: The width between data points. If `None`, then `delta_x = 1.0`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The computed integral.
    """

def simpson(x: Sequence[float], delta_x: float | None = None, threads: int | None = None) -> float:
    """
    Integrate a curve with Simpson's 1/3 rule.

    Approximates the definite integral using Simpson's 1/3 rule and
    with pre-computed x-values:

    ```text
    ∫(f(x)dx) ≈ (Δx/3) * [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 2f(xₙ₋₂) + 4f(xₙ₋₁) + f(xₙ)]
    ```

    Where `n` is the number of evenly spaced points in the data.

    Args:
        x: The 1-dimensional data to integrate with an even number of
            subintervals.
        delta_x: The width between data points. If `None`, then `delta_x = 1.0`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The computed integral.

    Errors:
        If the number of subintervals is odd.
    """
