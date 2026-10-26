"""
Decay simulation functions.

This module provides functions for simulating 1D and 3D monexponential or
multiexponential decay data.
"""

from _typeshed import Incomplete
from collections.abc import Sequence

def gaussian_exponential_decay_1d(samples: int, period: float, taus: Sequence[float], fractions: Sequence[float], total_counts: float, irf_center: float, irf_width: float, threads: int | None = None) -> Incomplete:
    """
    Create a 1D Gaussian IRF convolved monoexponential or multiexponential decay
    curve.

    Creates a 1D Gaussian instrument response function (IRF) convolved with an
    ideal monoexponential or multiexponential decay curve defined as the sum of
    one or more exponential components, each characterized by a lifetime (τ) and
    fractional intensity:

    ```text
    I(t) = [Σᵢ αᵢ × exp(-t/τᵢ)] ⊗ IRF(t)
    ```

    Args:
        samples: The number of discrete points that make up the decay curve.
        period: The period (*i.e.* time interval).
        taus: An array of lifetimes. For a monoexponential decay curve use a
            single tau value and a fractional intensity of `1.0`. For a
            multiexponential decay curve use two or more tau values, matched
            with their respective fractional intensity. The `taus` and
            `fractions` arrays must have the same length. Tau values set to
            `0.0` will be skipped.
        fractions: An array of fractional intensities for each tau in the `taus`
            array. The `fractions` array must be the same length as the `taus`
            array and sum to `1.0`. Fraction values set to `0.0` will be
            skipped.
        total_counts: The total intensity count (*e.g.* photon count) of the
            decay curve.
        irf_center: The temporal position of the IRF peak within the time range.
        irf_width: The full width at half maximum (FWHM) of the IRF.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The 1D Gaussian IRF convolved monoexponential or multiexponential decay
        curve.

    Errors:
        If `len(taus) != len(fractions)`. If fractions array does not sum to
        `1.0`.
    """

def gaussian_exponential_decay_3d(samples: int, period: float, taus: Sequence[float], fractions: Sequence[float], total_counts: float, irf_center: float, irf_width: float, shape: tuple[int, int], threads: int | None = None) -> Incomplete:
    """
    Create a 3D Gaussian IRF convolved monoexponential or multiexponential decay
    curve.

    Creates a 3D Gaussian instrument response function (IRF) convolved with an
    ideal monoexponential or multiexponential decay curve defined as the sum of
    one or more exponential components, each characterized by a lifetime (τ) and
    fractional intensity:

    ```text
    I(t) = [Σᵢ αᵢ × exp(-t/τᵢ)] ⊗ IRF(t)
    ```

    Args:
        samples: The number of discrete points that make up the decay curve.
        period: The period (*i.e.* time interval).
        taus: An array of lifetimes. For a monoexponential decay curve use a
            single tau value and a fractional intensity of `1.0`. For a
            multiexponential decay curve use two or more tau values, matched
            with their respective fractional intensity. The `taus` and
            `fractions` arrays must have the same length. Tau values set to
            `0.0` will be skipped.
        fractions: An array of fractional intensities for each tau in the `taus`
            array. The `fractions` array must be the same length as the `taus`
            array and sum to `1.0`. Fraction values set to `0.0` will be
            skipped.
        total_counts: The total intensity count (*e.g.* photon count) of the
            decay curve.
        irf_center: The temporal position of the IRF peak within the time range.
        irf_width: The full width at half maximum (FWHM) of the IRF.
        shape: The row and col shape to broadcast the decay curve into.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The 3D Gaussian IRF convolved monoexponential or multiexponential decay
        curve with dimension (row, col, t).

    Errors:
        If `len(taus) != len(fractions)`. If fractions array does not sum to
        `1.0`.
    """

def ideal_exponential_decay_1d(samples: int, period: float, taus: Sequence[float], fractions: Sequence[float], total_counts: float, threads: int | None = None) -> Incomplete:
    """
    Create a 1D ideal monoexponential or multiexponential decay curve.

    Creates a 1D ideal exponential decay curve by computing the sum of one or
    more exponential components, each characterized by a lifetime (τ) and
    fractional intensity as defined by:

    ```text
    I(t) = Σᵢ αᵢ × exp(-t/τᵢ)
    ```

    Where `αᵢ` are the pre-exponential factors derived from the fractional
    intensities and lifetimes.

    Args:
        samples: The number of discrete points that make up the decay curve.
        period: The period (*i.e.* time interval).
        taus: An array of lifetimes. For a monoexponential decay curve use a
            single tau value and a fractional intensity of `1.0`. For a
            multiexponential decay curve use two or more tau values, matched
            with their respective fractional intensity. The `taus` and
            `fractions` arrays must have the same length. Tau values set to
            `0.0` will be skipped.
        fractions: An array of fractional intensities for each tau in the `taus`
            array. The `fractions` array must be the same length as the `taus`
            array and sum to `1.0`. Fraction values set to `0.0` will be
            skipped.
        total_counts: The total intensity count (*e.g.* photon count) of the
            decay curve.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The 1D monoexponential or multiexponential decay curve.

    Errors:
        If `len(taus) != len(fractions)`. If fractions array does not sum to
        `1.0`.

    Reference:
        <https://doi.org/10.1111/j.1749-6632.1969.tb56231.x>
    """

def ideal_exponential_decay_3d(samples: int, period: float, taus: Sequence[float], fractions: Sequence[float], total_counts: float, shape: tuple[int, int], threads: int | None = None) -> Incomplete:
    """
    Create a 3D ideal monoexponential or multiexponential decay curve.

    Creates a 3D ideal exponential decay curve by computing the sum of one or
    more exponential components, each characterized by a lifetime (τ) and
    fractional intensity as defined by:

    ```text
    I(t) = Σᵢ αᵢ × exp(-t/τᵢ)
    ```

    Where `αᵢ` are the pre-exponential factors derived from the fractional
    intensities and lifetimes.

    Args:
        samples: The number of discrete points that make up the decay curve.
        period: The period (*i.e.* time interval).
        taus: An array of lifetimes. For a monoexponential decay curve use a
            single tau value and a fractional intensity of `1.0`. For a
            multiexponential decay curve use two or more tau values, matched
            with their respective fractional intensity. The `taus` and
            `fractions` arrays must have the same length. Tau values set to
            `0.0` will be skipped.
        fractions: An array of fractional intensities for each tau in the `taus`
            array. The `fractions` array must be the same length as the `taus`
            array and sum to `1.0`. Fraction values set to `0.0` will be
            skipped.
        total_counts: The total intensity count (*e.g.* photon count) of the
            decay curve.
        shape: The row and col shape to broadcast the decay curve into.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The 3D monoexponential or multiexponential decay curve with dimensions
        (row, col, t).

    Errors:
        If `len(taus) != len(fractions)`. If fractions array does not sum to
        `1.0`.

    Reference:
        <https://doi.org/10.1111/j.1749-6632.1969.tb56231.x>
    """

def irf_exponential_decay_1d(irf: Sequence[float], samples: int, period: float, taus: Sequence[float], fractions: Sequence[float], total_counts: float, threads: int | None = None) -> Incomplete:
    """
    Create a 1D IRF convolved monoexponential or multiexponential decay curve.

    Creates a 1D instrument response function (IRF) convolved with an ideal
    monoexponential or multiexponential decay curve defined as the sum of one or
    more exponential components, each characterized by a lifetime (τ) and
    fractional intensity:

    ```text
    I(t) = [Σᵢ αᵢ × exp(-t/τᵢ)] ⊗ IRF(t)
    ```

    Args:
        irf: The IRF as a 1D array.
        samples: The number of discrete points that make up the decay curve.
        period: The period (*i.e.* time interval).
        taus: An array of lifetimes. For a monoexponential decay curve use a
            single tau value and a fractional intensity of `1.0`. For a
            multiexponential decay curve use two or more tau values, matched
            with their respective fractional intensity. The `taus` and
            `fractions` arrays must have the same length. Tau values set to
            `0.0` will be skipped.
        fractions: An array of fractional intensities for each tau in the `taus`
            array. The `fractions` array must be the same length as the `taus`
            array and sum to `1.0`. Fraction values set to `0.0` will be
            skipped.
        total_counts: The total intensity count (*e.g.* photon count) of the
            decay curve.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The 1D IRF convolved monoexponential or multiexponential decay curve.

    Errors:
        If `len(taus) != len(fractions)`. If fractions array does not sum to
        `1.0`.
    """

def irf_exponential_decay_3d(irf: Sequence[float], samples: int, period: float, taus: Sequence[float], fractions: Sequence[float], total_counts: float, shape: tuple[int, int], threads: int | None = None) -> Incomplete:
    """
    Create a 3D IRF convolved monoexponential or multiexponential decay curve.

    Creates a 3D instrument response function (IRF) convolved with an ideal
    monoexponential or multiexponential decay curve defined as the sum of one or
    more exponential components, each characterized by a lifetime (τ) and
    fractional intensity:

    ```text
    I(t) = [Σᵢ αᵢ × exp(-t/τᵢ)] ⊗ IRF(t)
    ```

    Args:
        irf: The IRF as a 1D array.
        samples: The number of discrete points that make up the decay curve.
        period: The period (*i.e.* time interval).
        taus: An array of lifetimes. For a monoexponential decay curve use a
            single tau value and a fractional intensity of `1.0`. For a
            multiexponential decay curve use two or more tau values, matched
            with their respective fractional intensity. The `taus` and
            `fractions` arrays must have the same length. Tau values set to
            `0.0` will be skipped.
        fractions: An array of fractional intensities for each tau in the `taus`
            array. The `fractions` array must be the same length as the `taus`
            array and sum to `1.0`. Fraction values set to `0.0` will be
            skipped.
        total_counts: The total intensity count (*e.g.* photon count) of the
            decay curve.
        shape: The row and col shape to broadcast the decay curve into.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The 3D IRF convolved monoexponential or multiexponential decay curve
        with dimensions (row, col, t).

    Errors:
        If `len(taus) != len(fractions)`. If fractions array does not sum to
        `1.0`.
    """

def __getattr__(name: str) -> Incomplete: ...
