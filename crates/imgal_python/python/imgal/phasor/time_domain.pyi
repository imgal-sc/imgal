"""
Phasor time domain calculation functions.

This module provides time domain phasor calculation functions.
"""

from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Any

def gs_image(data: Any, period: float, mask: Incomplete | None = None, harmonic: float | None = None, axis: int | None = None, threads: int | None = None) -> Incomplete:
    """
    Compute the real and imaginary (G, S) coordinates of a 3D decay image.

    Computes the real (G) and imaginary (S) components using normalized sine
    and cosine Fourier transforms:

    ```text
    G = ∫(I(t) * cos(nωt) * dt) / ∫(I(t) * dt)
    S = ∫(I(t) * sin(nωt) * dt) / ∫(I(t) * dt)
    ```

    Args:
        data: The input 3D decay image.
        period: The period (*i.e.* time interval).
        harmonic: The harmonic value. If `None`, then `harmonic = 1.0`.
        axis: The decay or lifetime axis. If `None`, then `axis = 2`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The real and imaginary coordinates as a 3D (ch, row, col) image, where G
        and S are indexed at `0` and `1` respectively on the *channel* axis.

    Errors:
        If `axis >= 3`.
    """

def gs_roi(data: Any, period: float, rois: dict[int, Incomplete], harmonic: float | None = None, axis: int | None = None, threads: int | None = None) -> dict[int, Incomplete]:
    """
    Compute the real and imaginary (G, S) coordinates of a HashMap of ROI point
    clouds

    Computes the real and imaginary (G, S) coordinates each ROI point cloud
    stored in a HashMap.

    ```text
    G = ∫(I(t) * cos(nωt) * dt) / ∫(I(t) * dt)
    S = ∫(I(t) * sin(nωt) * dt) / ∫(I(t) * dt)
    ```

    Args:
        data: The input decay 3D image.
        period: The period (*i.e.* time interval).
        rois: A HashMap of point clouds representing Regions of Interests
            (ROIs). 2D ROIs are expected.
        harmonic: The harmonic value. If `None`, then `harmonic = 1.0`.
        axis: The decay or lifetime axis. If `None`, then `axis = 2`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        A HashMap where the keys are the ROI labels and values are the G and S
        values computed at each point in the input ROI point cloud. Each
        computed ROI point cloud has shape `(p, 2)`, where `p` is the number of
        points.

    Errors:
        If `axis >= 3`.
    """

def imaginary_coord(data: Sequence[float], period: float, harmonic: float | None = None, threads: int | None = None) -> float:
    """
    Compute the imaginary (S) component of a 1D decay array.

    Computes the imaginary (S) component is calculated using the normalized sine
    Fourier transform:

    ```text
    S = ∫(I(t) * sin(nωt) * dt) / ∫(I(t) * dt)
    ```

    Where `n` and `ω` are harmonic and omega values respectively.

    Args:
        data: The 1D decay array.
        period: The period (*i.e.* time interval).
        harmonic: The harmonic value. If `None`, then `harmonic = 1.0`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The imaginary component, S.
    """

def real_coord(data: Sequence[float], period: float, harmonic: float | None = None, threads: int | None = None) -> float:
    """
    Compute the real (G) component of a 1D decay curve.

    Computes the real (G) component is calculated using the normalized cosine
    Fourier transform:

    ```text
    G = ∫(I(t) * cos(nωt) * dt) / ∫(I(t) * dt)
    ```

    Where `n` and `ω` are harmonic and omega values respectively.

    Args:
        data: The 1D decay array.
        period: The period, (*i.e.* time interval).
        harmonic: The harmonic value. If `None`, then `harmonic = 1.0`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The real component, G.
    """

def __getattr__(name: str) -> Incomplete: ...
