"""
Blob simulation functions.

This module provides functions for simulating *n*-dimensional blobs.
"""

from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Any

def gaussian_metaballs(centers: Any, radii: Sequence[float], intensities: Sequence[float], falloffs: Sequence[float], background: float, shape: Sequence[int], threads: int | None = None) -> Incomplete:
    """
    Create an n-dimensional Gaussian metaballs image.

    Creates a simulated n-dimensional blobs image using a variant of Jim Blinn's
    metaballs blob simulation algorithm. Metaballs are n-dimensional blob
    isosurfaces that are able to interact with each other. This function uses a
    Gaussian falloff strategy to simulate a smooth and continuous blob border
    with no sharp edges.

    Args:
        centers: A 2D array with `(p, D)`, where `p` is the number of blobs and
            `D` is the number of dimensions.
        radii: A 1D array where each element represents a blob radius.
        intensities: A 1D array where each element represents a blob intensity.
        falloffs: A 1D array where each element represents the "falloff" value
            for a given blob that controls the rate of intensity decay from the
            blob center. High values result in a more blured border effect and
            low values have a more defined border.
        background: The background intensity value for the image.
        shape: The shape of the output n-dimensional array.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        An n-dimensional image containing the metaballs blob simulation, where
        each pixel value is the *sum* of Gaussian contributions from each blob
        and the background.

    Errors:
        If the number of blobs and `len(radii)` or `len(intensities)` do not
        match.
    """

def logistic_metaballs(centers: Any, radii: Sequence[float], intensities: Sequence[float], falloffs: Sequence[float], background: float, shape: Sequence[int], threads: int | None = None) -> Incomplete:
    """
    Create an n-dimensional logistic metaballs image.

    Creates a simulated n-dimensional blobs image using a variant of Jim Blinn's
    metaballs blob simulation algorithm. Metaballs are n-dimensional blob
    isosurfaces that are able to interact with each other. This function uses a
    logistic (sigmoid) falloff function to simulate smooth and crisp blob
    borders. Logistic metaballs, unlike traditional metaballs, do not fuse
    together but instead deform against neighboring blobs.

    Args:
        centers: A 2D array with `(p, D)`, where `p` is the number of blobs and
            `D` is the number of dimensions.
        radii: A 1D array where each element represents a blob radius.
        intensities: A 1D array where each element represents a blob intensity.
        falloffs: A 1D array where each element represents the "falloff" value
            for a given blob that controls the value transition steepness from
            the center of the blob to the edge. High values result in longer
            transitions to the background, creating larger or inflated blobs.
            Low values result in short or rapid transitions to the background,
            creating crisp edges.
        background: The background intensity value for the image.
        shape: The shape of the output n-dimensional array.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        An n-dimensional image containing the metaballs blob simulation, where
        each pixel value is the *maximum* contribution of any blob at that
        position.

    Errors:
        If the number of blobs and `len(radii)` or `len(intensities)` do not
        match.
    """

def __getattr__(name: str) -> Incomplete: ...
