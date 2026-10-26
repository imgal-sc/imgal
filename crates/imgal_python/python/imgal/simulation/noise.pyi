"""
Noise simulation functions.

This module provides noise simulation functions for creating new or mutating
data with reproducible noise.
"""

from _typeshed import Incomplete
from typing import Any

def poisson_noise(data: Any, scale: float, seed: int | None = None, threads: int | None = None) -> Any:
    """
    Create a new n-dimensional image with Poisson noise.

    Creates a new n-dimensional image of the input data with scaled Poisson
    noise (*i.e.* shot noise) using Knuth's algorithm.

    Args:
        data: The input n-dimensonal image.
        scale: The noise scale factor. Smaller values produce noiser output,
            while larger values produce output closer to the original input.
        seed: The seed value for the pseudo-random number generator.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum. Each thread will be initialized with its own
            pseudo-random number generator and thus *can not* return
            deterministic outputs. If `false`, sequential single-threaded
            computation is used which *is* deterministic.

    Returns:
        An image of the same dimensions as the input `data`, where each element
        is a Poisson-distributed sample derived from the corresponding input
        value.

    Reference:
        <https://en.wikipedia.org/wiki/Poisson_distribution>
    """

def poisson_noise_mut(data: Any, scale: float, seed: int | None = None, threads: int | None = None) -> None:
    """
    Mutate an n-dimensional image with Poisson noise.

    Mutates an n-dimensional image with scaled Poisson noise (*i.e.* shot noise)
    using Knuth's algorithm.

    Args:
        data: The input n-dimensonal image to mutate.
        scale: The noise scale factor. Smaller values produce noiser output,
            while larger values produce output closer to the original input.
        seed: The seed value for the pseudo-random number generator.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum. Each thread will be initialized with its own
            pseudo-random number generator and thus *can not* return
            deterministic outputs. If `false`, sequential single-threaded
            computation is used which *is* deterministic.

    Reference:
        <https://en.wikipedia.org/wiki/Poisson_distribution>
    """

def __getattr__(name: str) -> Incomplete: ...
