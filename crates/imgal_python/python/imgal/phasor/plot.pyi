"""
Phasor plot related functions.

This module provides time domain phasor plotting helper functions.
"""

from _typeshed import Incomplete
from collections.abc import Sequence

def gs_mask(data: Incomplete, g_coords: Sequence[float], s_coords: Sequence[float], axis: int | None = None, threads: int | None = None) -> Incomplete:
    """
    Map G and S coordinates back to the input phasor array as a boolean mask.

    Maps the G and S coordinates back to the input G/S phasor array and returns
    a 2D boolean mask where `true` indicates G and S coordiantes representing
    the `g_coords` and `s_coords` arrays.

    Args:
        data: The G/S 3D array.
        g_coords: A 1D array of `g` coordinates in the `data` array. The
            `g_coords` and `s_coords` array lengths must match.
        s_coords: A 1D array of `s` coordiantes in the `data` array. The
            `s_coords` and `g_coords` array lengths must match.
        axis: The channel axis. If `None`, then `axis = 2`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        A 2D boolean mask where `true` pixels represent values found in the
        `g_coords` and `s_coords` arrays.

    Errors:
        If `len(g_coords) != len(s_coords)`.
    """

def gs_modulation(g: float, s: float) -> float:
    """
    Compute the modulation of phasor G and S coordinates.

    Computes the modulation (M) of phasor G and S coordinates using the
    pythagorean theorem to find the hypotenuse (*i.e.* the modulation):

    ```text
    M = √(G² + S²)
    ````

    Args:
        g: The real component, G.
        s: The imaginary component, S.

    Returns:
        The modulation (M) of the (G, S) phasor coordinates.
    """

def gs_phase(g: float, s: float) -> float:
    """
    Compute the phase angle of phasor G and S coordinates.

    Computes the phase angle or phi (φ) of phasor G and S coordinates using:

    ```text
    φ = tan⁻¹(S / G)
    ````

    This implementation uses atan2 and computes the four quadrant arctangent of
    the phasor coordinates.

    Args:
        g: The real component, G.
        s: The imaginary component, S.

    Returns:
        The phase (phi, φ)  of the (G, S) phasor coordinates.
    """

def monoexponential_coords(tau: float, omega: float) -> tuple[float, float]:
    """
    Compute the G and S coordinates for a monoexponential decay.

    Computes the G and S coordinates for a monoexponential decay given as:

    ```text
    G = 1 / 1 + (ωτ)²
    S = ωτ / 1 + (ωτ)²
    ```

    Args:
        tau: The lifetime of a monoexponential decay.
        omega: The angular frequency.

    Returns:
        The monoexponential decay coordinates, (G, S).

    Reference:
        <https://doi.org/10.1117/1.JBO.25.7.071203>
    """

def __getattr__(name: str) -> Incomplete: ...
