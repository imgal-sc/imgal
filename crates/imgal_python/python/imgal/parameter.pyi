"""
Microscopy and imaging related parameter functions.

This module provides functions for calculating common microscopy imaging
parameters.
"""

from _typeshed import Incomplete

def abbe_diffraction_limit(wavelength: float, na: float) -> float:
    """
    Compute the Abbe diffraction limit.

    Computes Ernst Abbe's diffraction limit for a microscope using:

    ```text
    d = wavelength / 2 * NA
    ```

    Where `NA` is the numerical aperture of the objective.

    Args:
        wavelength: The wavelength of light in nanometers.
        na: The numerical aperture.

    Returns:
        The minimum distance required between two distinct objects to resolve
        them as separate entities.
    """

def omega(period: float) -> float:
    """
    Compute the angular frequency (omega) value.

    Computes the angular frequency, omega (ω), using the following equation:

    ```text
    ω = 2π/T
    ```

    Where `T` is the period.

    Args:
        The time period.

    Returns:
        The omega (ω) value.
    """

def __getattr__(name: str) -> Incomplete: ...
