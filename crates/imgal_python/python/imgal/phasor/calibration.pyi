"""
Phasor calibration functions.

This module provides time domain phasor calibration functions.
"""

from _typeshed import Incomplete
from typing import Any

def calibrate_coords(g: float, s: float, modulation: float, phase: float) -> tuple[float, float]:
    """
    Calibrate a real and imaginary (G, S) coordinates.

    Calibrates the real and imaginary (*e.g.* G and S) coordinates by rotating
    and scaling by phase (φ) and modulation (M) respectively using:

    ```text
    g = M * cos(φ)
    s = M * sin(φ)
    S' = G * s + S * g
    G' = G * g - S * s
    ```

    Where G' and S' are the calibrated real and imaginary values after rotation
    and scaling.

    Args:
        g: The real component (G) to calibrate.
        s: The imaginary (S) to calibrate.
        modulation: The modulation to scale the input (G, S) coordinates.
        phase: The phase, φ angle, to rotate the input (G, S) coordinates.

    Returns:
        The calibrated coordinates, (G, S).
    """

def calibrate_gs_image(data: Any, modulation: float, phase: float, axis: int | None = None, threads: int | None = None) -> Incomplete:
    """
    Calibrate a real and imaginary (G, S) 3D phasor image.

    Calibrates an input 3D phasor image by rotating and scaling G and S
    coordinates by phase (φ) and modulation (M) respectively using:

    ```text
    g = M * cos(φ)
    s = M * sin(φ)
    G' = G * g - S * s
    S' = G * s + S * g
    ```

    Where G' and S' are the calibrated real and imaginary values after rotation
    and scaling.

    Args:
        data: The input 3D phasor image, where G and S are channels `0` and `1`
            respectively.
        modulation: The modulation to scale the input (G, S) coordinates.
        phase: The phase, φ angle, to rotate the input (G, S) coordinates.
        axis: The channel axis. If `None`, then `axis = 2`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        A 3D image with the calibrated phasor values, where calibrated G and S
        are channels `0` and `1` respectively.
    """

def calibrate_gs_image_mut(data: Incomplete, modulation: float, phase: float, axis: int | None = None, threads: int | None = None) -> None:
    """
    Calibrate a real and imaginary (G, S) 3D phasor image.

    Calibrates an input 3D phasor image by rotating and scaling G and S
    coordinates by phase (φ) and modulation (M) respectively using:

    ```text
    g = M * cos(φ)
    s = M * sin(φ)
    G' = G * g - S * s
    S' = G * s + S * g
    ```

    Where G' and S' are the calibrated real and imaginary values after rotation
    and scaling.

    Args:
        data: The input 3D phasor image, where G and S are channels `0` and `1`
            respectively.
        modulation: The modulation to scale the input (G, S) coordinates.
        phase: The phase, φ angle, to rotate the input (G, S) coordinates.
        axis: The channel axis. If `None`, then `axis = 2`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.
    """

def calibrate_gs_roi_mut(data: dict[int, Incomplete], modulation: float, phase: float, threads: int | None = None) -> None:
    """
    Calibrate a real and imaginary (G, S) coordinates of a HashMap of phasor
    arrays.

    Calibrates an input HashMap of phasor (G, S) coordinate mutable array views
    by rotating and scaling G and S coordinates by phase (φ) and modulation (M)
    respectively using:

    ```text
    g = M * cos(φ)
    s = M * sin(φ)
    G' = G * g - S * s
    S' = G * s + S * g
    ```

    Where G' and S' are the calibrated real and imaginary values after rotation
    and scaling.

    Args:
        data: A HashMap of `u64` keys and mutable array views of the phasor
            data.
        modulation: The modulation to scale the input (G, S) coordinates.
        phase: The phase, φ angle, to rotate the input (G, S) coordinates.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.
    """

def modulation_and_phase(g: float, s: float, tau: float, omega: float) -> tuple[float, float]:
    """
    Compute the modulation and phase calibration values.

    Computes the modulation and phase calibration values from theoretical
    monoexponential coordinates (computed from `tau` and `omega`) and measured
    coordinates. The output, (M, φ), are the modulation and phase values to
    calibrate with.

    Args:
        g: The measured real (G) value.
        s: The measured imaginary (S) value.
        tau: The lifetime, τ.
        omega: The angular frequency, ω.

    Returns:
        The modulation and phase calibration values, (M, φ).
    """

def __getattr__(name: str) -> Incomplete: ...
