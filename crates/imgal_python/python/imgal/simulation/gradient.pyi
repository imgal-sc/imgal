"""
Image gradient simulation functions.

This module provides functions for simulating 2D and 3D gradient images.
"""

from _typeshed import Incomplete

def linear_gradient_2d(offset: int, scale: float, shape: tuple[int, int]) -> Incomplete:
    """
    Create a 2D image with a linear gradient.

    Creates a linear gradient of increasing values from the top of the array to
    the bottom along the row axis. Setting the `offset` parameter controls how
    far the gradient extends while the `scale` parameter controls the rate
    values increase per row.

    Args:
        offset: The number of rows from the top of the array that remain at
            zero.
        scale: The rate of increase per row. This value controls the steepness
            of the gradient.
        shape: The row and col shape of the gradient array.

    Returns:
        The 2D gradient image.
    """

def linear_gradient_3d(offset: int, scale: float, shape: tuple[int, int, int]) -> Incomplete:
    """
    Create a 3D image with a linear gradient.

    Creates a linear gradient of increasing values from the top of the array to
    the bottom along the pln or z axis. Setting the `offset` parameter controls
    how far the gradient extends while the `scale` parameter controls the rate
    values increase per pln.

    Args:
        offset: The number of plns from the top of the array tha tremain at
            zero.
        scale: The rate of increase per pln. This value controls the steepness
            of the gradient.
        shape: The pln, row and col shape of the gradient array.

    Returns:
        The 3D gradient image.
    """

def __getattr__(name: str) -> Incomplete: ...
