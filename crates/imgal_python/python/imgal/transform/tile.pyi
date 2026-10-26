"""
Image tiling functions.

This module provides *n*-dimensional image tiling functions.
"""

from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Any

def div_tile(data: Any, div: int, threads: int | None = None) -> list[Any]:
    """
    Tile an *n*-dimensional image using division tiling.

    Divides an *n*-dimensional image into a stack of array views representing
    tiles created from the input array. Each axis of the input array is divided
    by `div` into equally sized segments if `div` is a multiple of the length of
    the axis to be sliced. If `div` is *not* a muliple of the axis length then
    the remainder is added to the last tile's shape. This produces a total
    of `divⁿ` tiles for a given array, where `n` is the total number of
    dimensions. This function is *naive* in that it does not produce tiles
    intended for image fusing. Instead the tiles are simple array slices of
    the input data.

    Args:
        data: The input *n*-dimensional image to be tiled.
        div: The base number of divisions per axis. This value must be `>0`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        A list containing views of all tiles in row-major order. The length of
        the vector will be `divⁿ`, the number of tiles.

    Errors:
        If `div == 0`.
    """

def div_untile(tile_stack: Sequence[Any], div: int, shape: Sequence[int]) -> Any:
    """
    Untile a tile stack into an *n*-dimensional image.

    Reconstructs (*.i.e.* untiles) an *n*-dimensional image by assembling a
    stack of *n*-dimensional tiles as array views into a single output array of
    the given `shape`. The input `tile_stack` is assumed to contain tiles
    produced from the `div_tile` function or a similar tiling scheme where tiles
    are stored in row-major order. This function is *naive* in that it does not
    offer any border fusing strategies.

    Args:
        tile_stack: A vector containing views (*i.e.* tiles) to be reassembled
            into a single *n*-dimensional image.
        div: The base number of divisions per axis. This value must be `>0`.
        shape: The shape of the output array. Its dimensionality must match the
            dimensionality of the tiles.

    Returns:
        An *n*-dimensional image with the given `shape` containing all tiles in
        their corresponding positions.

    Errors:
        If `tile_stack` is empty. If `div == 0`. If `len(shape)` is not equal to
        the tile shape length. If expected tile shapes do not match given tile
        shapes. If the number of tiles given does not match the number of tiles
        expected.
    """

def __getattr__(name: str) -> Incomplete: ...
