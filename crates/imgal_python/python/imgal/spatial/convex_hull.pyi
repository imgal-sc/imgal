"""
2D and 3D convex hull functions.

This module provides functions for computing 2D and 3D convex hulls.
"""

from _typeshed import Incomplete
from typing import Any

def chan_2d(points: Any, threads: int | None = None) -> Any:
    """
    Create a convex hull from a 2D point cloud using Timothy Chan's algorithm.

    Constructs a 2D convex hull from a 2D point cloud using Timothy Chan's
    output-sensitive algorithm. The algorithm iterates with a growing guess *m*
    for the number of hull vertices *h*. In each phase, the point cloud is
    partitioned into groups of at most *m* points and a Graham scan is used to
    create a set of mini-hulls. A Jarvis march is then performed starting from
    the leftmost point. Each step queries every mini-hull for its right tangent
    from the current hull vertex and selects the candidate making the smallest
    clockwise turn as the next hull vertex. If the hull closes within *m* steps
    the algorithm terminates; otherwise *m* is squared and the algorithm
    repeats.

    Args:
        points: The 2D point cloud with shape `(n_points, 2)`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The vertices that comprise the convex hull in clockwise order.

    Errors:
        If `points.size == 0` or `len(points) < 3`.

    Reference:
        <https://en.wikipedia.org/wiki/Chan%27s_algorithm>
        <https://doi.org/10.1007%2FBF02712873>
    """

def graham_scan(points: Any, threads: int | None = None) -> Any:
    """
    Create a convex hull from a 2D point cloud using the Graham scan method.

    Constructs a 2D convex hull from a 2D point cloud using the Graham scan
    method, where points are sorted by their polar angle relative to the pivot
    point (the lowest and most left point). The convex hull is constructed by
    processing these angle sorted points and retaining only those where each
    point makes a left turn relative to the last two hull vertices.

    Args:
        points: The 2D point cloud with shape `(n_points, 2)`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The vertices that comprise the convex hull in counterclockwise order.

    Errors:
        If `points.size == 0` or `len(points) < 3`.

    Reference:
        <https://en.wikipedia.org/wiki/Graham_scan>
        <https://doi.org/10.1016/0020-0190(72)90045-2>
    """

def jarvis_march(points: Any, threads: int | None = None) -> Any:
    """
    Create a convex hull from a 2D point cloud using the Jarvis march method.

    Constructs a 2D convex hull from a 2D point cloud using the Jarvis march
    method (also known as the "gift wrapping algorithm"). The convex hull is
    constructed by finding the most left point (col) and iterating through all
    points in the cloud to find the smallest clockwise trun, from the current
    position.

    Args:
        points: The 2D point cloud with shape `(n_points, 2)`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The vertices that comprise the convex hull in clockwise order.

    Errors:
        If `points.size == 0` or `len(points) < 3`.

    Reference:
        <https://en.wikipedia.org/wiki/Gift_wrapping_algorithm>
        <https://doi.org/10.1016/0020-0190(73)90020-3>
    """

def quickhull_3d(points: Any, threads: int | None = None) -> tuple[Any, Incomplete]:
    """
    Create a convex hull from a 3D point cloud using the Quickhull method.

    Constructs a 3D convex hull from a point cloud using an incremental
    Quickhull strategy. The algorithm initializes with a tetrahedron and
    repeatedly expands the hull with points outside the current surface until no
    outside points remain.

    Args:
        points: The 3D point cloud with shape `(n_points, 3)`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The convex hull vertices and triangular faces. Face indices are relative
        to the returned hull vertices.

    Errors:
        If `points.size == 0` or `len(points) < 4`.

    Reference:
        <https://en.wikipedia.org/wiki/Quickhull>
        <https://doi.org/10.1145/235815.235821>
    """

def __getattr__(name: str) -> Incomplete: ...
