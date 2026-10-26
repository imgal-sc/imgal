"""
Halfplane functions.

This module provides functions for converting 2D convex hulls into
halfplanes, intersecting them and querying interior points.
"""

from _typeshed import Incomplete
from typing import Any

def edge_to_halfplane(a: Any, b: Any) -> Incomplete:
    """
    Convert the vertices of an edge into halfplane representation.

    Converts the two points defining an edge into halfplane representation.
    The outward-facing line equation is in the form `[Ny, Nx, d]`. The edge
    vertices are expected to be in `(row, col)` order.

    Args:
        a: Vertex `a` of the edge.
        b: Vertex `b` of the edge.

    Returns:
        The vector `[Ny, Nx, d]` describing the halfplane.

    Errors:
        If points `a` or `b` do not have length `2`.

    Reference:
        Preparata & Shamos, *Computational geometry an introduction* (1985)\\
        <https://doi.org/10.1007/978-1-4612-1098-6>
    """

def halfplane_intersection(halfplanes: Incomplete, interior_point: Any, threads: int | None = None) -> Incomplete:
    """
    Compute the intersection of a set of halfplanes.

    Computes the convex polygon formed by the intersection of a set of
    halfplanes. Each halfplane is represented by a row `[Ny, Nx, d]` and
    contains points satisfying `Ny * y + Nx * x + d < 0`. The interior point
    *must* lie strictly inside every halfplane. This function shifts the
    halfplanes relative to the interior point, maps them into "dual space" using
    line point duality, constructs a convex hull in dual space, and maps the
    resulting edges back into "primal space" intersection vertices.

    Args:
        halfplanes: The halfplanes with `(n_planes, 3)` shape, where each row is
            `[Ny, Nx, d]`.
        interior_point: A point with length `2` that lies strictly inside every
            halfplane and satisfies `Ny * y + Nx * x + d < 0`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the system's maximum.

    Returns:
        The vertices of the intersection polygon. The vertices have
        `(n_points, 2)` shape.

    Errors:
        If `halfplanes` is empty. If `halfplanes` axis 1 does not equal `3`. If
        the interior point length does not equal `2`.
    """

def hull_to_halfplane(vertices: Any, threads: int | None = None) -> Incomplete:
    """
    Convert the vertices of a hull into halfplane representation.

    Converts each edge of a hull into halfplane representation. Each edge is
    converted into an outward-facing line equation in the form `[Ny, Nx, d]`,
    where each row corresponds to one edge. The vertices are expected to be in
    `(row, col)` order.

    Args:
        vertices: The hull vertices with `(n_points, 2)` shape.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the system's maximum. Parallel computation returns an *unordered*
            set of halfplanes.

    Returns:
        The hull in halfplane representation where each row corresponds to one
        edge.

    Errors:
        If `vertices` is empty. If `vertices` axis 1 `!= 2`.
    """

def inside_halfplane_interior(halfplanes: Any, query: Any, include_boundary: bool, threads: int | None = None) -> bool:
    """
    Determine if a query point lies within the intersection of a set of
    halfplanes.

    Determines if the given 2D query point lies within the intersection of *all*
    the halfplanes. A point is considered inside the halfplane interior if it
    satisfies `Ny * y + Nx * x + d < 0` for all halfplanes.

    Args:
        halfplanes: The halfplanes with `(n_planes, 3)` shape, where each row is
            `[Ny, Nx, d]`.
        query: The query point to check if inside a halfplane with `(row, col)`
            order.
        include_boundary: If `true` then points on the line boundary are
            included as valid interior points. If `false` then boundary points
            are excluded.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the system's maximum.

    Returns:
        Returns `true` if `query` is inside all halfplanes, otherwise it returns
        `false`.

    Errors:
        If `halfplanes` is empty. If `halfplanes` axis 1 does not equal `3`. If
        the query point length does not equal `2`.
    """

def __getattr__(name: str) -> Incomplete: ...
