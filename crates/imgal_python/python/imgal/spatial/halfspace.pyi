"""
Halfspace functions.

This module provides functions for converting 3D convex hulls into
halfplanes, intersecting them and querying interior points.
"""

from _typeshed import Incomplete
from typing import Any

def face_to_halfspace(a: Any, b: Any, c: Any) -> Incomplete:
    """
    Convert the vertices of a tetrahedron face into halfspace representation.

    Converts the three points defining a face of a tetrahedron (*i.e.* a
    triangle) into halfspace representation. The outward-facing plane equation
    is in the form `[Nz, Ny, Nx, d]`. The triangle vertices are expected to be
    in `(pln, row, col)` order.

    Args:
        a: Vertex `a` of the triangle face.
        b: Vertex `b` of the triangle face.
        c: Vertex `c` of the triangle face.

    Returns:
        The vector `[Nz, Ny, Nx, d]` describing the halfspace.

    Errors:
        If points `a`, `b`, or `c` do not have length `3`.
    """

def halfspace_intersection(halfspaces: Incomplete, interior_point: Any, threads: int | None = None) -> tuple[Incomplete, Incomplete]:
    """
    Compute the intersection of a set of halfspaces.

    Computes the convex polyhedron formed by the intersection of a set of
    halfspaces. Each halfspace is represented by a row `[Nz, Ny, Nx, d]` and
    contains points satisfying `Nz * z + Ny * y + Nx * x + d < 0`. The interior
    point *must* lie strictly inside every halfspace. This function shifts the
    halfspaces relative to the interior point, maps them into "dual space" using
    line point duality, constructs a convex hull in dual space, and maps the
    resulting faces back into "primal space" intersection vertices.

    Args:
        halfspaces: The halfspaces with `(n_spaces, 4)` shape, where each row is
            `[Nz, Ny, Nx, d]`.
        interior_point: A point with length `3` that lies strictly inside every
            halfspace and satisfies `Nz * z + Ny * y + Nx * x + d < 0`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `Some(1)` sequential execution is used. If `Some(0)`,
            then the maximum available parallelism is used. Thread counts are
            clamped to the systems maximum.

    Returns:
        The vertices and triangular faces of the intersection polyhedron. The
        vertices have `(n_points, 3)` shape and the faces have
        `(n_triangles, 3)` shape.

    Errors:
        If `halfspaces` is empty. If `halfspaces` axis 1 does not equal `4`. If
        the interior point length does not equal `3`.
    """

def hull_to_halfspace(vertices: Any, faces: Any, threads: int | None = None) -> Incomplete:
    """
    Convert the vertices and triangular faces of a hull into halfspace
    representation.

    Converts each triangular face of a hull into halfspace representation. Each
    face is converted into an outward-facing plane equation in the form
    `[Nz, Ny, Nx, d]`, where each row corresponds to one face. The vertices are
    expected to be in `(pln, row, col)` order.

    Args:
        vertices: The hull vertices with `(n_points, 3)` shape.
        faces: The hull faces with `(n_triangle, 3)` shape.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `Some(1)` sequential execution is used. If `Some(0)`,
            then the maximum available parallelism is used. Thread counts are
            clamped to the systems maximum. Parallel computation returns an
            *unordered* set of halfspaces.

    Returns:
        The hull in halfspace representation where each ro corresponds to one
        face.

    Errors:
        If `vertices` and/or `faces` is empty. If `vertices` and/or `faces`
        axis 1 `!= 3`.
    """

def inside_halfspace_interior(halfspaces: Any, query: Any, include_boundary: bool, threads: int | None = None) -> bool:
    """
    Determine if a query point lies within the intersection of a set of
    halfspaces.

    Determines if the given 3D query point lies within the intersection of *all*
    the halfspaces. A point is considered inside the halfspace interior if it
    satisfies `Nz * z + Ny * y + Nx * x + d < 0` for all halfspaces.

    Args:
        halfspaces: The halfspaces with `(n_spaces, 4)` shape, where each row is
            `[Nz, Ny, Nx, d]`.
        query: The query point to check if inside a halfspace with
            `(pln, row, col)` order.
        include_boundary: If `true` then points on the face boundary are
            included as valid interior points. If `false` then boundary points
            are excluded.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        Returns `true` if `query` is inside all halfspaces, otherwise it returns
        `false`.

    Errors:
        If `halfspaces` is empty. If `halfspaces` axis 1 does not equal `4`. If
        the query point length does not equal `3`.
    """

def __getattr__(name: str) -> Incomplete: ...
