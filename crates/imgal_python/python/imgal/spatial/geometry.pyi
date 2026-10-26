"""
Computational geometry functions.

This module provides fundamental computational geometry functions for
working with polygons and polyhedra.
"""

from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Any

def hull_centroid(vertices: Any, threads: int | None = None) -> Incomplete:
    """
    Compute the centroid of a set of vertices.

    Computes the centroid of a set of `n` vertices in `d` dimensions. The
    centroid is the *arithmetic mean* position of all vertices in the input
    array, where each row is a vertex and each column is a coordinate
    dimension in `(row, col)` dimension order.

    Args:
        vertices: The hull vertices with `(n, d)` shape.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The centroid of the hull.

    Errors:
        If `vertices` is empty.
    """

def inside_polyhedron(vertices: Any, faces: Any, center: Any, query: Any, threads: int | None = None) -> bool:
    """
    Determine if a query point is inside a polyhedron.

    Determines if a 3D query point is inside the given polyhedron's interior.
    Each face of the polyhedron is used to form a tetrahedron with the `center`
    point. The query point is considered inside the polyhedron if it is inside
    one of the constituent tetrahedra. This function expects points in
    `(pln, row, col)` order.

    Args:
        vertices: The hull vertices with `(n_points, 3)` shape.
        faces: The hull faces with `(n_triangle, 3)` shape.
        center: The center point of the polyhedron.
        query: The query point to check if inside the polyhedron.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        Returns `true` if `query` is inside the polyhedron, otherwise it returns
        `false`.

    Errors:
        If `vertices` and/or `faces` is empty. If `vertices` and/or `faces` axis
        axis 1 `!= 3`. If `center` or `query` length does not equal `3`.
    """

def inside_tetrahedron(a: Any, b: Any, c: Any, d: Any, query: Any) -> bool:
    """
    Determine if a query point is inside a tetrahedron.

    Determines if a 3D query point is inside the given tetrahedron's interior.
    The query point is considered inside the tetrahedron if the point is found
    in the interior halfspace of each face. The function expects points and
    vertices in `(pln, row, col)` order.

    Args:
        a: Vertex `a` of the oriented plane.
        b: Vertex `b` of the oriented plane.
        c: Vertex `c` of the oriented plane.
        d: The reference point relative to plane `(a, b, c)`.
        query: The query point to check if inside the polyhedron.

    Returns:
        Returns `true` if `query` is inside the tetrahedron, otherwise it
        returns `false`.

    Errors:
        If points `a`, `b`, `c`, `d` and `query` are empty or do not have length
        equal to `3`.
    """

def orient_pred_2d(o: Any, a: Any, b: Any) -> float:
    """
    Compute the 2D orientation predicate of a triangle.

    Computes the 2D orientation predicate of triangle `(o, a, b)` in
    `(row, col)` dimension order. The sign of the returned value indicates
    on which side of the oriented line `(o, a)` the point `b` lies:

    - *Positive*: Point `b` lies to the left of the directed line `o -> a`
      (counterclockwise turn).
    - *Negative*: Point `b` lies to the right of the directed line `o -> a`
      (clockwise turn).
    - *Zero*: Points `o`, `a`, and `b` are collinear.

    Args:
        o: The origin vertex of the directed line.
        a: The endpoint vertex of the directed line.
        b: The reference point relative to line `(o, a)`.

    Returns:
        The orientation of the triangle.

    Errors:
        If points `o`, `a`, or `b` are empty or do not have length equal to `3`.

    Reference:
        <https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c>
        <https://doi.org/10.1007/PL00009321>
    """

def orient_pred_3d(a: Any, b: Any, c: Any, d: Any) -> float:
    """
    Compute the 3D orientation predicate of a tetrahedron.

    Computes the 3D orientation prpedicate of tetrahedrion `(a, b, c, d)` in
    `(pln, row, col)` dimension order. The sign of the returned value indicates
    on which side of the oriented plane `(a, b, c)` the point `d` lies:

    - *Positive*: Point `d` lies above the plane where `(a, b, c)` appears in
      counterclockwise order.
    - *Negative*: Point `d` lies below the plane where `(a, b, c)` appears in
      clockwise order.
    - *Zero*: Point `d` is coplanar with plane `(a, b, c)`.

    Args:
        a: Vertex `a` of the oriented plane.
        b: Vertex `b` of the oriented plane.
        c: Vertex `c` of the oriented plane.
        d: The reference point relative to plane `(a, b, c)`.

    Returns:
        The orientation of the tetrahedron.

    Errors:
        If points `a`, `b`, `c` and `d` are empty or do not have length equal to
        `3`.

    Reference:
        <https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c>
        <https://doi.org/10.1007/PL00009321>
    """

def polyhedron_volume(vertices: Any, faces: Any, apex: Sequence[float] | None = None, threads: int | None = None) -> float:
    """
    Compute the volume of a polyhedron.

    Computes the volume of a closed polyhedron defined by `vertices` and
    `faces`. Each face is turned into a tetrahedron with the `apex` point and
    their signed volumes summed. The function expects the polyhedron (*i.e.*
    hull) to have outward-facing normals. This function expects points in
    `(pln, row, col)` order.

    Args:
        vertices: The polyhedron (hull) vertices with `(n_points, 3)` shape.
        faces: The polyhedron (hull) faces with `(n_triangle, 3)` shape.
        apex: The shared apex point of all tetrahedra. If `None`, then
            `[0, 0, 0]` is used. Using a vertex of the hull can improve
            floating-point accuracy if the hull is far from the origin.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `Some(1)` sequential execution is used. If `Some(0)`,
            then the maximum available parallelism is used. Thread counts are
            clamped to the systems maximum.

    Returns:
        The volume of the polyhedron.

    Errors:
        If `vertices` and/or `faces` is empty. If `vertices` and/or `faces`
        axis 1 `!= 3`.
    """

def tetrahedron_volume(a: Any, b: Any, c: Any, d: Any) -> float:
    """
    Compute the signed volume of a tetrahedron.

    Computes the signed volume of tetrahedron `(a, b, c, d)` in
    `(pln, row, col)` dimension order. The sign of the returned value indicates
    on which side of the oriented plane `(a, b, c)` the point `d` lies:

    - *Positive*: Point `d` lies above the plane where `(a, b, c)` appears in
      counterclockwise order.
    - *Negative*: Point `d` lies below the plane where `(a, b, c)` appears in
      clockwise order.
    - *Zero*: Point `d` is coplanar with plane `(a, b, c)`.

    Args:
        a: Vertex `a` of the oriented plane.
        b: Vertex `b` of the oriented plane.
        c: Vertex `c` of the oriented plane.
        d: The reference point relative to plane `(a, b, c)`.

    Returns:
        The signed volume of the tetrahedron. Negative signs have volumes
        pointing towards `d` and positive signs have volumes pointing away.

    Errors:
        If points `a`, `b`, `c` and `d` are empty or do not have length equal to
        `3`.

    Reference:
        <https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c>
        <https://doi.org/10.1007/PL00009321>
    """

def __getattr__(name: str) -> Incomplete: ...
