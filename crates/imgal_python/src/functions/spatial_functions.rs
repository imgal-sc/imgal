use std::collections::HashMap;

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::spatial::geometry::{
    hull_centroid, inside_polyhedron, inside_tetrahedron, orient_pred_2d, orient_pred_3d,
    polyhedron_volume, tetrahedron_volume,
};
use imgal::spatial::halfspace::{
    face_to_halfspace, halfspace_intersection, hull_to_halfspace, inside_halfspace_interior,
};
use imgal::spatial::{convex_hull, roi};

/// Create a convex hull from a 2D point cloud using Timothy Chan's algorithm.
///
/// Constructs a 2D convex hull from a 2D point cloud using Timothy Chan's
/// output-sensitive algorithm. The algorithm iterates with a growing guess *m*
/// for the number of hull vertices *h*. In each phase, the point cloud is
/// partitioned into groups of at most *m* points and a Graham scan is used to
/// create a set of mini-hulls. A Jarvis march is then performed starting from
/// the leftmost point. Each step queries every mini-hull for its right tangent
/// from the current hull vertex and selects the candidate making the smallest
/// clockwise turn as the next hull vertex. If the hull closes within *m* steps
/// the algorithm terminates; otherwise *m* is squared and the algorithm
/// repeats.
///
/// Args:
///     points: The 2D point cloud with shape `(n_points, 2)`.
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///     The vertices that comprise the convex hull in clockwise order.
///
/// Errors:
///     If `points.size == 0` or `len(points) < 3`.
///
/// Reference:
///     <https://en.wikipedia.org/wiki/Chan%27s_algorithm>
///     <https://doi.org/10.1007%2FBF02712873>
#[pyfunction]
#[pyo3(name = "chan_2d")]
#[pyo3(signature = (points, threads=None))]
pub fn convex_hull_chan_2d<'py>(
    py: Python<'py>,
    points: Bound<'py, PyAny>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr) = points.extract::<PyReadonlyArray2<u8>>() {
        convex_hull::chan_2d(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u16>>() {
        convex_hull::chan_2d(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u64>>() {
        convex_hull::chan_2d(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<i64>>() {
        convex_hull::chan_2d(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f32>>() {
        convex_hull::chan_2d(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f64>>() {
        convex_hull::chan_2d(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Create a convex hull from a 2D point cloud using the Graham scan method.
///
/// Constructs a 2D convex hull from a 2D point cloud using the Graham scan
/// method, where points are sorted by their polar angle relative to the pivot
/// point (the lowest and most left point). The convex hull is constructed by
/// processing these angle sorted points and retaining only those where each
/// point makes a left turn relative to the last two hull vertices.
///
/// Args:
///     points: The 2D point cloud with shape `(n_points, 2)`.
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///     The vertices that comprise the convex hull in counterclockwise order.
///
/// Errors:
///     If `points.size == 0` or `len(points) < 3`.
///
/// Reference:
///     <https://en.wikipedia.org/wiki/Graham_scan>
///     <https://doi.org/10.1016/0020-0190(72)90045-2>
#[pyfunction]
#[pyo3(name = "graham_scan")]
#[pyo3(signature = (points, threads=None))]
pub fn convex_hull_graham_scan<'py>(
    py: Python<'py>,
    points: Bound<'py, PyAny>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr) = points.extract::<PyReadonlyArray2<u8>>() {
        convex_hull::graham_scan(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u16>>() {
        convex_hull::graham_scan(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u64>>() {
        convex_hull::graham_scan(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<i64>>() {
        convex_hull::graham_scan(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f32>>() {
        convex_hull::graham_scan(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f64>>() {
        convex_hull::graham_scan(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Create a convex hull from a 2D point cloud using the Jarvis march method.
///
/// Constructs a 2D convex hull from a 2D point cloud using the Jarvis march
/// method (also known as the "gift wrapping algorithm"). The convex hull is
/// constructed by finding the most left point (col) and iterating through all
/// points in the cloud to find the smallest clockwise trun, from the current
/// position.
///
/// Args:
///     points: The 2D point cloud with shape `(n_points, 2)`.
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///     The vertices that comprise the convex hull in clockwise order.
///
/// Errors:
///     If `points.size == 0` or `len(points) < 3`.
///
/// Reference:
///     <https://en.wikipedia.org/wiki/Gift_wrapping_algorithm>
///     <https://doi.org/10.1016/0020-0190(73)90020-3>
#[pyfunction]
#[pyo3(name = "jarvis_march")]
#[pyo3(signature = (points, threads=None))]
pub fn convex_hull_jarvis_march<'py>(
    py: Python<'py>,
    points: Bound<'py, PyAny>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr) = points.extract::<PyReadonlyArray2<u8>>() {
        convex_hull::jarvis_march(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u16>>() {
        convex_hull::jarvis_march(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u64>>() {
        convex_hull::jarvis_march(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<i64>>() {
        convex_hull::jarvis_march(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f32>>() {
        convex_hull::jarvis_march(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f64>>() {
        convex_hull::jarvis_march(arr.as_array(), threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Create a convex hull from a 3D point cloud using the Quickhull method.
///
/// Constructs a 3D convex hull from a point cloud using an incremental
/// Quickhull strategy. The algorithm initializes with a tetrahedron and
/// repeatedly expands the hull with points outside the current surface until no
/// outside points remain.
///
/// Args:
///     points: The 3D point cloud with shape `(n_points, 3)`.
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///     The convex hull vertices and triangular faces. Face indices are relative
///     to the returned hull vertices.
///
/// Errors:
///     If `points.size == 0` or `len(points) < 4`.
///
/// Reference:
///     <https://en.wikipedia.org/wiki/Quickhull>
///     <https://doi.org/10.1145/235815.235821>
#[pyfunction]
#[pyo3(name = "quickhull_3d")]
#[pyo3(signature = (points, threads=None))]
pub fn convex_hull_quickhull_3d<'py>(
    py: Python<'py>,
    points: Bound<'py, PyAny>,
    threads: Option<usize>,
) -> PyResult<(Bound<'py, PyAny>, Py<PyArray2<usize>>)> {
    if let Ok(arr) = points.extract::<PyReadonlyArray2<u8>>() {
        convex_hull::quickhull_3d(arr.as_array(), threads)
            .map(|output| {
                (
                    output.0.into_pyarray(py).into_any(),
                    output.1.into_pyarray(py).unbind(),
                )
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u16>>() {
        convex_hull::quickhull_3d(arr.as_array(), threads)
            .map(|output| {
                (
                    output.0.into_pyarray(py).into_any(),
                    output.1.into_pyarray(py).unbind(),
                )
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u64>>() {
        convex_hull::quickhull_3d(arr.as_array(), threads)
            .map(|output| {
                (
                    output.0.into_pyarray(py).into_any(),
                    output.1.into_pyarray(py).unbind(),
                )
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<i64>>() {
        convex_hull::quickhull_3d(arr.as_array(), threads)
            .map(|output| {
                (
                    output.0.into_pyarray(py).into_any(),
                    output.1.into_pyarray(py).unbind(),
                )
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f32>>() {
        convex_hull::quickhull_3d(arr.as_array(), threads)
            .map(|output| {
                (
                    output.0.into_pyarray(py).into_any(),
                    output.1.into_pyarray(py).unbind(),
                )
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f64>>() {
        convex_hull::quickhull_3d(arr.as_array(), threads)
            .map(|output| {
                (
                    output.0.into_pyarray(py).into_any(),
                    output.1.into_pyarray(py).unbind(),
                )
            })
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Compute the centroid of a set of vertices.
///
/// Computes the centroid of a set of `n` vertices in `d` dimensions. The
/// centroid is the *arithmetic mean* position of all vertices in the input
/// array, where each row is a vertex and each column is a coordinate
/// dimension in `(row, col)` dimension order.
///
/// Args:
///     vertices: The hull vertices with `(n, d)` shape.
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///     The centroid of the hull.
///
/// Errors:
///     If `vertices` is empty.
#[pyfunction]
#[pyo3(name = "hull_centroid")]
#[pyo3(signature = (vertices, threads=None))]
pub fn geometry_hull_centroid<'py>(
    py: Python<'py>,
    vertices: Bound<'py, PyAny>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<u8>>() {
        hull_centroid(arr_v.as_array(), threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<u16>>() {
        hull_centroid(arr_v.as_array(), threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<u64>>() {
        hull_centroid(arr_v.as_array(), threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<i64>>() {
        hull_centroid(arr_v.as_array(), threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<f32>>() {
        hull_centroid(arr_v.as_array(), threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<f64>>() {
        hull_centroid(arr_v.as_array(), threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Determine if a query point is inside a polyhedron.
///
/// Determines if a 3D query point is inside the given polyhedron's interior.
/// Each face of the polyhedron is used to form a tetrahedron with the `center`
/// point. The query point is considered inside the polyhedron if it is inside
/// one of the constituent tetrahedra. This function expects points in
/// `(pln, row, col)` order.
///
/// Args:
///     vertices: The hull vertices with `(n_points, 3)` shape.
///     faces: The hull faces with `(n_triangle, 3)` shape.
///     center: The center point of the polyhedron.
///     query: The query point to check if inside the polyhedron.
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///     Returns `true` if `query` is inside the polyhedron, otherwise it returns
///     `false`.
///
/// Errors:
///     If `vertices` and/or `faces` is empty. If `vertices` and/or `faces` axis
///     axis 1 `!= 3`. If `center` or `query` length does not equal `3`.
#[pyfunction]
#[pyo3(name = "inside_polyhedron")]
#[pyo3(signature = (vertices, faces, center, query, threads=None))]
pub fn geometry_inside_polyhedron<'py>(
    vertices: Bound<'py, PyAny>,
    faces: Bound<'py, PyAny>,
    center: Bound<'py, PyAny>,
    query: Bound<'py, PyAny>,
    threads: Option<usize>,
) -> PyResult<bool> {
    let arr_f = faces.extract::<PyReadonlyArray2<usize>>()?;
    if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<u8>>() {
        let arr_c = center.extract::<PyReadonlyArray1<u8>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<u8>>()?;
        inside_polyhedron(
            arr_v.as_array(),
            arr_f.as_array(),
            arr_c.as_array(),
            arr_q.as_array(),
            threads,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<u16>>() {
        let arr_c = center.extract::<PyReadonlyArray1<u16>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<u16>>()?;
        inside_polyhedron(
            arr_v.as_array(),
            arr_f.as_array(),
            arr_c.as_array(),
            arr_q.as_array(),
            threads,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<u64>>() {
        let arr_c = center.extract::<PyReadonlyArray1<u64>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<u64>>()?;
        inside_polyhedron(
            arr_v.as_array(),
            arr_f.as_array(),
            arr_c.as_array(),
            arr_q.as_array(),
            threads,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<i64>>() {
        let arr_c = center.extract::<PyReadonlyArray1<i64>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<i64>>()?;
        inside_polyhedron(
            arr_v.as_array(),
            arr_f.as_array(),
            arr_c.as_array(),
            arr_q.as_array(),
            threads,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<f32>>() {
        let arr_c = center.extract::<PyReadonlyArray1<f32>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<f32>>()?;
        inside_polyhedron(
            arr_v.as_array(),
            arr_f.as_array(),
            arr_c.as_array(),
            arr_q.as_array(),
            threads,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<i64>>() {
        let arr_c = center.extract::<PyReadonlyArray1<i64>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<i64>>()?;
        inside_polyhedron(
            arr_v.as_array(),
            arr_f.as_array(),
            arr_c.as_array(),
            arr_q.as_array(),
            threads,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<f64>>() {
        let arr_c = center.extract::<PyReadonlyArray1<f64>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<f64>>()?;
        inside_polyhedron(
            arr_v.as_array(),
            arr_f.as_array(),
            arr_c.as_array(),
            arr_q.as_array(),
            threads,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Determine if a query point is inside a tetrahedron.
///
/// Determines if a 3D query point is inside the given tetrahedron's interior.
/// The query point is considered inside the tetrahedron if the point is found
/// in the interior halfspace of each face. The function expects points and
/// vertices in `(pln, row, col)` order.
///
/// Args:
///     a: Vertex `a` of the oriented plane.
///     b: Vertex `b` of the oriented plane.
///     c: Vertex `c` of the oriented plane.
///     d: The reference point relative to plane `(a, b, c)`.
///     query: The query point to check if inside the polyhedron.
///
/// Returns:
///     Returns `true` if `query` is inside the tetrahedron, otherwise it
///     returns `false`.
///
/// Errors:
///     If points `a`, `b`, `c`, `d` and `query` are empty or do not have length
///     equal to `3`.
#[pyfunction]
#[pyo3(name = "inside_tetrahedron")]
pub fn geometry_inside_tetrahedron<'py>(
    a: Bound<'py, PyAny>,
    b: Bound<'py, PyAny>,
    c: Bound<'py, PyAny>,
    d: Bound<'py, PyAny>,
    query: Bound<'py, PyAny>,
) -> PyResult<bool> {
    if let Ok(arr_a) = a.extract::<PyReadonlyArray1<u8>>() {
        let arr_b = b.extract::<PyReadonlyArray1<u8>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<u8>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<u8>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<u8>>()?;
        inside_tetrahedron(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
            arr_q.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<u16>>() {
        let arr_b = b.extract::<PyReadonlyArray1<u16>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<u16>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<u16>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<u16>>()?;
        inside_tetrahedron(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
            arr_q.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<u64>>() {
        let arr_b = b.extract::<PyReadonlyArray1<u64>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<u64>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<u64>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<u64>>()?;
        inside_tetrahedron(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
            arr_q.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<i64>>() {
        let arr_b = b.extract::<PyReadonlyArray1<i64>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<i64>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<i64>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<i64>>()?;
        inside_tetrahedron(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
            arr_q.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<f32>>() {
        let arr_b = b.extract::<PyReadonlyArray1<f32>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<f32>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<f32>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<f32>>()?;
        inside_tetrahedron(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
            arr_q.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<f64>>() {
        let arr_b = b.extract::<PyReadonlyArray1<f64>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<f64>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<f64>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<f64>>()?;
        inside_tetrahedron(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
            arr_q.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Compute the 2D orientation predicate of a triangle.
///
/// Computes the 2D orientation predicate of triangle `(o, a, b)` in
/// `(row, col)` dimension order. The sign of the returned value indicates
/// on which side of the oriented line `(o, a)` the point `b` lies:
///
/// - *Positive*: Point `b` lies to the left of the directed line `o -> a`
///   (counterclockwise turn).
/// - *Negative*: Point `b` lies to the right of the directed line `o -> a`
///   (clockwise turn).
/// - *Zero*: Points `o`, `a`, and `b` are collinear.
///
/// Args:
///     o: The origin vertex of the directed line.
///     a: The endpoint vertex of the directed line.
///     b: The reference point relative to line `(o, a)`.
///
/// Returns:
///     The orientation of the triangle.
///
/// Errors:
///     If points `o`, `a`, or `b` are empty or do not have length equal to `3`.
///
/// Reference:
///     <https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c>
///     <https://doi.org/10.1007/PL00009321>
#[pyfunction]
#[pyo3(name = "orient_pred_2d")]
pub fn geometry_orient_pred_2d<'py>(
    o: Bound<'py, PyAny>,
    a: Bound<'py, PyAny>,
    b: Bound<'py, PyAny>,
) -> PyResult<f64> {
    if let Ok(arr_o) = o.extract::<PyReadonlyArray1<u8>>() {
        let arr_a = a.extract::<PyReadonlyArray1<u8>>()?;
        let arr_b = b.extract::<PyReadonlyArray1<u8>>()?;
        orient_pred_2d(arr_o.as_array(), arr_a.as_array(), arr_b.as_array())
            .map(|output| output)
            .map_err(map_imgal_error)
    } else if let Ok(arr_o) = o.extract::<PyReadonlyArray1<u16>>() {
        let arr_a = a.extract::<PyReadonlyArray1<u16>>()?;
        let arr_b = b.extract::<PyReadonlyArray1<u16>>()?;
        orient_pred_2d(arr_o.as_array(), arr_a.as_array(), arr_b.as_array())
            .map(|output| output)
            .map_err(map_imgal_error)
    } else if let Ok(arr_o) = o.extract::<PyReadonlyArray1<u64>>() {
        let arr_a = a.extract::<PyReadonlyArray1<u64>>()?;
        let arr_b = b.extract::<PyReadonlyArray1<u64>>()?;
        orient_pred_2d(arr_o.as_array(), arr_a.as_array(), arr_b.as_array())
            .map(|output| output)
            .map_err(map_imgal_error)
    } else if let Ok(arr_o) = o.extract::<PyReadonlyArray1<i64>>() {
        let arr_a = a.extract::<PyReadonlyArray1<i64>>()?;
        let arr_b = b.extract::<PyReadonlyArray1<i64>>()?;
        orient_pred_2d(arr_o.as_array(), arr_a.as_array(), arr_b.as_array())
            .map(|output| output)
            .map_err(map_imgal_error)
    } else if let Ok(arr_o) = o.extract::<PyReadonlyArray1<f32>>() {
        let arr_a = a.extract::<PyReadonlyArray1<f32>>()?;
        let arr_b = b.extract::<PyReadonlyArray1<f32>>()?;
        orient_pred_2d(arr_o.as_array(), arr_a.as_array(), arr_b.as_array())
            .map(|output| output)
            .map_err(map_imgal_error)
    } else if let Ok(arr_o) = o.extract::<PyReadonlyArray1<f64>>() {
        let arr_a = a.extract::<PyReadonlyArray1<f64>>()?;
        let arr_b = b.extract::<PyReadonlyArray1<f64>>()?;
        orient_pred_2d(arr_o.as_array(), arr_a.as_array(), arr_b.as_array())
            .map(|output| output)
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Compute the 3D orientation predicate of a tetrahedron.
///
/// Computes the 3D orientation prpedicate of tetrahedrion `(a, b, c, d)` in
/// `(pln, row, col)` dimension order. The sign of the returned value indicates
/// on which side of the oriented plane `(a, b, c)` the point `d` lies:
///
/// - *Positive*: Point `d` lies above the plane where `(a, b, c)` appears in
///   counterclockwise order.
/// - *Negative*: Point `d` lies below the plane where `(a, b, c)` appears in
///   clockwise order.
/// - *Zero*: Point `d` is coplanar with plane `(a, b, c)`.
///
/// Args:
///     a: Vertex `a` of the oriented plane.
///     b: Vertex `b` of the oriented plane.
///     c: Vertex `c` of the oriented plane.
///     d: The reference point relative to plane `(a, b, c)`.
///
/// Returns:
///     The orientation of the tetrahedron.
///
/// Errors:
///     If points `a`, `b`, `c` and `d` are empty or do not have length equal to
///     `3`.
///
/// Reference:
///     <https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c>
///     <https://doi.org/10.1007/PL00009321>
#[pyfunction]
#[pyo3(name = "orient_pred_3d")]
pub fn geometry_orient_pred_3d<'py>(
    a: Bound<'py, PyAny>,
    b: Bound<'py, PyAny>,
    c: Bound<'py, PyAny>,
    d: Bound<'py, PyAny>,
) -> PyResult<f64> {
    if let Ok(arr_a) = a.extract::<PyReadonlyArray1<u8>>() {
        let arr_b = b.extract::<PyReadonlyArray1<u8>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<u8>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<u8>>()?;
        orient_pred_3d(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<u16>>() {
        let arr_b = b.extract::<PyReadonlyArray1<u16>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<u16>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<u16>>()?;
        orient_pred_3d(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<u64>>() {
        let arr_b = b.extract::<PyReadonlyArray1<u64>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<u64>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<u64>>()?;
        orient_pred_3d(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<i64>>() {
        let arr_b = b.extract::<PyReadonlyArray1<i64>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<i64>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<i64>>()?;
        orient_pred_3d(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<f32>>() {
        let arr_b = b.extract::<PyReadonlyArray1<f32>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<f32>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<f32>>()?;
        orient_pred_3d(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<f64>>() {
        let arr_b = b.extract::<PyReadonlyArray1<f64>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<f64>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<f64>>()?;
        orient_pred_3d(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Compute the volume of a polyhedron.
///
/// Computes the volume of a closed polyhedron defined by `vertices` and
/// `faces`. Each face is turned into a tetrahedron with the `apex` point and
/// their signed volumes summed. The function expects the polyhedron (*i.e.*
/// hull) to have outward-facing normals. This function expects points in
/// `(pln, row, col)` order.
///
/// Args:
///     vertices: The polyhedron (hull) vertices with `(n_points, 3)` shape.
///     faces: The polyhedron (hull) faces with `(n_triangle, 3)` shape.
///     apex: The shared apex point of all tetrahedra. If `None`, then
///         `[0, 0, 0]` is used. Using a vertex of the hull can improve
///         floating-point accuracy if the hull is far from the origin.
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `Some(1)` sequential execution is used. If `Some(0)`,
///         then the maximum available parallelism is used. Thread counts are
///         clamped to the systems maximum.
///
/// Returns:
///     The volume of the polyhedron.
///
/// Errors:
///     If `vertices` and/or `faces` is empty. If `vertices` and/or `faces`
///     axis 1 `!= 3`.
#[pyfunction]
#[pyo3(name = "polyhedron_volume")]
#[pyo3(signature = (vertices, faces, apex=None, threads=None))]
pub fn geometry_polyhedron_volume<'py>(
    vertices: Bound<'py, PyAny>,
    faces: Bound<'py, PyAny>,
    apex: Option<Vec<f64>>,
    threads: Option<usize>,
) -> PyResult<f64> {
    let arr_f = faces.extract::<PyReadonlyArray2<usize>>()?;
    if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<u8>>() {
        let apex = apex.map(|v| v.into_iter().map(|e| e as u8).collect::<Vec<u8>>());
        polyhedron_volume(arr_v.as_array(), arr_f.as_array(), apex.as_ref(), threads)
            .map(|output| output)
            .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<u16>>() {
        let apex = apex.map(|v| v.into_iter().map(|e| e as u16).collect::<Vec<u16>>());
        polyhedron_volume(arr_v.as_array(), arr_f.as_array(), apex.as_ref(), threads)
            .map(|output| output)
            .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<u64>>() {
        let apex = apex.map(|v| v.into_iter().map(|e| e as u64).collect::<Vec<u64>>());
        polyhedron_volume(arr_v.as_array(), arr_f.as_array(), apex.as_ref(), threads)
            .map(|output| output)
            .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<i64>>() {
        let apex = apex.map(|v| v.into_iter().map(|e| e as i64).collect::<Vec<i64>>());
        polyhedron_volume(arr_v.as_array(), arr_f.as_array(), apex.as_ref(), threads)
            .map(|output| output)
            .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<f32>>() {
        let apex = apex.map(|v| v.into_iter().map(|e| e as f32).collect::<Vec<f32>>());
        polyhedron_volume(arr_v.as_array(), arr_f.as_array(), apex.as_ref(), threads)
            .map(|output| output)
            .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<f64>>() {
        let apex = apex.map(|v| v.into_iter().map(|e| e as f64).collect::<Vec<f64>>());
        polyhedron_volume(arr_v.as_array(), arr_f.as_array(), apex.as_ref(), threads)
            .map(|output| output)
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Compute the signed volume of a tetrahedron.
///
/// Computes the signed volume of tetrahedron `(a, b, c, d)` in
/// `(pln, row, col)` dimension order. The sign of the returned value indicates
/// on which side of the oriented plane `(a, b, c)` the point `d` lies:
///
/// - *Positive*: Point `d` lies above the plane where `(a, b, c)` appears in
///   counterclockwise order.
/// - *Negative*: Point `d` lies below the plane where `(a, b, c)` appears in
///   clockwise order.
/// - *Zero*: Point `d` is coplanar with plane `(a, b, c)`.
///
/// Args:
///     a: Vertex `a` of the oriented plane.
///     b: Vertex `b` of the oriented plane.
///     c: Vertex `c` of the oriented plane.
///     d: The reference point relative to plane `(a, b, c)`.
///
/// Returns:
///     The signed volume of the tetrahedron. Negative signs have volumes
///     pointing towards `d` and positive signs have volumes pointing away.
///
/// Errors:
///     If points `a`, `b`, `c` and `d` are empty or do not have length equal to
///     `3`.
///
/// Reference:
///     <https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c>
///     <https://doi.org/10.1007/PL00009321>
#[pyfunction]
#[pyo3(name = "tetrahedron_volume")]
pub fn geometry_tetrahedron_volume<'py>(
    a: Bound<'py, PyAny>,
    b: Bound<'py, PyAny>,
    c: Bound<'py, PyAny>,
    d: Bound<'py, PyAny>,
) -> PyResult<f64> {
    if let Ok(arr_a) = a.extract::<PyReadonlyArray1<u8>>() {
        let arr_b = b.extract::<PyReadonlyArray1<u8>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<u8>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<u8>>()?;
        tetrahedron_volume(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<u16>>() {
        let arr_b = b.extract::<PyReadonlyArray1<u16>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<u16>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<u16>>()?;
        tetrahedron_volume(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<u64>>() {
        let arr_b = b.extract::<PyReadonlyArray1<u64>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<u64>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<u64>>()?;
        tetrahedron_volume(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<i64>>() {
        let arr_b = b.extract::<PyReadonlyArray1<i64>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<i64>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<i64>>()?;
        tetrahedron_volume(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<f32>>() {
        let arr_b = b.extract::<PyReadonlyArray1<f32>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<f32>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<f32>>()?;
        tetrahedron_volume(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<f64>>() {
        let arr_b = b.extract::<PyReadonlyArray1<f64>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<f64>>()?;
        let arr_d = d.extract::<PyReadonlyArray1<f64>>()?;
        tetrahedron_volume(
            arr_a.as_array(),
            arr_b.as_array(),
            arr_c.as_array(),
            arr_d.as_array(),
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Convert the vertices of a tetrahedron face into halfspace representation.
///
/// Converts the three points defining a face of a tetrahedron (*i.e.* a
/// triangle) into halfspace representation. The outward-facing plane equation
/// is in the form `[Nz, Ny, Nx, d]`. The triangle vertices are expected to be
/// in `(pln, row, col)` order.
///
/// Args:
///     a: Vertex `a` of the triangle face.
///     b: Vertex `b` of the triangle face.
///     c: Vertex `c` of the triangle face.
///
/// Returns:
///     The vector `[Nz, Ny, Nx, d]` describing the halfspace.
///
/// Errors:
///     If points `a`, `b`, or `c` do not have length `3`.
#[pyfunction]
#[pyo3(name = "face_to_halfspace")]
pub fn halfspace_face_to_halfspace<'py>(
    py: Python<'py>,
    a: Bound<'py, PyAny>,
    b: Bound<'py, PyAny>,
    c: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if let Ok(arr_a) = a.extract::<PyReadonlyArray1<u8>>() {
        let arr_b = b.extract::<PyReadonlyArray1<u8>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<u8>>()?;
        face_to_halfspace(arr_a.as_array(), arr_b.as_array(), arr_c.as_array())
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<u16>>() {
        let arr_b = b.extract::<PyReadonlyArray1<u16>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<u16>>()?;
        face_to_halfspace(arr_a.as_array(), arr_b.as_array(), arr_c.as_array())
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<u64>>() {
        let arr_b = b.extract::<PyReadonlyArray1<u64>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<u64>>()?;
        face_to_halfspace(arr_a.as_array(), arr_b.as_array(), arr_c.as_array())
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<i64>>() {
        let arr_b = b.extract::<PyReadonlyArray1<i64>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<i64>>()?;
        face_to_halfspace(arr_a.as_array(), arr_b.as_array(), arr_c.as_array())
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<f32>>() {
        let arr_b = b.extract::<PyReadonlyArray1<f32>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<f32>>()?;
        face_to_halfspace(arr_a.as_array(), arr_b.as_array(), arr_c.as_array())
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr_a) = a.extract::<PyReadonlyArray1<f64>>() {
        let arr_b = b.extract::<PyReadonlyArray1<f64>>()?;
        let arr_c = c.extract::<PyReadonlyArray1<f64>>()?;
        face_to_halfspace(arr_a.as_array(), arr_b.as_array(), arr_c.as_array())
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Compute the intersection of a set of halfspaces.
///
/// Computes the convex polyhedron formed by the intersection of a set of
/// halfspaces. Each halfspace is represented by a row `[Nz, Ny, Nx, d]` and
/// contains points satisfying `Nz * z + Ny * y + Nx * x + d < 0`. The interior
/// point *must* lie strictly inside every halfspace. This function shifts the
/// halfspaces relative to the interior point, maps them into "dual space" using
/// line point duality, constructs a convex hull in dual space, and maps the
/// resulting faces back into "primal space" intersection vertices.
///
/// Args:
///     halfspaces: The halfspaces with `(n_spaces, 4)` shape, where each row is
///         `[Nz, Ny, Nx, d]`.
///     interior_point: A point with length `3` that lies strictly inside every
///         halfspace and satisfies `Nz * z + Ny * y + Nx * x + d < 0`.
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `Some(1)` sequential execution is used. If `Some(0)`,
///         then the maximum available parallelism is used. Thread counts are
///         clamped to the systems maximum.
///
/// Returns:
///     The vertices and triangular faces of the intersection polyhedron. The
///     vertices have `(n_points, 3)` shape and the faces have
///     `(n_triangles, 3)` shape.
///
/// Errors:
///     If `halfspaces` is empty. If `halfspaces` axis 1 does not equal `4`. If
///     the interior point length does not equal `3`.
#[pyfunction]
#[pyo3(name = "halfspace_intersection")]
#[pyo3(signature = (halfspaces, interior_point, threads=None))]
pub fn halfspace_halfspace_intersection<'py>(
    py: Python<'py>,
    halfspaces: PyReadonlyArray2<f64>,
    interior_point: Bound<'py, PyAny>,
    threads: Option<usize>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<usize>>)> {
    if let Ok(arr_ip) = interior_point.extract::<PyReadonlyArray1<u8>>() {
        halfspace_intersection(halfspaces.as_array(), arr_ip.as_array(), threads)
            .map(|output| (output.0.into_pyarray(py), output.1.into_pyarray(py)))
            .map_err(map_imgal_error)
    } else if let Ok(arr_ip) = interior_point.extract::<PyReadonlyArray1<u16>>() {
        halfspace_intersection(halfspaces.as_array(), arr_ip.as_array(), threads)
            .map(|output| (output.0.into_pyarray(py), output.1.into_pyarray(py)))
            .map_err(map_imgal_error)
    } else if let Ok(arr_ip) = interior_point.extract::<PyReadonlyArray1<u64>>() {
        halfspace_intersection(halfspaces.as_array(), arr_ip.as_array(), threads)
            .map(|output| (output.0.into_pyarray(py), output.1.into_pyarray(py)))
            .map_err(map_imgal_error)
    } else if let Ok(arr_ip) = interior_point.extract::<PyReadonlyArray1<i64>>() {
        halfspace_intersection(halfspaces.as_array(), arr_ip.as_array(), threads)
            .map(|output| (output.0.into_pyarray(py), output.1.into_pyarray(py)))
            .map_err(map_imgal_error)
    } else if let Ok(arr_ip) = interior_point.extract::<PyReadonlyArray1<f32>>() {
        halfspace_intersection(halfspaces.as_array(), arr_ip.as_array(), threads)
            .map(|output| (output.0.into_pyarray(py), output.1.into_pyarray(py)))
            .map_err(map_imgal_error)
    } else if let Ok(arr_ip) = interior_point.extract::<PyReadonlyArray1<f64>>() {
        halfspace_intersection(halfspaces.as_array(), arr_ip.as_array(), threads)
            .map(|output| (output.0.into_pyarray(py), output.1.into_pyarray(py)))
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Convert the vertices and triangular faces of a hull into halfspace
/// representation.
///
/// Converts each triangular face of a hull into halfspace representation. Each
/// face is converted into an outward-facing plane equation in the form
/// `[Nz, Ny, Nx, d]`, where each row corresponds to one face. The vertices are
/// expected to be in `(pln, row, col)` order.
///
/// Args:
///     vertices: The hull vertices with `(n_points, 3)` shape.
///     faces: The hull faces with `(n_triangle, 3)` shape.
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `Some(1)` sequential execution is used. If `Some(0)`,
///         then the maximum available parallelism is used. Thread counts are
///         clamped to the systems maximum. Parallel computation returns an
///         *unordered* set of halfspaces.
///
/// Returns:
///     The hull in halfspace representation where each ro corresponds to one
///     face.
///
/// Errors:
///     If `vertices` and/or `faces` is empty. If `vertices` and/or `faces`
///     axis 1 `!= 3`.
#[pyfunction]
#[pyo3(name = "hull_to_halfspace")]
#[pyo3(signature = (vertices, faces, threads=None))]
pub fn halfspace_hull_to_halfspace<'py>(
    py: Python<'py>,
    vertices: Bound<'py, PyAny>,
    faces: Bound<'py, PyAny>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr_f = faces.extract::<PyReadonlyArray2<usize>>()?;
    if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<u8>>() {
        hull_to_halfspace(arr_v.as_array(), arr_f.as_array(), threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<u16>>() {
        hull_to_halfspace(arr_v.as_array(), arr_f.as_array(), threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<u64>>() {
        hull_to_halfspace(arr_v.as_array(), arr_f.as_array(), threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<i64>>() {
        hull_to_halfspace(arr_v.as_array(), arr_f.as_array(), threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<f32>>() {
        hull_to_halfspace(arr_v.as_array(), arr_f.as_array(), threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<f64>>() {
        hull_to_halfspace(arr_v.as_array(), arr_f.as_array(), threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Determine if a query point lies within the intersection of a set of
/// halfspaces.
///
/// Determines if the given 3D query point lies within the intersection of *all*
/// the halfspaces. A point is considered inside the halfspace interior if it
/// satisfies `Nz * z + Ny * y + Nx * x + d < 0` for all halfspaces.
///
/// Args:
///     halfspaces: The halfspaces with `(n_spaces, 4)` shape, where each row is
///         `[Nz, Ny, Nx, d]`.
///     query: The query point to check if inside a halfspace with
///         `(pln, row, col)` order.
///     include_boundary: If `true` then points on the face boundary are
///         included as valid interior points. If `false` then boundary points
///         are excluded.
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///     Returns `true` if `query` is inside all halfspaces, otherwise it returns
///     `false`.
///
/// Errors:
///     If `halfspaces` is empty. If `halfspaces` axis 1 does not equal `4`. If
///     the query point length does not equal `3`.
#[pyfunction]
#[pyo3(name = "inside_halfspace_interior")]
#[pyo3(signature = (halfspaces, query, include_boundary, threads=None))]
pub fn halfspace_inside_halfspace_inerior<'py>(
    halfspaces: Bound<'py, PyAny>,
    query: Bound<'py, PyAny>,
    include_boundary: bool,
    threads: Option<usize>,
) -> PyResult<bool> {
    let arr_h = halfspaces.extract::<PyReadonlyArray2<f64>>()?;
    if let Ok(arr_q) = query.extract::<PyReadonlyArray1<u8>>() {
        inside_halfspace_interior(
            arr_h.as_array(),
            arr_q.as_array(),
            include_boundary,
            threads,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_q) = query.extract::<PyReadonlyArray1<u16>>() {
        inside_halfspace_interior(
            arr_h.as_array(),
            arr_q.as_array(),
            include_boundary,
            threads,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_q) = query.extract::<PyReadonlyArray1<u64>>() {
        inside_halfspace_interior(
            arr_h.as_array(),
            arr_q.as_array(),
            include_boundary,
            threads,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_q) = query.extract::<PyReadonlyArray1<i64>>() {
        inside_halfspace_interior(
            arr_h.as_array(),
            arr_q.as_array(),
            include_boundary,
            threads,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_q) = query.extract::<PyReadonlyArray1<f32>>() {
        inside_halfspace_interior(
            arr_h.as_array(),
            arr_q.as_array(),
            include_boundary,
            threads,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_q) = query.extract::<PyReadonlyArray1<f64>>() {
        inside_halfspace_interior(
            arr_h.as_array(),
            arr_q.as_array(),
            include_boundary,
            threads,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Create a ROI point cloud map from an n-dimensional label image.
///
/// Creates a region of interest (ROI) "cloud" map from an n-dimensional label
/// image. For a given input image each label is converted into a 2D array
/// representing a point cloud with shape `(p, D)`, where `p` and `D` are the
/// number of points and dimensions respectively. Each label's point cloud is
/// stored with it's associated key (*i.e.* label ID) in the output `HashMap`.
///
/// Args:
///     labels: The n-dimensional label image.
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///     A ROI `HashMap` where the keys are the ROI label IDs and values are the
///     ROI point clouds.
#[pyfunction]
#[pyo3(name = "roi_cloud_map")]
#[pyo3(signature = (labels, threads=None))]
pub fn roi_roi_cloud_map<'py>(
    py: Python<'py>,
    labels: Bound<'py, PyAny>,
    threads: Option<usize>,
) -> PyResult<HashMap<u64, Py<PyArray2<usize>>>> {
    if let Ok(arr) = labels.extract::<PyReadonlyArrayDyn<u64>>() {
        let cloud_map = roi::roi_cloud_map(arr.as_array(), threads);
        Ok(cloud_map
            .into_iter()
            .map(|(k, v)| (k, v.into_pyarray(py).unbind()))
            .collect())
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u64.",
        ))
    }
}

/// Create a ROI data map from n-dimensional data and a label image.
///
/// Creates a region of interest (ROI) "data" map from input n-dimensional data
/// and label images. For a given `data` and `labels` image pair, each
/// coordinate within every label in the label image is used to query the
/// image data. Each label's associated raw data is stored as a 1D array with
/// the label's key (*i.e.* label ID) in the output `HashMap`.
///
/// Args:
///     data: The input n-dimensional image data.
///     labels: The corresponding n-dimensional label image for `data`.
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///     A ROI `HashMap` where the keys are the ROI label IDs and the values are
///     1D arrays containing raw values from the ROI.
#[pyfunction]
#[pyo3(name = "roi_data_map")]
#[pyo3(signature = (data, labels, threads=None))]
pub fn roi_roi_data_map<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    labels: PyReadonlyArrayDyn<u64>,
    threads: Option<usize>,
) -> PyResult<HashMap<u64, Py<PyAny>>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        Ok(
            roi::roi_data_map(arr.as_array(), labels.as_array(), threads)
                .map(|output| {
                    output
                        .into_iter()
                        .map(|(k, v)| (k, v.into_pyarray(py).unbind().into_any()))
                        .collect()
                })
                .map_err(map_imgal_error)?,
        )
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        Ok(
            roi::roi_data_map(arr.as_array(), labels.as_array(), threads)
                .map(|output| {
                    output
                        .into_iter()
                        .map(|(k, v)| (k, v.into_pyarray(py).unbind().into_any()))
                        .collect()
                })
                .map_err(map_imgal_error)?,
        )
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        Ok(
            roi::roi_data_map(arr.as_array(), labels.as_array(), threads)
                .map(|output| {
                    output
                        .into_iter()
                        .map(|(k, v)| (k, v.into_pyarray(py).unbind().into_any()))
                        .collect()
                })
                .map_err(map_imgal_error)?,
        )
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i64>>() {
        Ok(
            roi::roi_data_map(arr.as_array(), labels.as_array(), threads)
                .map(|output| {
                    output
                        .into_iter()
                        .map(|(k, v)| (k, v.into_pyarray(py).unbind().into_any()))
                        .collect()
                })
                .map_err(map_imgal_error)?,
        )
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        Ok(
            roi::roi_data_map(arr.as_array(), labels.as_array(), threads)
                .map(|output| {
                    output
                        .into_iter()
                        .map(|(k, v)| (k, v.into_pyarray(py).unbind().into_any()))
                        .collect()
                })
                .map_err(map_imgal_error)?,
        )
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        Ok(
            roi::roi_data_map(arr.as_array(), labels.as_array(), threads)
                .map(|output| {
                    output
                        .into_iter()
                        .map(|(k, v)| (k, v.into_pyarray(py).unbind().into_any()))
                        .collect()
                })
                .map_err(map_imgal_error)?,
        )
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}
