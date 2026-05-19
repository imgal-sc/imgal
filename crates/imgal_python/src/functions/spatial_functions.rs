use std::collections::HashMap;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::spatial::geometry::{inside_polyhedron, inside_tetrahedron, orient_pred_2d};
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
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
///
/// Returns:
///     The vertices that comprise the convex hull in clockwise order.
#[pyfunction]
#[pyo3(name = "chan_2d")]
#[pyo3(signature = (points, parallel=None))]
pub fn convex_hull_chan_2d<'py>(
    py: Python<'py>,
    points: Bound<'py, PyAny>,
    parallel: Option<bool>,
) -> PyResult<Bound<'py, PyAny>> {
    let parallel = parallel.unwrap_or(false);
    if let Ok(arr) = points.extract::<PyReadonlyArray2<u8>>() {
        convex_hull::chan_2d(arr.as_array(), parallel)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u16>>() {
        convex_hull::chan_2d(arr.as_array(), parallel)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u64>>() {
        convex_hull::chan_2d(arr.as_array(), parallel)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<i64>>() {
        convex_hull::chan_2d(arr.as_array(), parallel)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f32>>() {
        convex_hull::chan_2d(arr.as_array(), parallel)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f64>>() {
        convex_hull::chan_2d(arr.as_array(), parallel)
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
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
///
/// Returns:
///     The vertices that comprise the convex hull in counterclockwise order.
#[pyfunction]
#[pyo3(name = "graham_scan")]
#[pyo3(signature = (points, parallel=None))]
pub fn convex_hull_graham_scan<'py>(
    py: Python<'py>,
    points: Bound<'py, PyAny>,
    parallel: Option<bool>,
) -> PyResult<Bound<'py, PyAny>> {
    let parallel = parallel.unwrap_or(false);
    if let Ok(arr) = points.extract::<PyReadonlyArray2<u8>>() {
        convex_hull::graham_scan(arr.as_array(), parallel)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u16>>() {
        convex_hull::graham_scan(arr.as_array(), parallel)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u64>>() {
        convex_hull::graham_scan(arr.as_array(), parallel)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<i64>>() {
        convex_hull::graham_scan(arr.as_array(), parallel)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f32>>() {
        convex_hull::graham_scan(arr.as_array(), parallel)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f64>>() {
        convex_hull::graham_scan(arr.as_array(), parallel)
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
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
///
/// Returns:
///     The vertices that comprise the convex hull in clockwise order.
#[pyfunction]
#[pyo3(name = "jarvis_march")]
#[pyo3(signature = (points, parallel=None))]
pub fn convex_hull_jarvis_march<'py>(
    py: Python<'py>,
    points: Bound<'py, PyAny>,
    parallel: Option<bool>,
) -> PyResult<Bound<'py, PyAny>> {
    let parallel = parallel.unwrap_or(false);
    if let Ok(arr) = points.extract::<PyReadonlyArray2<u8>>() {
        convex_hull::jarvis_march(arr.as_array(), parallel)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u16>>() {
        convex_hull::jarvis_march(arr.as_array(), parallel)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u64>>() {
        convex_hull::jarvis_march(arr.as_array(), parallel)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<i64>>() {
        convex_hull::jarvis_march(arr.as_array(), parallel)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f32>>() {
        convex_hull::jarvis_march(arr.as_array(), parallel)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f64>>() {
        convex_hull::jarvis_march(arr.as_array(), parallel)
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
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
///
/// Returns:
///     The convex hull vertices and triangular faces. Face indices are relative
///     to the returned hull vertices.
#[pyfunction]
#[pyo3(name = "quickhull_3d")]
#[pyo3(signature = (points, parallel=None))]
pub fn convex_hull_quickhull_3d<'py>(
    py: Python<'py>,
    points: Bound<'py, PyAny>,
    parallel: Option<bool>,
) -> PyResult<(Bound<'py, PyAny>, Py<PyArray2<usize>>)> {
    let parallel = parallel.unwrap_or(false);
    if let Ok(arr) = points.extract::<PyReadonlyArray2<u8>>() {
        convex_hull::quickhull_3d(arr.as_array(), parallel)
            .map(|output| {
                (
                    output.0.into_pyarray(py).into_any(),
                    output.1.into_pyarray(py).unbind(),
                )
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u16>>() {
        convex_hull::quickhull_3d(arr.as_array(), parallel)
            .map(|output| {
                (
                    output.0.into_pyarray(py).into_any(),
                    output.1.into_pyarray(py).unbind(),
                )
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u64>>() {
        convex_hull::quickhull_3d(arr.as_array(), parallel)
            .map(|output| {
                (
                    output.0.into_pyarray(py).into_any(),
                    output.1.into_pyarray(py).unbind(),
                )
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<i64>>() {
        convex_hull::quickhull_3d(arr.as_array(), parallel)
            .map(|output| {
                (
                    output.0.into_pyarray(py).into_any(),
                    output.1.into_pyarray(py).unbind(),
                )
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f32>>() {
        convex_hull::quickhull_3d(arr.as_array(), parallel)
            .map(|output| {
                (
                    output.0.into_pyarray(py).into_any(),
                    output.1.into_pyarray(py).unbind(),
                )
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f64>>() {
        convex_hull::quickhull_3d(arr.as_array(), parallel)
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
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
///
/// Returns:
///     Returns `true` if `query` is inside the polyhedron, otherwise it returns
///     `false`.
#[pyfunction]
#[pyo3(name = "inside_polyhedron")]
#[pyo3(signature = (vertices, faces, center, query, parallel=None))]
pub fn geometry_inside_polyhedron<'py>(
    vertices: Bound<'py, PyAny>,
    faces: Bound<'py, PyAny>,
    center: Bound<'py, PyAny>,
    query: Bound<'py, PyAny>,
    parallel: Option<bool>,
) -> PyResult<bool> {
    let parallel = parallel.unwrap_or(false);
    if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<u8>>() {
        let arr_f = faces.extract::<PyReadonlyArray2<usize>>()?;
        let arr_c = center.extract::<PyReadonlyArray1<u8>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<u8>>()?;
        inside_polyhedron(
            arr_v.as_array(),
            arr_f.as_array(),
            arr_c.as_array(),
            arr_q.as_array(),
            parallel,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<u16>>() {
        let arr_f = faces.extract::<PyReadonlyArray2<usize>>()?;
        let arr_c = center.extract::<PyReadonlyArray1<u16>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<u16>>()?;
        inside_polyhedron(
            arr_v.as_array(),
            arr_f.as_array(),
            arr_c.as_array(),
            arr_q.as_array(),
            parallel,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<u64>>() {
        let arr_f = faces.extract::<PyReadonlyArray2<usize>>()?;
        let arr_c = center.extract::<PyReadonlyArray1<u64>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<u64>>()?;
        inside_polyhedron(
            arr_v.as_array(),
            arr_f.as_array(),
            arr_c.as_array(),
            arr_q.as_array(),
            parallel,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<i64>>() {
        let arr_f = faces.extract::<PyReadonlyArray2<usize>>()?;
        let arr_c = center.extract::<PyReadonlyArray1<i64>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<i64>>()?;
        inside_polyhedron(
            arr_v.as_array(),
            arr_f.as_array(),
            arr_c.as_array(),
            arr_q.as_array(),
            parallel,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<f32>>() {
        let arr_f = faces.extract::<PyReadonlyArray2<usize>>()?;
        let arr_c = center.extract::<PyReadonlyArray1<f32>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<f32>>()?;
        inside_polyhedron(
            arr_v.as_array(),
            arr_f.as_array(),
            arr_c.as_array(),
            arr_q.as_array(),
            parallel,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<i64>>() {
        let arr_f = faces.extract::<PyReadonlyArray2<usize>>()?;
        let arr_c = center.extract::<PyReadonlyArray1<i64>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<i64>>()?;
        inside_polyhedron(
            arr_v.as_array(),
            arr_f.as_array(),
            arr_c.as_array(),
            arr_q.as_array(),
            parallel,
        )
        .map(|output| output)
        .map_err(map_imgal_error)
    } else if let Ok(arr_v) = vertices.extract::<PyReadonlyArray2<f64>>() {
        let arr_f = faces.extract::<PyReadonlyArray2<usize>>()?;
        let arr_c = center.extract::<PyReadonlyArray1<f64>>()?;
        let arr_q = query.extract::<PyReadonlyArray1<f64>>()?;
        inside_polyhedron(
            arr_v.as_array(),
            arr_f.as_array(),
            arr_c.as_array(),
            arr_q.as_array(),
            parallel,
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
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
///
/// Returns:
///     A ROI `HashMap` where the keys are the ROI label IDs and values are the
///     ROI point clouds.
#[pyfunction]
#[pyo3(name = "roi_cloud_map")]
#[pyo3(signature = (labels, parallel=None))]
pub fn roi_roi_cloud_map<'py>(
    py: Python<'py>,
    labels: Bound<'py, PyAny>,
    parallel: Option<bool>,
) -> PyResult<HashMap<u64, Py<PyArray2<usize>>>> {
    let parallel = parallel.unwrap_or(false);
    if let Ok(arr) = labels.extract::<PyReadonlyArrayDyn<u64>>() {
        let cloud_map = roi::roi_cloud_map(arr.as_array(), parallel);
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
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
///
/// Returns:
///     A ROI `HashMap` where the keys are the ROI label IDs and the values are
///     1D arrays containing raw values from the ROI.
#[pyfunction]
#[pyo3(name = "roi_data_map")]
#[pyo3(signature = (data, labels, parallel=None))]
pub fn roi_roi_data_map<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    labels: PyReadonlyArrayDyn<u64>,
    parallel: Option<bool>,
) -> PyResult<HashMap<u64, Py<PyAny>>> {
    let parallel = parallel.unwrap_or(false);
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        Ok(
            roi::roi_data_map(arr.as_array(), labels.as_array(), parallel)
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
            roi::roi_data_map(arr.as_array(), labels.as_array(), parallel)
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
            roi::roi_data_map(arr.as_array(), labels.as_array(), parallel)
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
            roi::roi_data_map(arr.as_array(), labels.as_array(), parallel)
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
            roi::roi_data_map(arr.as_array(), labels.as_array(), parallel)
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
            roi::roi_data_map(arr.as_array(), labels.as_array(), parallel)
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
