use std::collections::HashMap;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::spatial::convex_hull;

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
///     The 2D point cloud with shape `(n_points, 2)`.
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
///
/// Returns:
///     The vertices that comprise the convex hull in clockwise order.
#[pyfunction]
#[pyo3(name = "chan_2d")]
#[pyo3(signature = (points, parallel=None))]
pub fn spatial_chan_2d<'py>(
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
pub fn spatial_graham_scan<'py>(
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
pub fn spatial_jarvis_march<'py>(
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

/// TODO
#[pyfunction]
#[pyo3(name = "quick_hull_3d")]
#[pyo3(signature = (points, parallel=None))]
pub fn spatial_quick_hull_3d<'py>(
    py: Python<'py>,
    points: Bound<'py, PyAny>,
    parallel: Option<bool>,
) -> PyResult<(Bound<'py, PyAny>, Vec<[usize; 3]>)> {
    let parallel = parallel.unwrap_or(false);
    if let Ok(arr) = points.extract::<PyReadonlyArray2<u8>>() {
        convex_hull::quick_hull_3d(arr.as_array(), parallel)
            .map(|output| (output.0.into_pyarray(py).into_any(), output.1))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u16>>() {
        convex_hull::quick_hull_3d(arr.as_array(), parallel)
            .map(|output| (output.0.into_pyarray(py).into_any(), output.1))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<u64>>() {
        convex_hull::quick_hull_3d(arr.as_array(), parallel)
            .map(|output| (output.0.into_pyarray(py).into_any(), output.1))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<i64>>() {
        convex_hull::quick_hull_3d(arr.as_array(), parallel)
            .map(|output| (output.0.into_pyarray(py).into_any(), output.1))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f32>>() {
        convex_hull::quick_hull_3d(arr.as_array(), parallel)
            .map(|output| (output.0.into_pyarray(py).into_any(), output.1))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = points.extract::<PyReadonlyArray2<f64>>() {
        convex_hull::quick_hull_3d(arr.as_array(), parallel)
            .map(|output| (output.0.into_pyarray(py).into_any(), output.1))
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
pub fn spatial_roi_cloud_map<'py>(
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
pub fn spatial_roi_data_map<'py>(
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
