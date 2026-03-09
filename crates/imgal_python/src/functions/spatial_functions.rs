use std::collections::HashMap;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::spatial::roi;

/// Create a ROI point cloud map from an n-dimensional label image.
///
/// Creates a region of interest (ROI) map from an n-dimensional label image.
/// For a given input image each label is converted into a 2D array representing
/// a point cloud with shape `(p, D)`, where `p` and `D` are the number of
/// points and dimensions respectively.
///
/// Args:
///     data: An n-dimensional label image of type u64.
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
///
/// Returns:
///     A ROI HashMap where the keys are the ROI labels and values are the ROI
///     point clouds.
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
