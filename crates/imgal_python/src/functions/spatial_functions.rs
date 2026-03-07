use std::collections::HashMap;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

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
