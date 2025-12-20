use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use imgal::threshold;

/// Create a boolean mask from a threshold value.
///
/// Creates a threshold mask (as a boolean array) from the input image at the
/// given threshold value.
///
/// Args:
///     data: The input n-dimensional image or array.
///     threshold: The image pixel threshold value.
///
/// Returns:
///     A boolean array of the same shape as the input image with pixels that
///     are greater than the threshold value set as `true` and pixels that are
///     below the threshold value set as `false`.
#[pyfunction]
#[pyo3(name = "manual_mask")]
pub fn threshold_manual_mask<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    threshold: f64,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        return Ok(threshold::manual_mask(arr.as_array(), threshold as u8).into_pyarray(py));
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        return Ok(threshold::manual_mask(arr.as_array(), threshold as u16).into_pyarray(py));
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        return Ok(threshold::manual_mask(arr.as_array(), threshold as u64).into_pyarray(py));
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        return Ok(threshold::manual_mask(arr.as_array(), threshold as f32).into_pyarray(py));
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(threshold::manual_mask(arr.as_array(), threshold).into_pyarray(py));
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}

/// Create a boolean mask using Otsu's method.
///
/// Creates a boolean mask using Nobuyuki Otsu's automatic threshold method. The
/// Otsu threshold value used to create the mask is calculated by maximizing the
/// between-class variance of the assumed bimodal distribution in the image
/// histogram.
///
/// Args:
///     data: The input n-dimensional image or array.
///     bins: The number of bins to use to construct the image histogram for
///         Otsu's method. If `None`, then `bins = 256`.
///
/// Returns:
///     A boolean array of the same shape as the input image with pixels that
///     are greater than the computed Otsu threshold value set as `true` and
///     pixels that are below the Otsu threshold value set as `false`.
///
/// Reference:
///     <https://doi.org/10.1109/TSMC.1979.4310076>
#[pyfunction]
#[pyo3(name = "otsu_mask")]
#[pyo3(signature = (data, bins=None))]
pub fn threshold_otsu_mask<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    bins: Option<usize>,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        return Ok(threshold::otsu_mask(arr.as_array(), bins).into_pyarray(py));
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        return Ok(threshold::otsu_mask(arr.as_array(), bins).into_pyarray(py));
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        return Ok(threshold::otsu_mask(arr.as_array(), bins).into_pyarray(py));
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        return Ok(threshold::otsu_mask(arr.as_array(), bins).into_pyarray(py));
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(threshold::otsu_mask(arr.as_array(), bins).into_pyarray(py));
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64,f32, and f64.",
        ));
    }
}

/// Compute an image threshold with Otsu's method.
///
/// Calculates an image threshold value using Nobuyuki Otsu's automatic image
/// threshold method. The Otsu threshold value is calculated by maximizing the
/// between-class variance of the assumed bimodal distribution in the image
/// histogram.
///
/// Args:
///     data: The input n-dimensional image or array.
///     bins: The number of bins to use to construct the image histogram for
///         Otsu's method. If `None`, the `bins = 256`.
///
/// Returns:
///     The Otsu threshold value.
///
/// Reference:
///     <https://doi.org/10.1109/TSMC.1979.4310076>
#[pyfunction]
#[pyo3(name = "otsu_value")]
#[pyo3(signature = (data, bins=None))]
pub fn threshold_otsu_value<'py>(data: Bound<'py, PyAny>, bins: Option<usize>) -> PyResult<f64> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        return Ok(threshold::otsu_value(arr.as_array(), bins) as f64);
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        return Ok(threshold::otsu_value(arr.as_array(), bins) as f64);
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        return Ok(threshold::otsu_value(arr.as_array(), bins) as f64);
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        return Ok(threshold::otsu_value(arr.as_array(), bins) as f64);
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(threshold::otsu_value(arr.as_array(), bins));
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}
