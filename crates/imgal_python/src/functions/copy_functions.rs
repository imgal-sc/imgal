use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use imgal::copy;

use crate::error::map_imgal_error;

/// Duplicate an n-dimensional image.
///
/// Duplicates a given n-dimensional image by allocating a new array and copying
/// elements into it.
///
/// Args:
///     data: The input n-dimensional image to duplicate.
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
///
/// Returns:
///     A duplicate of the input image.
#[pyfunction]
#[pyo3(name = "duplicate")]
#[pyo3(signature = (data, parallel=None))]
pub fn copy_duplicate<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    parallel: Option<bool>,
) -> PyResult<Bound<'py, PyAny>> {
    let parallel = parallel.unwrap_or(false);
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        Ok(copy::duplicate(arr.as_array(), parallel)
            .into_pyarray(py)
            .into_any())
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        Ok(copy::duplicate(arr.as_array(), parallel)
            .into_pyarray(py)
            .into_any())
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        Ok(copy::duplicate(arr.as_array(), parallel)
            .into_pyarray(py)
            .into_any())
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i64>>() {
        Ok(copy::duplicate(arr.as_array(), parallel)
            .into_pyarray(py)
            .into_any())
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        Ok(copy::duplicate(arr.as_array(), parallel)
            .into_pyarray(py)
            .into_any())
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        Ok(copy::duplicate(arr.as_array(), parallel)
            .into_pyarray(py)
            .into_any())
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Duplicate n-dimensional image data into an exisiting array.
///
/// Duplicates a given array into an exisiting array with the same shape and
/// type.
///
/// Args:
///     data_a: The input n-dimensional image to copy data from.
///     data_b: The input n-dimensional image to copy data to.
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
#[pyfunction]
#[pyo3(name = "duplicate_into")]
#[pyo3(signature = (data_a, data_b, parallel=None))]
pub fn copy_duplicate_into<'py>(
    data_a: Bound<'py, PyAny>,
    data_b: Bound<'py, PyAny>,
    parallel: Option<bool>,
) -> PyResult<()> {
    let parallel = parallel.unwrap_or(false);
    if let Ok(arr_a) = data_a.extract::<PyReadonlyArrayDyn<u8>>() {
        let mut arr_b = data_b.extract::<PyReadwriteArrayDyn<u8>>()?;
        copy::duplicate_into(arr_a.as_array(), arr_b.as_array_mut(), parallel)
            .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArrayDyn<u16>>() {
        let mut arr_b = data_b.extract::<PyReadwriteArrayDyn<u16>>()?;
        copy::duplicate_into(arr_a.as_array(), arr_b.as_array_mut(), parallel)
            .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArrayDyn<u64>>() {
        let mut arr_b = data_b.extract::<PyReadwriteArrayDyn<u64>>()?;
        copy::duplicate_into(arr_a.as_array(), arr_b.as_array_mut(), parallel)
            .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArrayDyn<i64>>() {
        let mut arr_b = data_b.extract::<PyReadwriteArrayDyn<i64>>()?;
        copy::duplicate_into(arr_a.as_array(), arr_b.as_array_mut(), parallel)
            .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArrayDyn<f32>>() {
        let mut arr_b = data_b.extract::<PyReadwriteArrayDyn<f32>>()?;
        copy::duplicate_into(arr_a.as_array(), arr_b.as_array_mut(), parallel)
            .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArrayDyn<f64>>() {
        let mut arr_b = data_b.extract::<PyReadwriteArrayDyn<f64>>()?;
        copy::duplicate_into(arr_a.as_array(), arr_b.as_array_mut(), parallel)
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}
