use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use imgal::copy;

/// Duplicate an array.
///
/// Duplicates a given array by allocating a new array and copying elements into
/// it.
///
/// Args:
///     data: The input n-dimensional array to duplicate.
///     parallel: If `true`, parallel copying of the inpu tdata is used across
///         multiple threads. If `false`, sequential single-threaded copying is
///         used.
///
/// Returns:
///     A duplicate of the input array.
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
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ));
    }
}
