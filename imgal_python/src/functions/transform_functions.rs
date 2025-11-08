use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use imgal::transform::pad;

/// Pad an array isometrically with zeros.
///
/// This function pads an n-dimensional array isometrically (i.e equally on
/// all sides) with zero. The output padded array will be larger in each
/// dimension by 2 * "pad". The input data is centered within the new padded
/// array.
///
/// :param data: An n-dimensional array.
/// :param pad: The number of zeros to pad on each side of every axis. Each axis
///     increases by 2 * "pad".
/// :return: A new array containing the input data centered in a
///     zero-padded array.
#[pyfunction]
#[pyo3(name = "isometric_zero")]
pub fn pad_isometric_zero<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    pad: usize,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        return Ok(pad::isometric_zero(arr.as_array(), pad).into_pyarray(py).into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        return Ok(pad::isometric_zero(arr.as_array(), pad).into_pyarray(py).into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        return Ok(pad::isometric_zero(arr.as_array(), pad).into_pyarray(py).into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        return Ok(pad::isometric_zero(arr.as_array(), pad).into_pyarray(py).into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(pad::isometric_zero(arr.as_array(), pad).into_pyarray(py).into_any());
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}
