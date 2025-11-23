use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use imgal::transform::pad;

/// Pad an n-dimensional array isometrically with a constant value.
///
/// This function pads an n-dimensional array isometrically (i.e equally on
/// all sides) with a constant value. The output padded array will be larger in
/// each dimension by 2 * "pad". The input data is centered within the new
/// padded array.
///
/// :param data: An n-dimensional array.
/// :param value: The constant value to pad with.
/// :param pad: The number of constant values to pad on each side of every axis.
///     Each axis increases by 2 * "pad".
/// :return: A new array containing the input data centered in a
///     constant value padded array.
#[pyfunction]
#[pyo3(name = "isometric_pad_constant")]
pub fn pad_isometric_pad_constant<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    value: f64,
    pad: usize,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        return Ok(pad::isometric_pad_constant(arr.as_array(), value as u8, pad)
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        return Ok(pad::isometric_pad_constant(arr.as_array(), value as u16, pad)
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        return Ok(pad::isometric_pad_constant(arr.as_array(), value as u64, pad)
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        return Ok(pad::isometric_pad_constant(arr.as_array(), value as f32, pad)
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(pad::isometric_pad_constant(arr.as_array(), value, pad)
            .into_pyarray(py)
            .into_any());
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}

/// Pad an n-dimensional array isometrically with reflected values.
///
/// This function pads an n-dimensional array isometrically (_i.e._ equally on
/// all sides) with reflected/mirrored values. The output padded array will be
/// larger in each dimension by 2 * "pad". The input data is centered within the
/// new padded array.
///
/// :param data: An n-dimensional array.
/// :param pad: The number of reflected pixels to pad on each side of every
///     axis. Each axis increases by 2 * "pad".
/// :return: A new array containing the input data centered in a reflected
///     padded array.
#[pyfunction]
#[pyo3(name = "isometric_pad_reflect")]
pub fn pad_isometric_pad_reflect<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    pad: usize,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        return Ok(pad::isometric_pad_reflect(arr.as_array(), pad)
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        return Ok(pad::isometric_pad_reflect(arr.as_array(), pad)
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        return Ok(pad::isometric_pad_reflect(arr.as_array(), pad)
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        return Ok(pad::isometric_pad_reflect(arr.as_array(), pad)
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(pad::isometric_pad_reflect(arr.as_array(), pad)
            .into_pyarray(py)
            .into_any());
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}

/// Pad an n-dimensonal array isometrically with zeros.
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
#[pyo3(name = "isometric_pad_zero")]
pub fn pad_isometric_pad_zero<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    pad: usize,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        return Ok(pad::isometric_pad_zero(arr.as_array(), pad)
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        return Ok(pad::isometric_pad_zero(arr.as_array(), pad)
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        return Ok(pad::isometric_pad_zero(arr.as_array(), pad)
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        return Ok(pad::isometric_pad_zero(arr.as_array(), pad)
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(pad::isometric_pad_zero(arr.as_array(), pad)
            .into_pyarray(py)
            .into_any());
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}
