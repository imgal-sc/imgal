use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use imgal::transform::pad;

/// Pad an n-dimensional array with a constant value.
///
/// This function pads an n-dimensional array with constant values at specified
/// axes (asymmetrical padding) or isometrically (symmetrical padding). If
/// padding is asymmetrical then the specified padded axes will increase by
/// "pad". If padding is symmetrical then each dimension increases by 2 * "pad".
///
/// :param data: An n-dimensional array.
/// :param value: The constant value to pad with.
/// :param pad: The number of contant values to pad with. If "axes" is "None"
///     then each axis increases by 2 * "pad", otherwise each axis specified in
///     "axes" increases by "pad".
/// :param axes: An array of axes specifiying which axis to pad with constant
///     values. Each axis in "axes" is extended at the end (i.e. "right" side
///     padding) by the pad amount with constant values. If "axes" is "None"
///     then the input array is padded isometrically with the given constant
///     value.
/// :return: A new constant value padded array containing the input data.
#[pyfunction]
#[pyo3(name = "constant_pad")]
#[pyo3(signature = (data, value, pad, axes=None))]
pub fn pad_constant_pad<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    value: f64,
    pad: usize,
    axes: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        return Ok(
            pad::constant_pad(arr.as_array(), value as u8, pad, axes.as_deref())
                .into_pyarray(py)
                .into_any(),
        );
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        return Ok(
            pad::constant_pad(arr.as_array(), value as u16, pad, axes.as_deref())
                .into_pyarray(py)
                .into_any(),
        );
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        return Ok(
            pad::constant_pad(arr.as_array(), value as u64, pad, axes.as_deref())
                .into_pyarray(py)
                .into_any(),
        );
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        return Ok(
            pad::constant_pad(arr.as_array(), value as f32, pad, axes.as_deref())
                .into_pyarray(py)
                .into_any(),
        );
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(
            pad::constant_pad(arr.as_array(), value, pad, axes.as_deref())
                .into_pyarray(py)
                .into_any(),
        );
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}

/// Pad an n-dimensional array with reflected values.
///
/// This function pads an n-dimensional array with reflected values at specified
/// axes (asymmetrical padding) or isometrically (symmetrical padding). If
/// padding is asymmetrical then the specified padded axes will increase by
/// "pad". If padding is symmetrical then each dimension increases by 2 * "pad".
///
/// :param data: An n-dimensional array.
/// :param pad: The number of reflected values to pad with. If "axes" is "None"
///     then each axis increases by 2 * "pad", otherwise each axis specified in
///     "axes" increases by "pad".
/// :param axes: An array of axes specifiying which axis to pad with reflected
///     values. Each axis in "axes" is extended at the end (i.e. "right" side
///     padding) by the pad amount with reflected values. If "axes" is "None"
///     then the input array is padded isometrically with reflected values.
/// :return: A new reflected value padded array containing the input data.
#[pyfunction]
#[pyo3(name = "reflect_pad")]
#[pyo3(signature = (data, pad, axes=None))]
pub fn pad_reflect_pad<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    pad: usize,
    axes: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        return Ok(pad::reflect_pad(arr.as_array(), pad, axes.as_deref())
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        return Ok(pad::reflect_pad(arr.as_array(), pad, axes.as_deref())
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        return Ok(pad::reflect_pad(arr.as_array(), pad, axes.as_deref())
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        return Ok(pad::reflect_pad(arr.as_array(), pad, axes.as_deref())
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(pad::reflect_pad(arr.as_array(), pad, axes.as_deref())
            .into_pyarray(py)
            .into_any());
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}

/// Pad an n-dimensional array with zeros.
///
/// This function pads an n-dimensional array with zeros at specified axes
/// (asymmetrical padding) or isometrically (symmetrical padding). If padding is
/// asymmetrical then the specified padded axes will increase by "pad". If
/// padding is symmetrical then each dimension increases by 2 * "pad".
///
/// :param data: An n-dimensional array.
/// :param pad: The number of zeros to pad with. If "axes" is "None" then each
///     axis increases by 2 * "pad", otherwise each axis specified in "axes"
///     increases by "pad".
/// :param axes: An array of axes specifiying which axis to pad with zeros. Each
///     axis in "axes" is extended at the end (i.e. "right" side padding) by the
///     pad amount with zeros. If "axes" is "None" then the input array is
///     padded isometrically with zeros.
/// :return: A new zero padded array containing the input data.
#[pyfunction]
#[pyo3(name = "zero_pad")]
#[pyo3(signature = (data, pad, axes=None))]
pub fn pad_zero_pad<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    pad: usize,
    axes: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        return Ok(pad::zero_pad(arr.as_array(), pad, axes.as_deref())
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        return Ok(pad::zero_pad(arr.as_array(), pad, axes.as_deref())
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        return Ok(pad::zero_pad(arr.as_array(), pad, axes.as_deref())
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        return Ok(pad::zero_pad(arr.as_array(), pad, axes.as_deref())
            .into_pyarray(py)
            .into_any());
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(pad::zero_pad(arr.as_array(), pad, axes.as_deref())
            .into_pyarray(py)
            .into_any());
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}
