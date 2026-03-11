use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::transform::{pad, tile};

/// Pad an n-dimensional image with a constant value.
///
/// Pads an n-dimensional image with a constant value symmetrically or
/// asymmetrically, along each axis. Symmetric padding increases each axis
/// length by `2 * pad`, where `pad` is the value specified in `pad_config` for
/// that axis. Asymmetric padding increases each axis length by `pad`, adding
/// the specified number of elements at the end of the axis.
///
/// Args:
///     data: The input n-dimensional image to be padded.
///     value: The constant value to use for padding.
///     pad_config: A slice specifying the pad width for each axis of `data`.
///     direction: A `u8` value to indicate which direction to pad. There are
///         three valid pad directions:
///          - 0: End (right or bottom)
///          - 1: Start (left or top)
///          - 2: Symmetric (both sides)
///
///     If `None`, then `direction = 2` (symmetric padding).
///
/// Returns:
///     A new constant value padded image containing the input data.
#[pyfunction]
#[pyo3(name = "constant_pad")]
#[pyo3(signature = (data, value, pad_config, direction=None))]
pub fn pad_constant_pad<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    value: f64,
    pad_config: Vec<usize>,
    direction: Option<u8>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        pad::constant_pad(arr.as_array(), value as u8, &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        pad::constant_pad(arr.as_array(), value as u16, &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        pad::constant_pad(arr.as_array(), value as u64, &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i64>>() {
        pad::constant_pad(arr.as_array(), value as i64, &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        pad::constant_pad(arr.as_array(), value as f32, &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        pad::constant_pad(arr.as_array(), value, &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Pad an n-dimensional image with reflected values.
///
/// Pads an n-dimensional image with reflected values symmetrically or
/// asymmetrically, along each axis. Symmetric padding increases each axis
/// length by `2 * pad`, where `pad` is the value specified in `pad_config` for
/// that axis. Asymmetric padding increases each axis length by `pad`, adding
/// the specified number of elements at the end of the axis.
///
/// Args:
///     data: The input n-dimensional image to be padded.
///     pad_config: A slice specifying the pad width for each axis of `data`.
///     direction: A `u8` value to indicate which direction to pad. There are
///         three valid pad directions:
///          - 0: End (right or bottom)
///          - 1: Start (left or top)
///          - 2: Symmetric (both sides)
///
///     If `None`, then `direction = 2` (symmetric padding).
///
/// Returns:
///     A new reflected value padded image containing the input data.
#[pyfunction]
#[pyo3(name = "reflect_pad")]
#[pyo3(signature = (data, pad_config, direction=None))]
pub fn pad_reflect_pad<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    pad_config: Vec<usize>,
    direction: Option<u8>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        pad::reflect_pad(arr.as_array(), &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        pad::reflect_pad(arr.as_array(), &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        pad::reflect_pad(arr.as_array(), &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i64>>() {
        pad::reflect_pad(arr.as_array(), &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        pad::reflect_pad(arr.as_array(), &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        pad::reflect_pad(arr.as_array(), &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Pad an n-dimensional image with reflected values.
///
/// Pads an n-dimensional image with reflected values symmetrically or
/// asymmetrically, along each axis. Symmetric padding increases each axis
/// length by `2 * pad`, where `pad` is the value specified in `pad_config` for
/// that axis. Asymmetric padding increases each axis length by `pad`, adding
/// the specified number of elements at the end of the axis.
///
/// Args:
///     data: The input n-dimensional image to be padded.
///     pad_config: A slice specifying the pad width for each axis of `data`.
///     direction: A `u8` value to indicate which direction to pad. There are
///         three valid pad directions:
///          - 0: End (right or bottom)
///          - 1: Start (left or top)
///          - 2: Symmetric (both sides)
///
///     If `None`, then `direction = 2` (symmetric padding).
///
/// Returns:
///     A new reflected value padded image containing the input data.
#[pyfunction]
#[pyo3(name = "zero_pad")]
#[pyo3(signature = (data, pad_config, direction=None))]
pub fn pad_zero_pad<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    pad_config: Vec<usize>,
    direction: Option<u8>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        pad::zero_pad(arr.as_array(), &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        pad::zero_pad(arr.as_array(), &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        pad::zero_pad(arr.as_array(), &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i64>>() {
        pad::zero_pad(arr.as_array(), &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        pad::zero_pad(arr.as_array(), &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        pad::zero_pad(arr.as_array(), &pad_config, direction)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Tile an n-dimensional image using division tiling.
///
/// Divides an n-dimensional image into a stack of array views representing
/// tiles created from the input array. Each axis of the input array is divided
/// by `div` into equally sized segments if `div` is a multiple of the length of
/// the axis to be sliced. If `div` is *not* a muliple of the axis length then
/// the remainder is added to the last tile's shape. This produces a total
/// of `divⁿ` tiles for a given array, where `n` is the total number of
/// dimensions. This function is *naive* in that it does not produce tiles
/// intended for image fusing. Instead the tiles are simple array slices of
/// the input data.
///
/// Args:
///     data: The input n-dimensional image to be tiled.
///     div: The base number of divisions per axis. This value must be `>0`.
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
///
/// Returns:
///     A list containing views of all tiles in row-major order. The length of
///     the vector will be `divⁿ`, the number of tiles.
#[pyfunction]
#[pyo3(name = "div_tile")]
#[pyo3(signature = (data, div, parallel=None))]
pub fn tile_div_tile<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    div: usize,
    parallel: Option<bool>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let parallel = parallel.unwrap_or(false);
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        tile::div_tile(arr.as_array(), div, parallel)
            .map(|output| {
                output
                    .iter()
                    .map(|v| v.to_owned().into_pyarray(py).into_any())
                    .collect()
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        tile::div_tile(arr.as_array(), div, parallel)
            .map(|output| {
                output
                    .iter()
                    .map(|v| v.to_owned().into_pyarray(py).into_any())
                    .collect()
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        tile::div_tile(arr.as_array(), div, parallel)
            .map(|output| {
                output
                    .iter()
                    .map(|v| v.to_owned().into_pyarray(py).into_any())
                    .collect()
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i64>>() {
        tile::div_tile(arr.as_array(), div, parallel)
            .map(|output| {
                output
                    .iter()
                    .map(|v| v.to_owned().into_pyarray(py).into_any())
                    .collect()
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        tile::div_tile(arr.as_array(), div, parallel)
            .map(|output| {
                output
                    .iter()
                    .map(|v| v.to_owned().into_pyarray(py).into_any())
                    .collect()
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        tile::div_tile(arr.as_array(), div, parallel)
            .map(|output| {
                output
                    .iter()
                    .map(|v| v.to_owned().into_pyarray(py).into_any())
                    .collect()
            })
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Untile a tile stack into an n-dimensional image.
///
/// Reconstructs (*.i.e.* untiles) an n-dimensional image by assembling a stack
/// of n-dimensional tiles as array views into a single output array of the
/// given `shape`. The input `tile_stack` is assumed to contain tiles resulting
/// from the `div_tile` function or a similar tiling scheme where tiles are
/// stored in row-major order. This function is *naive* in that it does not
/// offer any border fusing strategies.
///
/// Args:
///     tile_stack: A vector containing views (*i.e.* tiles) to be reassembled
///         into a single n-dimensional image.
///     div: The base number of divisions per axis. This value must be `>0`.
///     shape: The shape of the output array. Its dimensionality must match the
///         dimensionality of the tiles.
///
/// Returns:
///     An n-dimensional image with the given `shape` containing all tiles in
///     their corresponding positions.
#[pyfunction]
#[pyo3(name = "div_untile")]
pub fn tile_div_untile<'py>(
    py: Python<'py>,
    tile_stack: Vec<Bound<'py, PyAny>>,
    div: usize,
    shape: Vec<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(stack) = tile_stack
        .iter()
        .map(|t| t.extract::<PyReadonlyArrayDyn<u8>>())
        .collect::<Result<Vec<_>, _>>()
    {
        let arrs = stack.iter().map(|arr| arr.as_array()).collect();
        tile::div_untile(arrs, div, &shape)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(stack) = tile_stack
        .iter()
        .map(|t| t.extract::<PyReadonlyArrayDyn<u16>>())
        .collect::<Result<Vec<_>, _>>()
    {
        let arrs = stack.iter().map(|arr| arr.as_array()).collect();
        tile::div_untile(arrs, div, &shape)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(stack) = tile_stack
        .iter()
        .map(|t| t.extract::<PyReadonlyArrayDyn<u64>>())
        .collect::<Result<Vec<_>, _>>()
    {
        let arrs = stack.iter().map(|arr| arr.as_array()).collect();
        tile::div_untile(arrs, div, &shape)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(stack) = tile_stack
        .iter()
        .map(|t| t.extract::<PyReadonlyArrayDyn<i64>>())
        .collect::<Result<Vec<_>, _>>()
    {
        let arrs = stack.iter().map(|arr| arr.as_array()).collect();
        tile::div_untile(arrs, div, &shape)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(stack) = tile_stack
        .iter()
        .map(|t| t.extract::<PyReadonlyArrayDyn<f32>>())
        .collect::<Result<Vec<_>, _>>()
    {
        let arrs = stack.iter().map(|arr| arr.as_array()).collect();
        tile::div_untile(arrs, div, &shape)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(stack) = tile_stack
        .iter()
        .map(|t| t.extract::<PyReadonlyArrayDyn<f64>>())
        .collect::<Result<Vec<_>, _>>()
    {
        let arrs = stack.iter().map(|arr| arr.as_array()).collect();
        tile::div_untile(arrs, div, &shape)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}
