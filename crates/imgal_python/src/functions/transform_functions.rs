use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::transform::project::sum_project;
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
///         If `None`, then `direction = 2` (symmetric padding).
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///     A new constant value padded image containing the input data.
///
/// Errors:
///     If `len(pad_config) != data.ndim`.
#[pyfunction]
#[pyo3(name = "constant_pad")]
#[pyo3(signature = (data, value, pad_config, direction=None, threads=None))]
pub fn pad_constant_pad<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    value: f64,
    pad_config: Vec<usize>,
    direction: Option<u8>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        pad::constant_pad(arr.as_array(), value as u8, &pad_config, direction, threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        pad::constant_pad(
            arr.as_array(),
            value as u16,
            &pad_config,
            direction,
            threads,
        )
        .map(|output| output.into_pyarray(py).into_any())
        .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        pad::constant_pad(
            arr.as_array(),
            value as u64,
            &pad_config,
            direction,
            threads,
        )
        .map(|output| output.into_pyarray(py).into_any())
        .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i64>>() {
        pad::constant_pad(
            arr.as_array(),
            value as i64,
            &pad_config,
            direction,
            threads,
        )
        .map(|output| output.into_pyarray(py).into_any())
        .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        pad::constant_pad(
            arr.as_array(),
            value as f32,
            &pad_config,
            direction,
            threads,
        )
        .map(|output| output.into_pyarray(py).into_any())
        .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        pad::constant_pad(arr.as_array(), value, &pad_config, direction, threads)
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
///         If `None`, then `direction = 2` (symmetric padding).
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///     A new reflected value padded image containing the input data.
///
/// Errors:
///     If `len(pad_config) != data.ndim`.
#[pyfunction]
#[pyo3(name = "reflect_pad")]
#[pyo3(signature = (data, pad_config, direction=None, threads=None))]
pub fn pad_reflect_pad<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    pad_config: Vec<usize>,
    direction: Option<u8>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        pad::reflect_pad(arr.as_array(), &pad_config, direction, threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        pad::reflect_pad(arr.as_array(), &pad_config, direction, threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        pad::reflect_pad(arr.as_array(), &pad_config, direction, threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i64>>() {
        pad::reflect_pad(arr.as_array(), &pad_config, direction, threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        pad::reflect_pad(arr.as_array(), &pad_config, direction, threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        pad::reflect_pad(arr.as_array(), &pad_config, direction, threads)
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
///         If `None`, then `direction = 2` (symmetric padding).
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///     A new reflected value padded image containing the input data.
///
/// Errors:
///     If `len(pad_config) != data.ndim`.
#[pyfunction]
#[pyo3(name = "zero_pad")]
#[pyo3(signature = (data, pad_config, direction=None, threads=None))]
pub fn pad_zero_pad<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    pad_config: Vec<usize>,
    direction: Option<u8>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        pad::zero_pad(arr.as_array(), &pad_config, direction, threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        pad::zero_pad(arr.as_array(), &pad_config, direction, threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        pad::zero_pad(arr.as_array(), &pad_config, direction, threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i64>>() {
        pad::zero_pad(arr.as_array(), &pad_config, direction, threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        pad::zero_pad(arr.as_array(), &pad_config, direction, threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        pad::zero_pad(arr.as_array(), &pad_config, direction, threads)
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
#[pyo3(name = "sum_project")]
#[pyo3(signature = (data, axis=None, threads=None))]
pub fn project_sum_project<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    axis: Option<usize>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        sum_project(arr.as_array(), axis, threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        sum_project(arr.as_array(), axis, threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        sum_project(arr.as_array(), axis, threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i64>>() {
        sum_project(arr.as_array(), axis, threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        sum_project(arr.as_array(), axis, threads)
            .map(|output| output.into_pyarray(py).into_any())
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        sum_project(arr.as_array(), axis, threads)
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
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///     A list containing views of all tiles in row-major order. The length of
///     the vector will be `divⁿ`, the number of tiles.
///
/// Errors:
///     If `div == 0`.
#[pyfunction]
#[pyo3(name = "div_tile")]
#[pyo3(signature = (data, div, threads=None))]
pub fn tile_div_tile<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    div: usize,
    threads: Option<usize>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        tile::div_tile(arr.as_array(), div, threads)
            .map(|output| {
                output
                    .iter()
                    .map(|v| v.to_owned().into_pyarray(py).into_any())
                    .collect()
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        tile::div_tile(arr.as_array(), div, threads)
            .map(|output| {
                output
                    .iter()
                    .map(|v| v.to_owned().into_pyarray(py).into_any())
                    .collect()
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        tile::div_tile(arr.as_array(), div, threads)
            .map(|output| {
                output
                    .iter()
                    .map(|v| v.to_owned().into_pyarray(py).into_any())
                    .collect()
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i64>>() {
        tile::div_tile(arr.as_array(), div, threads)
            .map(|output| {
                output
                    .iter()
                    .map(|v| v.to_owned().into_pyarray(py).into_any())
                    .collect()
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        tile::div_tile(arr.as_array(), div, threads)
            .map(|output| {
                output
                    .iter()
                    .map(|v| v.to_owned().into_pyarray(py).into_any())
                    .collect()
            })
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        tile::div_tile(arr.as_array(), div, threads)
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
///
/// Errors:
///     If `tile_stack` is empty. If `div == 0`. If `len(shape)` is not equal to
///     the tile shape length. If expected tile shapes do not match given tile
///     shapes. If the number of tiles given does not match the number of tiles
///     expected.
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
