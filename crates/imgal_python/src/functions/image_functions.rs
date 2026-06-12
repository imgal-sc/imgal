use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::image;

/// Create an image histogram from an n-dimensional image.
///
/// Creates a 1D image histogram from an n-dimensional image.
///
/// Args:
///     data: The input n-dimensional image.
///     bins: The number of bins to use for the image histogram. If `None`, then
///         `bins = 256`.
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///     The image histogram of the input n-dimensional image of size `bins`.
///     Each element represents the count of values falling into the
///     corresponding bin.
///
/// Errors:
///     If the input data array is empty or `bins == 0`.
#[pyfunction]
#[pyo3(name = "histogram")]
#[pyo3(signature = (data, bins=None, threads=None))]
pub fn image_histogram<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    bins: Option<usize>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        image::histogram(arr.as_array(), bins, threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        image::histogram(arr.as_array(), bins, threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        image::histogram(arr.as_array(), bins, threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i64>>() {
        image::histogram(arr.as_array(), bins, threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        image::histogram(arr.as_array(), bins, threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        image::histogram(arr.as_array(), bins, threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Compute the histogram bin midpoint value from a bin index.
///
/// Computes the midpoint value of an image histogram bin at the given index.
/// The midpoint value is the center value of the bin range.
///
/// Args:
///     index: The histogram bin index.
///     min: The minimum value of the source data used to construct the
///         histogram.
///     max: The maximum value of the source data used to construct the
///         histogram.
///     bins: The number of bins in the histogram.
///
/// Returns:
///      The midpoint bin value of the specified index.
///
/// Errors:
///      If `bins == 0`.
#[pyfunction]
#[pyo3(name = "histogram_bin_midpoint")]
pub fn image_histogram_bin_midpoint(
    index: usize,
    min: f64,
    max: f64,
    bins: usize,
) -> PyResult<f64> {
    image::histogram_bin_midpoint(index, min, max, bins)
        .map(|output| output)
        .map_err(map_imgal_error)
}

/// Compute the histogram bin value range from a bin index.
///
/// Computes the start and end values (*i.e.* the range) for a specified
/// histogram bin index.
///
/// Args:
///     index: The histogram bin index.
///     min: The minimum value of the source data used to construct the
///         histogram.
///     max: The maximum value of the source data used to construct the
///   histogram.
///         bins: The number of bins in the histogram.
///
/// Returns:
///     A tuple containing the start and end values representing the value range
///     of the specified bin index.
///
/// Errors:
///     If `bins == 0`.
#[pyfunction]
#[pyo3(name = "histogram_bin_range")]
pub fn image_histogram_bin_range(
    index: usize,
    min: f64,
    max: f64,
    bins: usize,
) -> PyResult<(f64, f64)> {
    image::histogram_bin_range(index, min, max, bins)
        .map(|output| output)
        .map_err(map_imgal_error)
}

/// Normalize an n-dimensional image using percentile-based minimum and maximum.
///
/// Performs percentile-based normalization of an input n-dimensional image with
/// minimum and maximum percentage within the range of `0.0` to `100.0`.
///
/// Args:
///     data: The input n-dimensional image to normalize.
///     min: The minimum normalization percentile in the range `0.0` to `100.0`.
///     max: The maximum normalization percentile in the range `0.0` to `100.0`.
///     clip: Boolean to indicate whether to clamp the normalized values to the
///         range `0.0` to `100.0`. If `None`, then `clip = false`.
///     axis: The axis to comppute percentiles idependently along. Each subview
///         along this axis normalized with the independent percentiles. If
///         `None`, then the input `data` is flattened.
///     epsilon: A small positive value to avoid division by zero. If `None`,
///         then `epsilon = 1e-20`.
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///     The percentile normalized n-dimensional image.
///
/// Errors:
///     If `min` and/or `max` are outside of range `0.0` to `1.0`.
#[pyfunction]
#[pyo3(name = "percentile_normalize")]
#[pyo3(signature = (data, min, max, clip=None, axis=None, epsilon=None, threads=None))]
pub fn normalize_percentile_normalize<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    min: f64,
    max: f64,
    clip: Option<bool>,
    axis: Option<usize>,
    epsilon: Option<f64>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let clip = clip.unwrap_or(false);
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        image::percentile_normalize(arr.as_array(), min, max, clip, axis, epsilon, threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        image::percentile_normalize(arr.as_array(), min, max, clip, axis, epsilon, threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        image::percentile_normalize(arr.as_array(), min, max, clip, axis, epsilon, threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i64>>() {
        image::percentile_normalize(arr.as_array(), min, max, clip, axis, epsilon, threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        image::percentile_normalize(arr.as_array(), min, max, clip, axis, epsilon, threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        image::percentile_normalize(arr.as_array(), min, max, clip, axis, epsilon, threads)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}
