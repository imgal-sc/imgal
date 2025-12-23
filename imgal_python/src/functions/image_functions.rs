use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::image;

/// Create an image histogram from an n-dimensional array.
///
/// Creates a 1-dimensional image histogram from an n-dimensional array.
///
/// Args:
///     data: The input n-dimensional array.
///     bins: The number of bins to use for the image histogram. If `None`, then
///         `bins = 256`.
///
/// Returns:
///     The image histogram of the input n-dimensional array of size `bins`.
///     Each element represents the count of values falling into the
///     corresponding bin.
#[pyfunction]
#[pyo3(name = "histogram")]
#[pyo3(signature = (data, bins=None))]
pub fn image_histogram<'py>(data: Bound<'py, PyAny>, bins: Option<usize>) -> PyResult<Vec<i64>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        return Ok(image::histogram(arr.as_array(), bins));
    }
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        return Ok(image::histogram(arr.as_array(), bins));
    }
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        return Ok(image::histogram(arr.as_array(), bins));
    }
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        return Ok(image::histogram(arr.as_array(), bins));
    }
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(image::histogram(arr.as_array(), bins));
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
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
#[pyfunction]
#[pyo3(name = "histogram_bin_midpoint")]
pub fn image_histogram_bin_midpoint(index: usize, min: f64, max: f64, bins: usize) -> f64 {
    image::histogram_bin_midpoint(index, min, max, bins)
}

/// Compute the histogram bin value range from a bin index.
///
/// Computes the start and end values (_i.e._ the range) for a specified
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
#[pyfunction]
#[pyo3(name = "histogram_bin_range")]
pub fn image_histogram_bin_range(index: usize, min: f64, max: f64, bins: usize) -> (f64, f64) {
    image::histogram_bin_range(index, min, max, bins)
}

/// Normalize an n-dimensional array using percentile-based minimum and maximum.
///
/// Performs percentile-based normalization of an input n-dimensional array with
/// minimum and maximum percentage within the range of `0.0` to `100.0`.
///
/// Args:
///     data: An n-dimensional array to normalize.
///     min: The minimum percentage to normalize.
///     max: The maximum percentage to normalize.
///     clip: Boolean to indicate whether to clamp the normalized values to the
///         range `0.0` to `100.0`. If `None`, then `clip = false`.
///     epsilon: A small positive value to avoid division by zero. If `None`,
///         then `epsilon = 1e-20`.
///
/// Returns:
///     The percentile normalized n-dimensonal array.
#[pyfunction]
#[pyo3(name = "percentile_normalize")]
#[pyo3(signature = (data, min, max, clip=None, epsilon=None))]
pub fn normalize_percentile_normalize<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    min: f64,
    max: f64,
    clip: Option<bool>,
    epsilon: Option<f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        image::percentile_normalize(arr.as_array(), min, max, clip, epsilon)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        image::percentile_normalize(arr.as_array(), min, max, clip, epsilon)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        image::percentile_normalize(arr.as_array(), min, max, clip, epsilon)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        image::percentile_normalize(arr.as_array(), min, max, clip, epsilon)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        image::percentile_normalize(arr.as_array(), min, max, clip, epsilon)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}
