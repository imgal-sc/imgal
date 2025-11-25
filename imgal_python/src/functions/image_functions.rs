use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use imgal::image;

/// Compute the image histogram from an n-dimensional array.
///
/// This function computes an image (_i.e._ frequency) histogram for the values
/// in the input n-dimensional array.
///
/// :param data: The input n-dimensional array to construct the histogram from.
/// :param bins: The number of bins to use for the histogram, default = 256.
/// :return: The histogram of the input n-dimensional array of size `bins`.
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
/// This function computes the midpoint value of an image histogram bin index.
/// The midpoint value is the center value of the bin range.
///
/// :param index: The histogram bin index.
/// :param min: The minimum value of the source data used to construct the
///     histogram.
/// :param max: The maximum value of the source data used to construct the
///     histogram.
/// :param bins: The number of bins in the histogram.
/// :return: The midpoint bin value of the specified index.
#[pyfunction]
#[pyo3(name = "histogram_bin_midpoint")]
pub fn image_histogram_bin_midpoint(index: usize, min: f64, max: f64, bins: usize) -> f64 {
    image::histogram_bin_midpoint(index, min, max, bins)
}

/// Compute the histogram bin value range from a bin index.
///
/// This function computes the start and end values (_i.e._ the range) for a
/// specified bin index.
///
/// :param index: The histogram bin index.
/// :param min: The minimum value of the source data used to construct the
///     histogram.
/// :param max: The maximum value of the source data used to construct the
///     histogram.
/// :param bins: The number of bins in the histogram.
/// :return: A tuple containing the start and end values representing the
///     value range of the specified bin index.
#[pyfunction]
#[pyo3(name = "histogram_bin_range")]
pub fn image_histogram_bin_range(index: usize, min: f64, max: f64, bins: usize) -> (f64, f64) {
    image::histogram_bin_range(index, min, max, bins)
}

/// Normalize an n-dimensional array using percentile-based minimum and maximum.
///
/// This function performs percentile-based normalization of an input
/// n-dimensional array.
///
/// :param data: An n-dimensional array to normalize.
/// :param min: The minimum percentage to normalize by in range (0..100).
/// :param max: The maximum percentage to normalize by in range (0..100).
/// :param clip: Boolean to indicate whether to clamp the normalized values to
///     the range (0..1), default = false.
/// :param epsilon: A small positive value to avoid division by zero, default =
//      1e-20.
/// :return: The percentile normalized n-dimensonal array.
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
        return Ok(
            image::percentile_normalize(arr.as_array(), min, max, clip, epsilon).into_pyarray(py),
        );
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        return Ok(
            image::percentile_normalize(arr.as_array(), min, max, clip, epsilon).into_pyarray(py),
        );
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        return Ok(
            image::percentile_normalize(arr.as_array(), min, max, clip, epsilon).into_pyarray(py),
        );
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        return Ok(
            image::percentile_normalize(arr.as_array(), min, max, clip, epsilon).into_pyarray(py),
        );
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(
            image::percentile_normalize(arr.as_array(), min, max, clip, epsilon).into_pyarray(py),
        );
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}
