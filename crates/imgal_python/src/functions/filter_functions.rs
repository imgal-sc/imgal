use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use imgal::filter;

/// Convolve two 1D signals using the Fast Fourier Transform (FFT).
///
/// Computes the convolution of two discrete signals (`data_a` and `data_b`) by
/// transforming them into the frequency domain, multiplying them, and then
/// transforming the result back into a signal. This function uses "same-length"
/// trimming with the first parameter `data_a`. This means that the returned
/// convolution's array length will have the same length as `data_a`.
///
/// Args:
///     data_a: The first input signal to FFT convolve. Returned convolution
///         arrays will be "same-length" trimmed to `data_a`'s length.
///     data_b: The second input signal to FFT convolve.
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///        The FFT convolved result of the same length as input signal `data_a`.
#[pyfunction]
#[pyo3(name = "fft_convolve_1d")]
#[pyo3(signature = (data_a, data_b, threads=None))]
pub fn filter_fft_convolve_1d(
    py: Python,
    data_a: Vec<f64>,
    data_b: Vec<f64>,
    threads: Option<usize>,
) -> PyResult<Bound<PyArray1<f64>>> {
    Ok(filter::fft_convolve_1d(&data_a, &data_b, threads).into_pyarray(py))
}

/// Deconvolve two 1D signals using the Fast Fourier Transform (FFT).
///
/// Computes the deconvolution of two discrete signals (`data_a` and `data_b`)
/// by transforming them into the frequency domain, dividing them, and then
/// transforming the result back into a signal. This function uses "same-length"
/// trimming with the first parameter `data_a`. This means that the returned
/// deconvolution's array length will have the same length as `data_a`.
///
/// Args:
///     data_a: The first input signal to FFT deconvolve. Returned deconvolution
///         arrays will be "same-length" trimmed to `data_a`'s length.
///     data_b: The second input singal to FFT deconvolve.
///     epsilon: An epsilon value to prevent division by zero errors (default =
///         `1e-8`).
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///     The FFT deconvolved result of the same length as input signal `data_a`.
#[pyfunction]
#[pyo3(name = "fft_deconvolve_1d")]
#[pyo3(signature = (data_a, data_b, epsilon=None, threads=None))]
pub fn filter_fft_deconvolve_1d(
    py: Python,
    data_a: Vec<f64>,
    data_b: Vec<f64>,
    epsilon: Option<f64>,
    threads: Option<usize>,
) -> PyResult<Bound<PyArray1<f64>>> {
    Ok(filter::fft_deconvolve_1d(&data_a, &data_b, epsilon, threads).into_pyarray(py))
}
