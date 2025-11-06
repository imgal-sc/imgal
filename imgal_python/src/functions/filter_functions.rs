use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use imgal::filter;

/// Convolve two 1-dimensional signals using the Fast Fourier Transform (FFT).
///
/// Compute the convolution of two discrete signals ("a" and "b") by transforming
/// them to the frequency domain, multiplying them, and then transforming the
/// result back into a signal. This function uses "same-length" trimming with the
/// first parameter "a". This means that the returned convolution's array length
/// will have the same length as "a".
///
/// :param a: The first input signal to FFT convolve. Returned convolution arrays
///     will be "same-length" trimmed to "a"'s length.
/// :param b: The second input signal to FFT convolve.
/// :return: The FFT convolved result of the same length as input signal "a".
#[pyfunction]
#[pyo3(name = "fft_convolve_1d")]
pub fn filter_fft_convolve_1d(
    py: Python,
    a: Vec<f64>,
    b: Vec<f64>,
) -> PyResult<Bound<PyArray1<f64>>> {
    Ok(filter::fft_convolve_1d(&a, &b).into_pyarray(py))
}

/// Deconvolve two 1-dimensional signals using the Fast Fourier Transform (FFT).
///
/// Compute the deconvolution of two discrete signals (`a` and `b`) by transforming
/// them to the frequency domain, dividing them, and then transforming the result
/// back into a signal. This function uses "same-length" triming with the first
/// parameter "a". This means that the returned deconvolution's array length will
/// have the same length as "a".
///
/// :param a: The first input signal to FFT deconvolve. Returned deconvolution arrays
///     will be "same-length" trimmed to "a"'s length.
/// :param b: The second input signal to deconvolve.
/// :param epsilon: An epsilon value to prevent division by zero errors (default =
///     1e-8).
/// :return: The FFT deconvolved result of the same length as input signal "a".
#[pyfunction]
#[pyo3(name = "fft_deconvolve_1d")]
#[pyo3(signature = (a, b, epsilon=None))]
pub fn filter_fft_deconvolve_1d(
    py: Python,
    a: Vec<f64>,
    b: Vec<f64>,
    epsilon: Option<f64>,
) -> PyResult<Bound<PyArray1<f64>>> {
    Ok(filter::fft_deconvolve_1d(&a, &b, epsilon).into_pyarray(py))
}
