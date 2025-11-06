use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::distribution;

/// Generate a normalized Gaussian distribution over a specified range.
///
/// This function creates a discrete Gaussian distribution by sampling the continuous
/// Gaussian probability density function at evenly spaced points across a given range.
/// The resulting distribution is normalized so that all values sum to 1.0.
/// The function implements the Gaussian probability density function:
///
/// f(x) = exp(-((x - μ)² / (2σ²)))
///
/// where:
/// - `x` is the position along the range.
/// - `μ` is the center (mean).
/// - `σ` is the sigma (standard deviation).
///
///
/// :param sigma: The standard deviation of the Gaussian distribution (i.e. the width).
/// :param bins: The number of discrete points to sample the Gaussian distribution.
/// :param range: The total width of the sampling range.
/// :param center: The mean (center) of the Gaussian distribution (i.e. the peak).
/// :return: The normalized Gaussian distribution.
#[pyfunction]
#[pyo3(name = "gaussian")]
pub fn distribution_gaussian(
    py: Python,
    sigma: f64,
    bins: usize,
    range: f64,
    center: f64,
) -> PyResult<Bound<PyArray1<f64>>> {
    Ok(distribution::gaussian(sigma, bins, range, center).into_pyarray(py))
}

/// Compute quantile of a probability using the inverse normal cumulative
/// distribution function.
///
/// The function calculates the quantile (z-score) corresponding to a given
/// cumulative probability "p" using Peter Acklam's rational approximation
/// algorithm. Acklam's algorithm has a relative error of less than 1.15e-9.
///
/// :param p: The probability value in the range of 0.0 to 1.0.
/// :reeturn: The quantile (z-score) corresponding to the given probability
///    "p".
#[pyfunction]
#[pyo3(name = "inverse_normal_cdf")]
pub fn distribution_inverse_cdf(p: f64) -> PyResult<f64> {
    distribution::inverse_normal_cdf(p)
        .map(|output| output)
        .map_err(map_imgal_error)
}
