use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::distribution;

/// Compute the quantile of a probability using the inverse normal cumulative
/// distribution function.
///
/// Computes the quantile (*z-score*) corresponding to a given cumulative
/// probabililty `prob` using Peter Acklam's rational approximation algorithm.
/// Acklam's algorithm has a relative error of less than `1.15e-9`.
///
/// Args:
///     prob: The probability value in the range of `0.0` to `1.0`.
///
/// Returns:
///     The quantile (z-score) corresponding to the given probability `prob`.
///
/// Reference:
///     <https://home.online.no/~pjacklam/notes/invnorm/>
#[pyfunction]
#[pyo3(name = "inverse_normal_cdf")]
pub fn distribution_inverse_normal_cdf(p: f64) -> PyResult<f64> {
    distribution::inverse_normal_cdf(p)
        .map(|output| output)
        .map_err(map_imgal_error)
}

/// Create a normalized Gaussian distribution over a specified range.
///
/// Creates a discrete Gaussian distribution by sampling the continuous Gaussian
/// probability density function at evenly spaced points across a given range.
/// The resulting distribution is normalized so that all values sum to `1.0`.
/// This function implements the Gaussian probability density function:
///
/// ```text
/// f(x) = exp(-((x - μ)² / (2σ²)))
/// ```
///
/// Where:
/// - `x` is the position along the range.
/// - `μ` is the center (mean).
/// - `σ` is the sigma (standard deviation).
///
/// Args:
///     sigma: The standard deviation of the Gaussian distribution (*i.e.* the
///         width).
///     bins: The number of discrete points to sample the Gaussian distribution.
///     width: The total width of the sampling range.
///     center: The mean (center) of the Gaussian distribution (*i.e.* the
///         peak).
///     threads: The requested number of threads to use for parallel execution.
///         If `None` or `1` sequential execution is used. If `0`, then the
///         maximum available parallelism is used. Thread counts are clamped to
///         the systems maximum.
///
/// Returns:
///     The normalized Gaussian distribution.
#[pyfunction]
#[pyo3(name = "normalized_gaussian")]
#[pyo3(signature = (sigma, bins, width, center, threads=None))]
pub fn distribution_normalized_gaussian(
    py: Python,
    sigma: f64,
    bins: usize,
    width: f64,
    center: f64,
    threads: Option<usize>,
) -> PyResult<Bound<PyArray1<f64>>> {
    Ok(distribution::normalized_gaussian(sigma, bins, width, center, threads).into_pyarray(py))
}
