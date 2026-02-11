use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::distribution;

/// Compute the quantile of a probability using the inverse normal cumulative
/// distribution function.
///
/// Computes the quantile (_z-score_) corresponding to a given cumulative
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
///     sigma: The standard deviation of the Gaussian distribution (_i.e._ the
///         width).
///     bins: The number of discrete points to sample the Gaussian distribution.
///     range: The total width of the sampling range.
///     center: The mean (center) of the Gaussian distribution (_i.e._ the
///         peak).
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
///
/// Returns:
///     The normalized Gaussian distribution.
#[pyfunction]
#[pyo3(name = "normalized_gaussian")]
#[pyo3(signature = (sigma, bins, range, center, parallel=None))]
pub fn distribution_normalized_gaussian(
    py: Python,
    sigma: f64,
    bins: usize,
    range: f64,
    center: f64,
    parallel: Option<bool>,
) -> PyResult<Bound<PyArray1<f64>>> {
    let parallel = parallel.unwrap_or(false);

    Ok(distribution::normalized_gaussian(sigma, bins, range, center, parallel).into_pyarray(py))
}
