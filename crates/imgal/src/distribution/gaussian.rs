use rayon::prelude::*;

use crate::statistics::sum;

/// Create a normalized Gaussian distribution over a specified range.
///
/// # Description
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
/// # Arguments
///
/// * `sigma`: The standard deviation of the Gaussian distribution (_i.e._ the
///   width).
/// * `bins`: The number of discrete points to sample the Gaussian distribution.
/// * `range`: The total width of the sampling range.
/// * `center`: The mean (center) of the Gaussian distribution (_i.e._ the
///   peak).
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `Vec<f64>`: The normalized Gaussian distribution.
pub fn normalized_gaussian(
    sigma: f64,
    bins: usize,
    range: f64,
    center: f64,
    parallel: bool,
) -> Vec<f64> {
    let mut gauss_arr = vec![0.0; bins];
    let width = range / (bins as f64 - 1.0);
    let sigma_sq = 2.0 * sigma * sigma;
    if parallel {
        gauss_arr.par_iter_mut().enumerate().for_each(|(i, v)| {
            let d = (i as f64 * width) - center;
            *v = (-(d * d) / sigma_sq).exp();
        });
        let g_sum = sum(&gauss_arr, false);
        gauss_arr.iter_mut().for_each(|v| *v /= g_sum);
    } else {
        gauss_arr.iter_mut().enumerate().for_each(|(i, v)| {
            let d = (i as f64 * width) - center;
            *v = (-(d * d) / sigma_sq).exp();
        });
        let g_sum = sum(&gauss_arr, false);
        gauss_arr.iter_mut().for_each(|v| *v /= g_sum);
    }

    gauss_arr
}
