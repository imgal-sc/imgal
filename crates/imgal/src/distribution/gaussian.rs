use ndarray::Array1;
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
/// * `sigma`: The standard deviation of the Gaussian distribution (*i.e.* the
///   width).
/// * `bins`: The number of discrete points to sample the Gaussian distribution.
/// * `width`: The total width of the sampling range.
/// * `center`: The mean (center) of the Gaussian distribution (*i.e.* the
///   peak).
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
///
/// # Returns
///
/// * `Array1<f64>`: The normalized Gaussian distribution.
pub fn normalized_gaussian(
    sigma: f64,
    bins: usize,
    width: f64,
    center: f64,
    threads: Option<usize>,
) -> Array1<f64> {
    let mut gauss_arr = vec![0.0; bins];
    let width = width / (bins as f64 - 1.0);
    let sigma_sq = 2.0 * sigma * sigma;
    let gauss_calc = |(i, v): (usize, &mut f64)| {
        let d = (i as f64 * width) - center;
        *v = (-(d * d) / sigma_sq).exp();
    };
    par!(threads,
        seq_exp: gauss_arr.iter_mut().enumerate().for_each(gauss_calc),
        par_exp: gauss_arr.par_iter_mut().enumerate().for_each(gauss_calc));
    let gauss_sum = sum(&gauss_arr, threads);
    par!(threads,
        seq_exp: gauss_arr.iter_mut().for_each(|v| *v /= gauss_sum),
        par_exp: gauss_arr.par_iter_mut().for_each(|v| *v /= gauss_sum));
    Array1::from_vec(gauss_arr)
}
