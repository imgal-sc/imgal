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
///
/// # Returns
///
/// * `Vec<f64>`: The normalized Gaussian distribution.
pub fn normalized_gaussian(sigma: f64, bins: usize, range: f64, center: f64) -> Vec<f64> {
    let mut r = vec![0.0; bins];
    let mut g = vec![0.0; bins];
    let width = range / (bins as f64 - 1.0);
    r.iter_mut().enumerate().for_each(|(i, v)| {
        *v = i as f64 * width;
    });
    let sigma_sq_2 = 2.0 * sigma.powi(2);
    g.iter_mut().enumerate().for_each(|(i, v)| {
        *v = (-((r[i] - center).powi(2)) / sigma_sq_2).exp();
    });
    let g_sum = sum(&g);
    g.iter_mut().for_each(|v| {
        *v /= g_sum;
    });

    g
}
