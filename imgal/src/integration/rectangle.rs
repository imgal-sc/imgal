use crate::statistics::sum;
use crate::traits::numeric::AsNumeric;

/// Integrate a curve with the midpoint rule.
///
/// # Description
///
/// Approximates the definite integral using the midpoint rule
/// with pre-computed x-values:
///
/// ```text
/// ∫f(x) dx ≈ Δx * [f(x₁) + f(x₂) + ... + f(xₙ)]
/// ```
///
/// # Arguments
///
/// * `x`: The 1-dimensional array to integrate.
/// * `delta_x`: The width between data points, default = 1.0.
///
/// # Returns
///
/// * `f64`: The computed integral.
#[inline]
pub fn midpoint<T>(x: &[T], delta_x: Option<f64>) -> f64
where
    T: AsNumeric,
{
    delta_x.unwrap_or(1.0) * sum(x).to_f64()
}
