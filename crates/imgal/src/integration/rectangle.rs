use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr};

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
/// * `x`: The n-dimensional array to integrate.
/// * `delta_x`: The width between data points. If `None`, then `delta_x = 1.0`.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `f64`: The computed integral.
#[inline]
pub fn midpoint<'a, T, A, D>(x: A, delta_x: Option<f64>, parallel: bool) -> f64
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let x: ArrayBase<ViewRepr<&'a T>, D> = x.into();
    delta_x.unwrap_or(1.0) * sum(x, parallel).to_f64()
}
