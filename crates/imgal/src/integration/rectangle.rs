use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr};

use crate::prelude::*;
use crate::statistics::sum;

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
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
///
/// # Returns
///
/// * `f64`: The computed integral.
#[inline]
pub fn midpoint<'a, T, A, D>(x: A, delta_x: Option<f64>, threads: Option<usize>) -> f64
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let x: ArrayBase<ViewRepr<&'a T>, D> = x.into();
    delta_x.unwrap_or(1.0) * sum(x, threads).to_f64()
}
