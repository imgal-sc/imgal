use ndarray::{ArrayBase, AsArray, Ix1, ViewRepr, s};
use rayon::prelude::*;

use crate::prelude::*;

/// Integrate a curve with Simpson's 1/3 rule and the trapezoid rule.
///
/// # Description
///
/// Approximates the definite integral using Simpson's 1/3 rule and
/// the trapezoid rule (for odd number of subintervals) with pre-computed
/// x-values:
///
/// ```text
/// ∫(f(x)dx) ≈ (Δx/3) * [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 2f(xₙ₋₂) + 4f(xₙ₋₁) + f(xₙ)]
/// ```
///
/// Where `n` is the number of evenly spaced points in the data. If there is an
/// odd number of subintervals, the final subinterval is integrated using the
/// trapezoid rule:
///
/// ```text
/// ∫(f(x)dx) ≈ (Δx/2) * [f(x₀) + f(x₁)]
/// ```
///
/// # Arguments
///
/// * `x`: The 1-dimensional data to integrate.
/// * `delta_x`: The width between data points. If `None`, then `delta_x = 1.0`.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
///
/// # Returns
///
/// * `f64`: The computed integral.
pub fn composite_simpson<'a, T, A>(x: A, delta_x: Option<f64>, threads: Option<usize>) -> f64
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let x: ArrayBase<ViewRepr<&'a T>, Ix1> = x.into();
    let d_x: f64 = delta_x.unwrap_or(1.0);
    // SAFE: these unwraps are safe because in both cases the number of
    // subintervals is even
    let n: usize = x.len() - 1;
    if n.is_multiple_of(2) {
        simpson(x, delta_x, threads).unwrap()
    } else {
        let integral: f64 = simpson(x.slice(s![..n]), delta_x, threads).unwrap();
        let trap: f64 = (d_x / 2.0) * (x[n - 1] + x[n]).to_f64();
        integral + trap
    }
}

/// Integrate a curve with Simpson's 1/3 rule.
///
/// # Description
///
/// Approximates the definite integral using Simpson's 1/3 rule and
/// with pre-computed x-values:
///
/// ```text
/// ∫(f(x)dx) ≈ (Δx/3) * [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 2f(xₙ₋₂) + 4f(xₙ₋₁) + f(xₙ)]
/// ```
///
/// Where `n` is the number of evenly spaced points in the data.
///
/// # Arguments
///
/// * `x`: The 1-dimensional data to integrate with an even number of
///   subintervals.
/// * `delta_x`: The width between data points. If `None`, then `delta_x = 1.0`.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
///
/// # Returns
///
/// * `Ok(f64)`: The computed integral.
/// * `Err(ImgalError)`: If the number of subintervals is odd.
pub fn simpson<'a, T, A>(
    x: A,
    delta_x: Option<f64>,
    threads: Option<usize>,
) -> Result<f64, ImgalError>
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let x: ArrayBase<ViewRepr<&'a T>, Ix1> = x.into();
    let d_x: f64 = delta_x.unwrap_or(1.0);
    // perfrom Simpson's 1/3 rule for an even number of subintervals only
    let n: usize = x.len() - 1;
    if !n.is_multiple_of(2) {
        return Err(ImgalError::InvalidGeneric {
            msg: "An odd number of subintervals is not allowed in Simpson's 1/3 rule integration.",
        });
    }
    let seq_integration_calc = || {
        let integral = (1..n).fold((x[0] + x[n]).to_f64(), |acc, i| {
            let coef = if i % 2 == 1 { 4.0 } else { 2.0 };
            acc + coef * x[i].to_f64()
        });
        (d_x / 3.0) * integral
    };
    let par_integration_calc = || {
        let integral = (1..n)
            .into_par_iter()
            .map(|i| {
                let coef = if i % 2 == 1 { 4.0 } else { 2.0 };
                coef * x[i].to_f64()
            })
            .reduce(|| 0.0, |acc, v| acc + v)
            + (x[0] + x[n]).to_f64();
        (d_x / 3.0) * integral
    };
    Ok(par!(threads,
        seq_exp: seq_integration_calc(),
        par_exp: par_integration_calc()))
}
