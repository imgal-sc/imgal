use ndarray::{ArrayBase, AsArray, Ix1, ViewRepr, s};

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

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
///
/// # Returns
///
/// * `f64`: The computed integral.
pub fn composite_simpson<'a, T, A>(x: A, delta_x: Option<f64>) -> f64
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let view: ArrayBase<ViewRepr<&'a T>, Ix1> = x.into();
    let d_x: f64 = delta_x.unwrap_or(1.0);

    // perform standard Simpson's rule if the number of subintervals is even,
    // if odd then slice for Simposon's rule and perform the trapezoid rule on
    // the last subinterval
    let n: usize = view.len() - 1;
    if n % 2 == 0 {
        simpson(view, delta_x).unwrap()
    } else {
        let integral: f64 = simpson(view.slice(s![..n]), delta_x).unwrap();
        let trap: f64 = (d_x / 2.0) * (view[n - 1] + view[n]).to_f64();
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
///
/// # Returns
///
/// * `Ok(f64)`: The computed integral.
/// * `Err(ImgalError)`: If the number of subintervals is odd.
pub fn simpson<'a, T, A>(x: A, delta_x: Option<f64>) -> Result<f64, ImgalError>
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let view: ArrayBase<ViewRepr<&'a T>, Ix1> = x.into();
    let d_x: f64 = delta_x.unwrap_or(1.0);

    // perfrom Simpson's 1/3 rule for an even number of subintervals only
    let n: usize = view.len() - 1;
    if n % 2 == 0 {
        let mut coef: f64;
        let mut integral: f64 = (view[0] + view[n]).to_f64();
        for i in 1..n {
            coef = if i % 2 == 1 { 4.0 } else { 2.0 };
            integral += coef * view[i].to_f64();
        }
        Ok((d_x / 3.0) * integral)
    } else {
        return Err(ImgalError::InvalidGeneric {
            msg: "An odd number of subintervals is not allowed in Simpson's 1/3 rule integration.",
        });
    }
}
