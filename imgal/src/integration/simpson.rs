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
/// Where "n" is the number of evenly spaced points in the data. If there is an
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
/// * `delta_x`: The width between data points, default = 1.0.
///
/// # Returns
///
/// * `f64`: The computed integral.
pub fn composite_simpson<T>(x: &[T], delta_x: Option<f64>) -> f64
where
    T: AsNumeric,
{
    // set default delta x if necessary
    let d_x: f64 = delta_x.unwrap_or(1.0);
    // find the number of subintervals
    let n: usize = x.len() - 1;
    // check for even number of subintervals
    if n % 2 == 0 {
        simpson(x, delta_x).unwrap()
    } else {
        // compute the even subintervals with Simpson's rule
        let integral: f64 = simpson(&x[..n], delta_x).unwrap();
        // compute the last subinterval with a trapizoid
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
/// Where "n" is the number of evenly spaced points in the data.
///
/// # Arguments
///
/// * `x`: The 1-dimensional data to integrate with an even number of subintervals.
/// * `delta_x`: The width between data points, default = 1.0.
///
/// # Returns
///
/// * `Ok(f64)`: The computed integral.
/// * `Err(ImgalError)`: If the number of subintervals is odd.
pub fn simpson<T>(x: &[T], delta_x: Option<f64>) -> Result<f64, ImgalError>
where
    T: AsNumeric,
{
    // set default delta x if necessary
    let d_x: f64 = delta_x.unwrap_or(1.0);
    // find the number of subintervals
    let n: usize = x.len() - 1;
    // check for even number of subintervals
    if n % 2 == 0 {
        // compute integal with Simpson's rule
        let mut coef: f64;
        let mut integral: f64 = (x[0] + x[n]).to_f64();
        for i in 1..n {
            coef = if i % 2 == 1 { 4.0 } else { 2.0 };
            integral += coef * x[i].to_f64();
        }
        Ok((d_x / 3.0) * integral)
    } else {
        return Err(ImgalError::InvalidArrayGeneric {
            msg: "An odd number of subintervals is not allowed in Simpson's 1/3 rule integration.",
        });
    }
}
