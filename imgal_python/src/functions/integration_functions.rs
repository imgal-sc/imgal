use pyo3::prelude::*;

use imgal::integration;

/// Integrate a curve with Simpson's 1/3 rule and the trapezoid rule.
///
/// Approximates the definite integral using Simpson's 1/3 rule and
/// the trapezoid rule (for odd number of subintervals) with pre-computed
/// x-values:
///
/// ∫(f(x)dx) ≈ (Δx/3) * [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 2f(xₙ₋₂) + 4f(xₙ₋₁) + f(xₙ)]
///
/// Where "n" is the number of evenly spaced points in the data. If there is an
/// odd number of subintervals, the final subinterval is integrated using the
/// trapezoid rule:
///
/// ∫(f(x)dx) ≈ (Δx/2) * [f(x₀) + f(x₁)]
///
/// :param x: The 1-dimensional data to integrate.
/// :param delta_x: The width between data points, defualt = 1.0.
/// :return: The computed integral.
#[pyfunction]
#[pyo3(name = "composite_simpson")]
#[pyo3(signature = (x, delta_x=None))]
pub fn integration_composite_simpson(x: Vec<f64>, delta_x: Option<f64>) -> f64 {
    integration::composite_simpson(&x, delta_x)
}

/// Integrate a curve with the midpoint rule.
///
/// Approximates the definite integral using the midpoint rule
/// with pre-computed x-values:
///
/// ∫f(x) dx ≈ Δx * [f(x₁) + f(x₂) + ... + f(xₙ)]
///
/// :param x: The n-dimensional data to integrate.
/// :param delta_x: The width between data points, default = 1.0.
/// :return: The computed integral.
#[pyfunction]
#[pyo3(name = "midpoint")]
#[pyo3(signature = (x, delta_x=None))]
pub fn integration_midpoint(x: Vec<f64>, delta_x: Option<f64>) -> f64 {
    integration::midpoint(&x, delta_x)
}

/// Integrate a curve with Simpson's 1/3 rule.
///
/// Approximates the definite integral using Simpson's 1/3 rule and
/// with pre-computed x-values:
///
/// ∫(f(x)dx) ≈ (Δx/3) * [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 2f(xₙ₋₂) + 4f(xₙ₋₁) + f(xₙ)]
///
/// Where "n" is the number of evenly spaced points in the data.
///
/// :param x: The 1-dimensional data to integrate with an even number of
///    subintervals.
/// :param delta_x: The width between data points, defualt = 1.0.
/// :return: The computed integral.
#[pyfunction]
#[pyo3(name = "simpson")]
#[pyo3(signature = (x, delta_x=None))]
pub fn integration_simpson(x: Vec<f64>, delta_x: Option<f64>) -> f64 {
    integration::simpson(&x, delta_x).unwrap()
}
