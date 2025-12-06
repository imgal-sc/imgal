use std::f64::consts::PI;

use crate::traits::numeric::AsNumeric;

/// Compute the angular frequency (omega) value.
///
/// # Description
///
/// Computes the angular frequency, omega (ω), using the following equation:
///
/// ```text
/// ω = 2π/T
/// ```
///
/// Where `T` is the period.
///
/// # Arguments
///
/// * `period`: The time period.
///
/// # Returns
///
/// * `f64`: The omega (ω) value.
#[inline(always)]
pub fn omega<T>(period: T) -> f64
where
    T: AsNumeric,
{
    2.0 * PI / period.to_f64()
}
