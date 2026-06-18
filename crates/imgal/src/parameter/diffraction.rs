use crate::prelude::*;

/// Compute the Abbe diffraction limit.
///
/// # Description
///
/// Computes Ernst Abbe's diffraction limit for a microscope using:
///
/// ```text
/// d = wavelength / 2 * NA
/// ```
///
/// Where `NA` is the numerical aperture of the objective.
///
/// # Arguments
///
/// * `wavelength`: The wavelength of light in nanometers.
/// * `na`: The numerical aperture.
///
/// # Returns
///
/// * `f64`: The minimum distance required between two distinct objects to
///   resolve them as separate entities.
#[inline]
pub fn abbe_diffraction_limit<T>(wavelength: T, na: f64) -> f64
where
    T: AsNumeric,
{
    wavelength.to_f64() / (2.0 * na)
}
