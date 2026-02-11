use pyo3::prelude::*;

use imgal::parameter;

/// Compute the Abbe diffraction limit.
///
/// Computes Ernst Abbe's diffraction limit for a microscope using:
///
/// ```text
/// d = wavelength / 2 * NA
/// ```
///
/// Where `NA` is the numerical aperture of the objective.
///
/// Args:
///     wavelength: The wavelength of light in nanometers.
///     na: The numerical aperture.
///
/// Returns:
///     Abbe's diffraction limit.
#[pyfunction]
#[pyo3(name = "abbe_diffraction_limit")]
pub fn parameter_abbe_diffraction_limit(wavelength: f64, na: f64) -> f64 {
    parameter::abbe_diffraction_limit(wavelength, na)
}

/// Compute the angular frequency (omega) value.
///
/// Computes the angular frequency, omega (ω), using the following equation:
///
/// ```text
/// ω = 2π/T
/// ```
///
/// Where `T` is the period.
///
/// Args:
///     The time period.
///
/// Returns:
///     The omega (ω) value.
#[pyfunction]
#[pyo3(name = "omega")]
pub fn parameter_omega(period: f64) -> PyResult<f64> {
    Ok(parameter::omega(period))
}
