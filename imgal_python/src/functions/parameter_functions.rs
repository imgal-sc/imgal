use pyo3::prelude::*;

use imgal::parameter;

/// Compute the Abbe diffraction limit.
///
/// Compute Ernst Abbe's diffraction limit using:
///
/// d = wavelength / 2 * NA
///
/// Where "NA" is the numerical aperture of the objective.
///
/// :param wavelength: The wavelength of light.
/// :param na: The numerical aperture.
/// :return: Abbe's diffraction limit.
#[pyfunction]
#[pyo3(name = "abbe_diffraction_limit")]
pub fn parameter_abbe_diffraction_limit(wavelength: f64, na: f64) -> f64 {
    parameter::abbe_diffraction_limit(wavelength, na)
}

/// Compute the angular frequency (omega) value.
///
/// Compute the angular frequency, omega (ω), using the following equation:
///
/// ω = 2π/T
///
/// Where "T" is the period.
///
/// :param period: The time period.
/// :return: The omega (ω) value.
#[pyfunction]
#[pyo3(name = "omega")]
pub fn parameter_omega(period: f64) -> PyResult<f64> {
    Ok(parameter::omega(period))
}
