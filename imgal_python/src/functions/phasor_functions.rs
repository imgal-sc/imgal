use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray3,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::phasor::{calibration, plot, time_domain};

/// Calibrate a real and imaginary (G, S) coordinates.
///
/// Calibrates the real and imaginary (_e.g._ G and S) coordinates by rotating
/// and scaling by phase (φ) and modulation (M) respectively using:
///
/// ```text
/// g = M * cos(φ)
/// s = M * sin(φ)
/// S' = G * s + S * g
/// G' = G * g - S * s
/// ```
///
/// Where G' and S' are the calibrated real and imaginary values after rotation
/// and scaling.
///
/// Args:
///     g: The real component (G) to calibrate.
///     s: The imaginary (S) to calibrate.
///     modulation: The modulation to scale the input (G, S) coordinates.
///     phase: The phase, φ angle, to rotate the input (G, S) coordinates.
///
/// Returns:
///     The calibrated coordinates, (G, S).
#[pyfunction]
#[pyo3(name = "calibrate_coords")]
pub fn calibration_calibrate_coords(g: f64, s: f64, modulation: f64, phase: f64) -> (f64, f64) {
    calibration::calibrate_coords(g, s, modulation, phase)
}

/// Calibrate a real and imaginary (G, S) 3-dimensional phasor image.
///
/// Calibrates an input 3-dimensional phasor image by rotating and scaling G and
/// S coordinates by phase (φ) and modulation (M) respectively using:
///
/// ```text
/// g = M * cos(φ)
/// s = M * sin(φ)
/// G' = G * g - S * s
/// S' = G * s + S * g
/// ```
///
/// Where G' and S' are the calibrated real and imaginary values after rotation
/// and scaling.
///
/// This function creates a new array and does not mutate the input array.
///
/// Args:
///     data: The 3-dimensional phasor image, where G and S are channels `0` and
///         `1` respectively.
///     modulation: The modulation to scale the input (G, S) coordinates.
///     phase: The phase, φ angle, to rotate the input (G, S) coordinates.
///     axis: The channel axis. If `None`, then `axis = 2`.
///
/// Returns:
///     A 3-dimensional array with the calibrated phasor values, where
///     calibrated G and S are channels `0` and `1` respectively.
#[pyfunction]
#[pyo3(name = "calibrate_gs_image")]
#[pyo3(signature = (data, modulation, phase, axis=None))]
pub fn calibration_calibrate_gs_image<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    modulation: f64,
    phase: f64,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    // pattern match and extract allowed array types
    if let Ok(arr) = data.extract::<PyReadonlyArray3<u8>>() {
        return Ok(
            calibration::calibrate_gs_image(arr.as_array(), modulation, phase, axis)
                .into_pyarray(py),
        );
    } else if let Ok(arr) = data.extract::<PyReadonlyArray3<u16>>() {
        return Ok(
            calibration::calibrate_gs_image(arr.as_array(), modulation, phase, axis)
                .into_pyarray(py),
        );
    } else if let Ok(arr) = data.extract::<PyReadonlyArray3<u64>>() {
        return Ok(
            calibration::calibrate_gs_image(arr.as_array(), modulation, phase, axis)
                .into_pyarray(py),
        );
    } else if let Ok(arr) = data.extract::<PyReadonlyArray3<f32>>() {
        return Ok(
            calibration::calibrate_gs_image(arr.as_array(), modulation, phase, axis)
                .into_pyarray(py),
        );
    } else if let Ok(arr) = data.extract::<PyReadonlyArray3<f64>>() {
        return Ok(
            calibration::calibrate_gs_image(arr.as_array(), modulation, phase, axis)
                .into_pyarray(py),
        );
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}

/// Calibrate a real and imaginary (G, S) 3-dimensional phasor image.
///
/// Calibrates an input 3-dimensional phasor image by rotating and scaling G and
/// S coordinates by phase (φ) and modulation (M) respectively using:
///
/// ```text
/// g = M * cos(φ)
/// s = M * sin(φ)
/// G' = G * g - S * s
/// S' = G * s + S * g
/// ```
///
/// Where G' and S' are the calibrated real and imaginary values after rotation
/// and scaling.
///
/// This function mutates the input array and does not create a new array.
///
/// Args:
///     data: The 3-dimensional phasor image, where G and S are channels `0` and
///         `1` respectively.
///     modulation: The modulation to scale the input (G, S) coordinates.
///     phase: The phase, φ angle, to rotate the input (G, S) coordinates.
///     axis: The channel axis. If `None`, then `axis = 2`.
#[pyfunction]
#[pyo3(name = "calibrate_gs_image_mut")]
#[pyo3(signature = (data, modulation, phase, axis=None))]
pub fn calibration_calibrate_gs_image_mut(
    mut data: PyReadwriteArray3<f64>,
    modulation: f64,
    phase: f64,
    axis: Option<usize>,
) {
    let arr = data.as_array_mut();
    calibration::calibrate_gs_image_mut(arr, modulation, phase, axis);
}

/// Compute the modulation and phase calibration values.
///
/// Computes the modulation and phase calibration values from theoretical
/// monoexponential coordinates (computed from `tau` and `omega`) and measured
/// coordinates. The output, (M, φ), are the modulation and phase values to
/// calibrate with.
///
/// Args:
///     g: The measured real (G) value.
///     s: The measured imaginary (S) value.
///     tau: The lifetime, τ.
///     omega: The angular frequency, ω.
///
/// Returns:
///     The modulation and phase calibration values, (M, φ).
#[pyfunction]
#[pyo3(name = "modulation_and_phase")]
pub fn calibration_modulation_and_phase(g: f64, s: f64, tau: f64, omega: f64) -> (f64, f64) {
    calibration::modulation_and_phase(g, s, tau, omega)
}

/// Map G and S coordinates back to the input phasor array as a boolean mask.
///
/// Maps the G and S coordinates back to the input G/S phasor array and returns
/// a 2-dimensional boolean mask where `true` indicates G and S coordiantes
/// representing the `g_coords` and `s_coords` arrays.
///
/// Args:
///     data: The G/S 3-dimensional array.
///     g_coords: A 1-dimensional array of `g` coordinates in the `data` array.
///         The `g_coords` and `s_coords` array lengths must match.
///     s_coords: A 1-dimensional array of `s` coordiantes in the `data` array.
///         The `s_coords` and `g_coords` array lengths must match.
///     axis: The channel axis. If `None`, then `axis = 2`.
///
/// Returns:
///     A 2-dimensional boolean mask where `true` pixels represent values found
///     in the `g_coords` and `s_coords` arrays.
#[pyfunction]
#[pyo3(name = "gs_mask")]
#[pyo3(signature = (data, g_coords, s_coords, axis=None))]
pub fn plot_gs_mask<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<f64>,
    g_coords: Vec<f64>,
    s_coords: Vec<f64>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    plot::gs_mask(data.as_array(), &g_coords, &s_coords, axis)
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
}

/// Compute the modulation of phasor G and S coordinates.
///
/// Computes the modulation (M) of phasor G and S coordinates using the
/// pythagorean theorem to find the hypotenuse (_i.e._ the modulation):
///
/// ```text
/// M = √(G² + S²)
/// ````
///
/// Args:
///     g: The real component, G.
///     s: The imaginary component, S.
///
/// Returns:
///     The modulation (M) of the (G, S) phasor coordinates.
#[pyfunction]
#[pyo3(name = "gs_modulation")]
pub fn plot_gs_modulation(g: f64, s: f64) -> f64 {
    plot::gs_modulation(g, s)
}

/// Compute the phase angle of phasor G and S coordinates.
///
/// Computes the phase angle or phi (φ) of phasor G and S coordinates using:
///
/// ```text
/// φ = tan⁻¹(S / G)
/// ````
///
/// This implementation uses atan2 and computes the four quadrant arctangent of
/// the phasor coordinates.
///
/// Args:
///     g: The real component, G.
///     s: The imaginary component, S.
///
/// Returns:
///     The phase (phi, φ)  of the (G, S) phasor coordinates.
#[pyfunction]
#[pyo3(name = "gs_phase")]
pub fn plot_gs_phase(g: f64, s: f64) -> f64 {
    plot::gs_phase(g, s)
}

/// Compute the G and S coordinates for a monoexponential decay.
///
/// Computes the G and S coordinates for a monoexponential decay given as:
///
/// ```text
/// G = 1 / 1 + (ωτ)²
/// S = ωτ / 1 + (ωτ)²
/// ```
///
/// Args:
///     tau: The lifetime of a monoexponential decay.
///     omega: The angular frequency.
///
/// Returns:
///     The monoexponential decay coordinates, (G, S).
///
/// Reference:
///     <https://doi.org/10.1117/1.JBO.25.7.071203>
#[pyfunction]
#[pyo3(name = "monoexponential_coords")]
pub fn plot_monoexponential_coords(tau: f64, omega: f64) -> (f64, f64) {
    plot::monoexponential_coords(tau, omega)
}

/// Compute the real and imaginary (G, S) coordinates of a 3-dimensional decay
/// image.
///
/// Computes the real (G) and imaginary (S) components using normalized sine
/// and cosine Fourier transforms:
///
/// ```text
/// G = ∫(I(t) * cos(nωt) * dt) / ∫(I(t) * dt)
/// S = ∫(I(t) * sin(nωt) * dt) / ∫(I(t) * dt)
/// ```
///
/// Args:
///     data: I(t), the decay data image.
///     period: The period (_i.e._ time interval).
///     harmonic: The harmonic value. If `None`, then `harmonic = 1.0`.
///     axis: The decay or lifetime axis. If `None`, then `axis = 2`.
///
/// Returns:
///     The real and imaginary coordinates as a 3D (ch, row, col) image, where G
///     and S are indexed at `0` and `1` respectively on the _channel_ axis.
#[pyfunction]
#[pyo3(name = "gs_image")]
#[pyo3(signature = (data, period, mask=None, harmonic=None, axis=None))]
pub fn time_domain_gs_image<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    period: f64,
    mask: Option<PyReadonlyArray2<bool>>,
    harmonic: Option<f64>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    // pattern match and extract allowed array types
    if let Ok(arr) = data.extract::<PyReadonlyArray3<u8>>() {
        if let Some(m) = mask {
            return time_domain::gs_image(
                arr.as_array(),
                period,
                Some(m.as_array()),
                harmonic,
                axis,
            )
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error);
        } else {
            return time_domain::gs_image(arr.as_array(), period, None, harmonic, axis)
                .map(|output| output.into_pyarray(py))
                .map_err(map_imgal_error);
        }
    } else if let Ok(arr) = data.extract::<PyReadonlyArray3<u16>>() {
        if let Some(m) = mask {
            return time_domain::gs_image(
                arr.as_array(),
                period,
                Some(m.as_array()),
                harmonic,
                axis,
            )
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error);
        } else {
            return time_domain::gs_image(arr.as_array(), period, None, harmonic, axis)
                .map(|output| output.into_pyarray(py))
                .map_err(map_imgal_error);
        }
    } else if let Ok(arr) = data.extract::<PyReadonlyArray3<u64>>() {
        if let Some(m) = mask {
            return time_domain::gs_image(
                arr.as_array(),
                period,
                Some(m.as_array()),
                harmonic,
                axis,
            )
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error);
        } else {
            return time_domain::gs_image(arr.as_array(), period, None, harmonic, axis)
                .map(|output| output.into_pyarray(py))
                .map_err(map_imgal_error);
        }
    } else if let Ok(arr) = data.extract::<PyReadonlyArray3<f32>>() {
        if let Some(m) = mask {
            return time_domain::gs_image(
                arr.as_array(),
                period,
                Some(m.as_array()),
                harmonic,
                axis,
            )
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error);
        } else {
            return time_domain::gs_image(arr.as_array(), period, None, harmonic, axis)
                .map(|output| output.into_pyarray(py))
                .map_err(map_imgal_error);
        }
    } else if let Ok(arr) = data.extract::<PyReadonlyArray3<f64>>() {
        if let Some(m) = mask {
            return time_domain::gs_image(
                arr.as_array(),
                period,
                Some(m.as_array()),
                harmonic,
                axis,
            )
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error);
        } else {
            return time_domain::gs_image(arr.as_array(), period, None, harmonic, axis)
                .map(|output| output.into_pyarray(py))
                .map_err(map_imgal_error);
        }
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}

/// Compute the imaginary (S) component of a 1-dimensional decay curve.
///
/// Computes the imaginary (S) component is calculated using the normalized sine
/// Fourier transform:
///
/// ```text
/// S = ∫(I(t) * sin(nωt) * dt) / ∫(I(t) * dt)
/// ```
///
/// Where `n` and `ω` are harmonic and omega values respectively.
///
/// Args:
///     data: I(t), the 1-dimensonal decay curve.
///     period: The period (_i.e._ time interval).
///     harmonic: The harmonic value. If `None`, then `harmonic = 1.0`.
///
/// Returns:
///     The imaginary component, S.
#[pyfunction]
#[pyo3(name = "imaginary_coord")]
#[pyo3(signature = (data, period, harmonic=None))]
pub fn time_domain_imaginary_coord(data: Vec<f64>, period: f64, harmonic: Option<f64>) -> f64 {
    time_domain::imaginary_coord(&data, period, harmonic)
}

/// Compute the real (G) component of a 1-dimensional decay curve.
///
/// Computes the real (G) component is calculated using the normalized cosine
/// Fourier transform:
///
/// ```text
/// G = ∫(I(t) * cos(nωt) * dt) / ∫(I(t) * dt)
/// ```
///
/// Where `n` and `ω` are harmonic and omega values respectively.
///
/// Args:
///     data: I(t), the 1-dimensional decay curve.
///     period: The period, (_i.e._ time interval).
///     harmonic: The harmonic value. If `None`, then `harmonic = 1.0`.
///
/// Returns:
///     The real component, G.
#[pyfunction]
#[pyo3(name = "real_coord")]
#[pyo3(signature = (data, period, harmonic=None))]
pub fn time_domain_real_coord(data: Vec<f64>, period: f64, harmonic: Option<f64>) -> f64 {
    time_domain::real_coord(&data, period, harmonic)
}
