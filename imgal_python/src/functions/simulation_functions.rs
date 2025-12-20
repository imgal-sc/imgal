use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray3,
    PyReadwriteArray1, PyReadwriteArray3,
};
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::simulation;

/// Create a 1-dimensional Gaussian IRF convolved monoexponential or
/// multiexponential decay curve.
///
/// Creates a 1-dimensional Gaussian instrument response function (IRF)
/// convolved monoexponential or multiexponential decay curve. The ideal decay
/// curve is defined as the sum of one or more exponential components, each
/// characterized by a lifetime (τ) and fractional intensity:
///
/// ```text
/// I(t) = Σᵢ αᵢ × exp(-t/τᵢ)
/// ```
///
/// Args:
///     samples: The number of discrete points that make up the decay curve.
///     period: The period (_i.e._ time interval).
///     taus: An array of lifetimes. For a monoexponential decay curve use a
///         single tau value and a fractional intensity of `1.0`. For a
///         multiexponential decay curve use two or more tau values, matched
///         with their respective fractional intensity. The `taus` and
///         `fractions` arrays must have the same length. Tau values set to
///         `0.0` will be skipped.
///     fractions: An array of fractional intensities for each tau in the `taus`
///         array. The `fractions` array must be the same length as the `taus`
///         array and sum to `1.0`. Fraction values set to `0.0` will be
///         skipped.
///     total_counts: The total intensity count (_e.g._ photon count) of the
///         decay curve.
///     irf_center: The temporal position of the IRF peak within the time range.
///     irf_width: The full width at half maximum (FWHM) of the IRF.
///
/// Returns:
///     The 1-dimensional Gaussian IRF convolved monoexponential or
///     multiexponential decay curve.
#[pyfunction]
#[pyo3(name = "gaussian_exponential_decay_1d")]
pub fn decay_gaussian_exponential_decay_1d(
    py: Python,
    samples: usize,
    period: f64,
    taus: Vec<f64>,
    fractions: Vec<f64>,
    total_counts: f64,
    irf_center: f64,
    irf_width: f64,
) -> PyResult<Bound<PyArray1<f64>>> {
    simulation::decay::gaussian_exponential_decay_1d(
        samples,
        period,
        &taus,
        &fractions,
        total_counts,
        irf_center,
        irf_width,
    )
    .map(|output| output.into_pyarray(py))
    .map_err(map_imgal_error)
}

/// Create a 3-dimensional Gaussian IRF convolved monoexponential or
/// multiexponential decay curve.
///
/// Creates a 3-dimensional Gaussian instrument response function (IRF)
/// convolved monoexponential or multiexponential decay curve. The ideal decay
/// curve is defined as the sum of one or more exponential components, each
/// characterized by a lifetime (τ) and fractional intensity:
///
/// ```text
/// I(t) = Σᵢ αᵢ × exp(-t/τᵢ)
/// ```
///
/// Args:
///     samples: The number of discrete points that make up the decay curve.
///     period: The period (_i.e._ time interval).
///     taus: An array of lifetimes. For a monoexponential decay curve use a
///         single tau value and a fractional intensity of `1.0`. For a
///         multiexponential decay curve use two or more tau values, matched
///         with their respective fractional intensity. The `taus` and
///         `fractions` arrays must have the same length. Tau values set to
///         `0.0` will be skipped.
///     fractions: An array of fractional intensities for each tau in the `taus`
///         array. The `fractions` array must be the same length as the `taus`
///         array and sum to `1.0`. Fraction values set to `0.0` will be
///         skipped.
///     total_counts: The total intensity count (_e.g._ photon count) of the
///         decay curve.
///     irf_center: The temporal position of the IRF peak within the time range.
///     irf_width: The full width at half maximum (FWHM) of the IRF.
///     shape: The row and col shape to broadcast the decay curve into.
///
/// Returns:
///     The 3-dimensional Gaussian IRF convolved monoexponential or
///     multiexponential decay curve with dimension (row, col, t).
#[pyfunction]
#[pyo3(name = "gaussian_exponential_decay_3d")]
pub fn decay_gaussian_exponential_decay_3d(
    py: Python,
    samples: usize,
    period: f64,
    taus: Vec<f64>,
    fractions: Vec<f64>,
    total_counts: f64,
    irf_center: f64,
    irf_width: f64,
    shape: (usize, usize),
) -> PyResult<Bound<PyArray3<f64>>> {
    simulation::decay::gaussian_exponential_decay_3d(
        samples,
        period,
        &taus,
        &fractions,
        total_counts,
        irf_center,
        irf_width,
        shape,
    )
    .map(|output| output.into_pyarray(py))
    .map_err(map_imgal_error)
}

/// Create a 1-dimensional ideal monoexponential or multiexponential decay
/// curve.
///
/// Creates a 1-dimensional ideal exponential decay curve by computing the sum
/// of one or more exponential components, each characterized by a lifetime (τ)
/// and fractional intensity as defined by:
///
/// ```text
/// I(t) = Σᵢ αᵢ × exp(-t/τᵢ)
/// ```
///
/// Where `αᵢ` are the pre-exponential factors derived from the fractional
/// intensities and lifetimes.
///
/// Args:
///     samples: The number of discrete points that make up the decay curve.
///     period: The period (_i.e._ time interval).
///     taus: An array of lifetimes. For a monoexponential decay curve use a
///         single tau value and a fractional intensity of `1.0`. For a
///         multiexponential decay curve use two or more tau values, matched
///         with their respective fractional intensity. The `taus` and
///         `fractions` arrays must have the same length. Tau values set to
///         `0.0` will be skipped.
///     fractions: An array of fractional intensities for each tau in the `taus`
///         array. The `fractions` array must be the same length as the `taus`
///         array and sum to `1.0`. Fraction values set to `0.0` will be
///         skipped.
///     total_counts: The total intensity count (_e.g._ photon count) of the
///         decay curve.
///
/// Returns:
///     The 1-dimensional monoexponential or multiexponential decay curve.
///
/// Reference:
///     <https://doi.org/10.1111/j.1749-6632.1969.tb56231.x>
#[pyfunction]
#[pyo3(name = "ideal_exponential_decay_1d")]
pub fn decay_ideal_exponential_decay_1d(
    py: Python,
    samples: usize,
    period: f64,
    taus: Vec<f64>,
    fractions: Vec<f64>,
    total_counts: f64,
) -> PyResult<Bound<PyArray1<f64>>> {
    simulation::decay::ideal_exponential_decay_1d(samples, period, &taus, &fractions, total_counts)
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
}

/// Create a 3-dimensional ideal monoexponential or multiexponential decay
/// curve.
///
/// Creates a 3-dimensional ideal exponential decay curve by computing the sum
/// of one or more exponential components, each characterized by a lifetime (τ)
/// and fractional intensity as defined by:
///
/// ```text
/// I(t) = Σᵢ αᵢ × exp(-t/τᵢ)
/// ```
///
/// Where `αᵢ` are the pre-exponential factors derived from the fractional
/// intensities and lifetimes.
///
/// Args:
///     samples: The number of discrete points that make up the decay curve.
///     period: The period (_i.e._ time interval).
///     taus: An array of lifetimes. For a monoexponential decay curve use a
///         single tau value and a fractional intensity of `1.0`. For a
///         multiexponential decay curve use two or more tau values, matched
///         with their respective fractional intensity. The `taus` and
///         `fractions` arrays must have the same length. Tau values set to
///         `0.0` will be skipped.
///     fractions: An array of fractional intensities for each tau in the `taus`
///         array. The `fractions` array must be the same length as the `taus`
///         array and sum to `1.0`. Fraction values set to `0.0` will be
///         skipped.
///     total_counts: The total intensity count (_e.g._ photon count) of the
///         decay curve.
///     shape: The row and col shape to broadcast the decay curve into.
///
/// Returns:
///     The 3-dimensional monoexponential or multiexponential decay curve with
///     dimensions (row, col, t).
///
/// Reference:
///     <https://doi.org/10.1111/j.1749-6632.1969.tb56231.x>
#[pyfunction]
#[pyo3(name = "ideal_exponential_decay_3d")]
pub fn decay_ideal_exponential_decay_3d(
    py: Python,
    samples: usize,
    period: f64,
    taus: Vec<f64>,
    fractions: Vec<f64>,
    total_counts: f64,
    shape: (usize, usize),
) -> PyResult<Bound<PyArray3<f64>>> {
    simulation::decay::ideal_exponential_decay_3d(
        samples,
        period,
        &taus,
        &fractions,
        total_counts,
        shape,
    )
    .map(|output| output.into_pyarray(py))
    .map_err(map_imgal_error)
}

/// Create a 1-dimensional IRF convolved monoexponential or multiexponential
/// decay curve.
///
/// Creates a 1-dimensional instrument response function (IRF) convolved
/// monoexponential or multiexponential decay curve. The ideal decay curve is
/// defined as the sum of one or more exponential components, each characterized
/// by a lifetime (τ) and fractional intensity:
///
/// ```text
/// I(t) = Σᵢ αᵢ × exp(-t/τᵢ)
/// ```
///
/// Args:
///     irf: The IRF as a 1-dimensional array.
///     samples: The number of discrete points that make up the decay curve.
///     period: The period (_i.e._ time interval).
///     taus: An array of lifetimes. For a monoexponential decay curve use a
///         single tau value and a fractional intensity of `1.0`. For a
///         multiexponential decay curve use two or more tau values, matched
///         with their respective fractional intensity. The `taus` and
///         `fractions` arrays must have the same length. Tau values set to
///         `0.0` will be skipped.
///     fractions: An array of fractional intensities for each tau in the `taus`
///         array. The `fractions` array must be the same length as the `taus`
///         array and sum to `1.0`. Fraction values set to `0.0` will be
///         skipped.
///     total_counts: The total intensity count (_e.g._ photon count) of the
///         decay curve.
///
/// Returns:
///     The 1-dimensional IRF convolved monoexponential or multiexponential
///     decay curve.
#[pyfunction]
#[pyo3(name = "irf_exponential_decay_1d")]
pub fn decay_irf_exponential_decay_1d(
    py: Python,
    irf: Vec<f64>,
    samples: usize,
    period: f64,
    taus: Vec<f64>,
    fractions: Vec<f64>,
    total_counts: f64,
) -> PyResult<Bound<PyArray1<f64>>> {
    simulation::decay::irf_exponential_decay_1d(
        &irf,
        samples,
        period,
        &taus,
        &fractions,
        total_counts,
    )
    .map(|output| output.into_pyarray(py))
    .map_err(map_imgal_error)
}

/// Create a 3-dimensional IRF convolved monoexponential or multiexponential
/// decay curve.
///
/// Creates a 3-dimensional instrument response function (IRF) convolved
/// monoexponential or multiexponential decay curve. The ideal decay curve is
/// defined as the sum of one or more exponential components, each characterized
/// by a lifetime (τ) and fractional intensity:
///
/// ```text
/// I(t) = Σᵢ αᵢ × exp(-t/τᵢ)
/// ```
///
/// Args:
///     irf: The IRF as a 1-dimensional array.
///     samples: The number of discrete points that make up the decay curve.
///     period: The period (_i.e._ time interval).
///     taus: An array of lifetimes. For a monoexponential decay curve use a
///         single tau value and a fractional intensity of `1.0`. For a
///         multiexponential decay curve use two or more tau values, matched
///         with their respective fractional intensity. The `taus` and
///         `fractions` arrays must have the same length. Tau values set to
///         `0.0` will be skipped.
///     fractions: An array of fractional intensities for each tau in the `taus`
///         array. The `fractions` array must be the same length as the `taus`
///         array and sum to `1.0`. Fraction values set to `0.0` will be
///         skipped.
///     total_counts: The total intensity count (_e.g._ photon count) of the
///         decay curve.
///     shape: The row and col shape to broadcast the decay curve into.
///
/// Returns:
///     The 3-dimensional IRF convolved monoexponential or multiexponential
///     decay curve with dimensions (row, col, t).
#[pyfunction]
#[pyo3(name = "irf_exponential_decay_3d")]
pub fn decay_irf_exponential_decay_3d(
    py: Python,
    irf: Vec<f64>,
    samples: usize,
    period: f64,
    taus: Vec<f64>,
    fractions: Vec<f64>,
    total_counts: f64,
    shape: (usize, usize),
) -> PyResult<Bound<PyArray3<f64>>> {
    simulation::decay::irf_exponential_decay_3d(
        &irf,
        samples,
        period,
        &taus,
        &fractions,
        total_counts,
        shape,
    )
    .map(|output| output.into_pyarray(py))
    .map_err(map_imgal_error)
}

/// Create a 2-dimensional array with a linear gradient.
///
/// Creates a linear gradient of increasing values from the top of the array to
/// the bottom along the row axis. Setting the `offset` parameter controls how
/// far the gradient extends while the `scale` parameter controls the rate
/// values increase per row.
///
/// Args:
///     offset: The number of rows from the top of the array that remain at
///         zero.
///     scale: The rate of increase per row. This value controls the steepness
///         of the gradient.
///     shape: The row and col shape of the gradient array.
///
/// Returns:
///     The 2-dimensional gradient array.
#[pyfunction]
#[pyo3(name = "linear_gradient_2d")]
pub fn gradient_linear_gradient_2d(
    py: Python,
    offset: usize,
    scale: f64,
    shape: (usize, usize),
) -> Bound<PyArray2<f64>> {
    simulation::gradient::linear_gradient_2d(offset, scale, shape).into_pyarray(py)
}

/// Create a 3-dimensional array with a linear gradient.
///
/// Creates a linear gradient of increasing values from the top of the array to
/// the bottom along the pln or z axis. Setting the `offset` parameter controls
/// how far the gradient extends while the `scale` parameter controls the rate
/// values increase per pln.
///
/// Args:
///     offset: The number of plns from the top of the array tha tremain at
///         zero.
///     scale: The rate of increase per pln. This value controls the steepness
///         of the gradient.
///     shape: The pln, row and col shape of the gradient array.
///
/// Returns:
///     The 3-dimensional gradient array.
#[pyfunction]
#[pyo3(name = "linear_gradient_3d")]
pub fn gradient_linear_gradient_3d(
    py: Python,
    offset: usize,
    scale: f64,
    shape: (usize, usize, usize),
) -> Bound<PyArray3<f64>> {
    simulation::gradient::linear_gradient_3d(offset, scale, shape).into_pyarray(py)
}

/// Create a 1-dimensional Gaussian instrument response function (IRF).
///
/// Creates a Gaussian IRF by converting "full width at half maximum" (FWHM)
/// parameters into a normalized Gaussian distribution. The FWHM is converted to
/// standard deviation using the relationship:
///
/// ```text
/// σ = FWHM / (2 × √(2 × ln(2)))
/// ```
///
/// Where `ln(2) ≈ 0.693147` is the natural logarithm of `2`.
///
/// Args:
///     bins: The number of discrete points to sample the Gaussian distribution.
///     time_range: The total time range over which to simulate the IRF.
///     irf_center: The temporal position of the IRF peak within the time range.
///     irf_width: The full width at half maximum (FWHM) of the IRF.
///
/// Returns:
///     The simulated 1-dimensional IRF curve.
#[pyfunction]
#[pyo3(name = "gaussian_irf_1d")]
pub fn instrument_gaussian_irf_1d(
    py: Python,
    bins: usize,
    time_range: f64,
    irf_center: f64,
    irf_width: f64,
) -> PyResult<Bound<PyArray1<f64>>> {
    Ok(
        simulation::instrument::gaussian_irf_1d(bins, time_range, irf_center, irf_width)
            .into_pyarray(py),
    )
}

/// Apply Poisson noise on a 1-dimensional array.
///
/// Applies Poisson noise (_i.e._ shot noise) on a 1-dimensional array of data.
/// An element-wise lambda value (scaled by the `scale` parameter) is used to
/// simulate the Poisson noise with variable signal strength.
///
/// This function creates a new array and does not mutate the input array.
///
/// Args:
///     data: The input 1-dimensional array.
///     scale: The scale factor.
///     seed: Pseudorandom number generator seed. Set the `seed` value to apply
///         homogenous noise to the input array. If `None`, then heterogenous
///         noise is applied to the input array.
///
/// Returns:
///     A 1-dimensional array of the input data with Poisson noise applied.
#[pyfunction]
#[pyo3(name = "poisson_noise_1d")]
#[pyo3(signature = (data, scale, seed=None))]
pub fn noise_poisson_noise_1d<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    scale: f64,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // pattern match and extract allowed array types
    if let Ok(arr) = data.extract::<PyReadonlyArray1<u8>>() {
        return Ok(
            simulation::noise::poisson_noise_1d(arr.as_slice().unwrap(), scale, seed)
                .into_pyarray(py),
        );
    } else if let Ok(arr) = data.extract::<PyReadonlyArray1<u16>>() {
        return Ok(
            simulation::noise::poisson_noise_1d(arr.as_slice().unwrap(), scale, seed)
                .into_pyarray(py),
        );
    } else if let Ok(arr) = data.extract::<PyReadonlyArray1<u64>>() {
        return Ok(
            simulation::noise::poisson_noise_1d(arr.as_slice().unwrap(), scale, seed)
                .into_pyarray(py),
        );
    } else if let Ok(arr) = data.extract::<PyReadonlyArray1<f32>>() {
        return Ok(
            simulation::noise::poisson_noise_1d(arr.as_slice().unwrap(), scale, seed)
                .into_pyarray(py),
        );
    } else if let Ok(arr) = data.extract::<PyReadonlyArray1<f64>>() {
        return Ok(
            simulation::noise::poisson_noise_1d(arr.as_slice().unwrap(), scale, seed)
                .into_pyarray(py),
        );
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}

/// Apply Poisson noise on a 1-dimensional array.
///
/// Applies Poisson noise (_i.e._ shot noise) on a 1-dimensional array of data.
/// An element-wise lambda value (scaled by the `scale` parameter) is used to
/// simulate the Poisson noise with variable signal strength.
///
/// This function mutates the input array and does not create a new array.
///
/// Args:
///     data: The input 1-dimensional array view to mutate.
///     scale: The scale factor.
///     seed: Pseudorandom number generator seed. Set the `seed` value to apply
///         homogenous noise to the input array. If `None`, then heterogenous
///         noise is applied to the input array.
#[pyfunction]
#[pyo3(name = "poisson_noise_1d_mut")]
#[pyo3(signature= (data, scale, seed=None))]
pub fn noise_poisson_noise_1d_mut(mut data: PyReadwriteArray1<f64>, scale: f64, seed: Option<u64>) {
    // get mutable slice, all 1D arrays are contiguous
    let d = data.as_slice_mut().unwrap();
    simulation::noise::poisson_noise_1d_mut(d, scale, seed);
}

/// Apply Poisson noise on a 3-dimensional array.
///
/// Applies Poisson noise (_i.e._ shot noise) on a 3-dimensional array of data.
/// An element-wise lambda value (scaled by the `scale` parameter) is used to
/// simulate Poisson noise with variable signal strength.
///
/// This function creates a new array and does not mutate the input array.
///
/// Args:
///     data: The input 3-dimensional array.
///     scale: The scale factor.
///     seed: Pseudorandom number generator seed. Set the `seed` value to apply
///         homogenous noise to the input array. If `None`, then heterogenous
///         noise is applied to the input array.
///     axis: The signal data axis. If `None`, then `axis = 2`.
///
/// Returns:
///     A 3-dimensional array of the input data with Poisson noise applied.
#[pyfunction]
#[pyo3(name = "poisson_noise_3d")]
#[pyo3(signature = (data, scale, seed=None, axis=None))]
pub fn noise_poisson_noise_3d<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    scale: f64,
    seed: Option<u64>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    // pattern match and extract allowed array types
    if let Ok(arr) = data.extract::<PyReadonlyArray3<u8>>() {
        simulation::noise::poisson_noise_3d(arr.as_array(), scale, seed, axis)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray3<u16>>() {
        simulation::noise::poisson_noise_3d(arr.as_array(), scale, seed, axis)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray3<u64>>() {
        simulation::noise::poisson_noise_3d(arr.as_array(), scale, seed, axis)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray3<f32>>() {
        simulation::noise::poisson_noise_3d(arr.as_array(), scale, seed, axis)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray3<f64>>() {
        simulation::noise::poisson_noise_3d(arr.as_array(), scale, seed, axis)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}

/// Apply Poisson noise on a 3-dimensional array.
///
/// Applies Poisson noise (_i.e._ shot noise) on a 3-dimensional array of data.
/// An element-wise lambda value (scaled by the `scale` parameter) is used to
/// simulate Poisson noise with variable signal strength.
///
/// This function mutates the input array and does not create a new array.
///
/// Args:
///     data: The input 3-dimensional array to mutate.
///     scale: The scale factor.
///     seed: Pseudorandom number generator seed. Set the `seed` value to apply
///         homogenous noise to the input array. If `None`, then heterogenous
///         noise is applied to the input array.
///     axis: The signal data axis. If `None`, then `axis = 2`.
#[pyfunction]
#[pyo3(name = "poisson_noise_3d_mut")]
#[pyo3(signature = (data, scale, seed=None, axis=None))]
pub fn noise_poisson_noise_3d_mut(
    mut data: PyReadwriteArray3<f64>,
    scale: f64,
    seed: Option<u64>,
    axis: Option<usize>,
) {
    let arr = data.as_array_mut();
    simulation::noise::poisson_noise_3d_mut(arr, scale, seed, axis);
}
