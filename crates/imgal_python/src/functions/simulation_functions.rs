use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyReadwriteArrayDyn,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::simulation;

/// Create an n-dimensional Gaussian metaballs image.
///
/// Creates a simulated n-dimensional blobs image using a variant of Jim Blinn's
/// metaballs blob simulation algorithm. Metaballs are n-dimensional blob
/// isosurfaces that are able to interact with each other. This function uses a
/// Gaussian falloff strategy to simulate a smooth and continuous blob border
/// with no sharp edges.
///
/// Args:
///     centers: A 2D array with `(p, D)`, where `p` is the number of blobs and
///         `D` is the number of dimensions.
///     radii: A 1D array where each element represents a blob radius.
///     intensities: A 1D array where each element represents a blob intensity.
///     falloffs: A 1D array where each element represents the "falloff" value
///         for a given blob that controls the rate of intensity decay from the
///         blob center. High values result in a more blured border effect and
///         low values have a more defined border.
///     background: The background intensity value for the image.
///     shape: The shape of the output n-dimensional array.
///
/// Returns:
///     An n-dimensional image containing the metaballs blob simulation, where
///     each pixel value is the *sum* of Gaussian contributions from each blob
///     and the background.
#[pyfunction]
#[pyo3(name = "gaussian_metaballs")]
pub fn blob_gaussian_metaballs<'py>(
    py: Python<'py>,
    centers: Bound<'py, PyAny>,
    radii: Vec<f64>,
    intensities: Vec<f64>,
    falloffs: Vec<f64>,
    background: f64,
    shape: Vec<usize>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    if let Ok(cen_arr) = centers.extract::<PyReadonlyArray2<u8>>() {
        simulation::blob::gaussian_metaballs(
            cen_arr.as_array(),
            &radii.iter().map(|&v| v as u8).collect::<Vec<u8>>(),
            &intensities.iter().map(|&v| v as u8).collect::<Vec<u8>>(),
            &falloffs.iter().map(|&v| v as u8).collect::<Vec<u8>>(),
            background as u8,
            &shape,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(cen_arr) = centers.extract::<PyReadonlyArray2<u16>>() {
        simulation::blob::gaussian_metaballs(
            cen_arr.as_array(),
            &radii.iter().map(|&v| v as u16).collect::<Vec<u16>>(),
            &intensities.iter().map(|&v| v as u16).collect::<Vec<u16>>(),
            &falloffs.iter().map(|&v| v as u16).collect::<Vec<u16>>(),
            background as u16,
            &shape,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(cen_arr) = centers.extract::<PyReadonlyArray2<u64>>() {
        simulation::blob::gaussian_metaballs(
            cen_arr.as_array(),
            &radii.iter().map(|&v| v as u64).collect::<Vec<u64>>(),
            &intensities.iter().map(|&v| v as u64).collect::<Vec<u64>>(),
            &falloffs.iter().map(|&v| v as u64).collect::<Vec<u64>>(),
            background as u64,
            &shape,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(cen_arr) = centers.extract::<PyReadonlyArray2<i64>>() {
        simulation::blob::gaussian_metaballs(
            cen_arr.as_array(),
            &radii.iter().map(|&v| v as i64).collect::<Vec<i64>>(),
            &intensities.iter().map(|&v| v as i64).collect::<Vec<i64>>(),
            &falloffs.iter().map(|&v| v as i64).collect::<Vec<i64>>(),
            background as i64,
            &shape,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(cen_arr) = centers.extract::<PyReadonlyArray2<f32>>() {
        simulation::blob::gaussian_metaballs(
            cen_arr.as_array(),
            &radii.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
            &intensities.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
            &falloffs.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
            background as f32,
            &shape,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(cen_arr) = centers.extract::<PyReadonlyArray2<f64>>() {
        simulation::blob::gaussian_metaballs(
            cen_arr.as_array(),
            &radii,
            &intensities,
            &falloffs,
            background,
            &shape,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Create an n-dimensional logistic metaballs image.
///
/// Creates a simulated n-dimensional blobs image using a variant of Jim Blinn's
/// metaballs blob simulation algorithm. Metaballs are n-dimensional blob
/// isosurfaces that are able to interact with each other. This function uses a
/// logistic (sigmoid) falloff function to simulate smooth and crisp blob
/// borders. Logistic metaballs, unlike traditional metaballs, do not fuse
/// together but instead deform against neighboring blobs.
///
/// Args:
///     centers: A 2D array with `(p, D)`, where `p` is the number of blobs and
///         `D` is the number of dimensions.
///     radii: A 1D array where each element represents a blob radius.
///     intensities: A 1D array where each element represents a blob intensity.
///     falloffs: A 1D array where each element represents the "falloff" value
///         for a given blob that controls the value transition steepness from
///         the center of the blob to the edge. High values result in longer
///         transitions to the background, creating larger or inflated blobs.
///         Low values result in short or rapid transitions to the background,
///         creating crisp edges.
///     background: The background intensity value for the image.
///     shape: The shape of the output n-dimensional array.
///
/// Returns:
///     An n-dimensional image containing the metaballs blob simulation, where
///     each pixel value is the *maximum* contribution of any blob at that
///     position.
#[pyfunction]
#[pyo3(name = "logistic_metaballs")]
pub fn blob_logistic_metaballs<'py>(
    py: Python<'py>,
    centers: Bound<'py, PyAny>,
    radii: Vec<f64>,
    intensities: Vec<f64>,
    falloffs: Vec<f64>,
    background: f64,
    shape: Vec<usize>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    if let Ok(cen_arr) = centers.extract::<PyReadonlyArray2<u8>>() {
        simulation::blob::logistic_metaballs(
            cen_arr.as_array(),
            &radii.iter().map(|&v| v as u8).collect::<Vec<u8>>(),
            &intensities.iter().map(|&v| v as u8).collect::<Vec<u8>>(),
            &falloffs.iter().map(|&v| v as u8).collect::<Vec<u8>>(),
            background as u8,
            &shape,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(cen_arr) = centers.extract::<PyReadonlyArray2<u16>>() {
        simulation::blob::logistic_metaballs(
            cen_arr.as_array(),
            &radii.iter().map(|&v| v as u16).collect::<Vec<u16>>(),
            &intensities.iter().map(|&v| v as u16).collect::<Vec<u16>>(),
            &falloffs.iter().map(|&v| v as u16).collect::<Vec<u16>>(),
            background as u16,
            &shape,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(cen_arr) = centers.extract::<PyReadonlyArray2<u64>>() {
        simulation::blob::logistic_metaballs(
            cen_arr.as_array(),
            &radii.iter().map(|&v| v as u64).collect::<Vec<u64>>(),
            &intensities.iter().map(|&v| v as u64).collect::<Vec<u64>>(),
            &falloffs.iter().map(|&v| v as u64).collect::<Vec<u64>>(),
            background as u64,
            &shape,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(cen_arr) = centers.extract::<PyReadonlyArray2<i64>>() {
        simulation::blob::logistic_metaballs(
            cen_arr.as_array(),
            &radii.iter().map(|&v| v as i64).collect::<Vec<i64>>(),
            &intensities.iter().map(|&v| v as i64).collect::<Vec<i64>>(),
            &falloffs.iter().map(|&v| v as i64).collect::<Vec<i64>>(),
            background as i64,
            &shape,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(cen_arr) = centers.extract::<PyReadonlyArray2<f32>>() {
        simulation::blob::logistic_metaballs(
            cen_arr.as_array(),
            &radii.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
            &intensities.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
            &falloffs.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
            background as f32,
            &shape,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(cen_arr) = centers.extract::<PyReadonlyArray2<f64>>() {
        simulation::blob::logistic_metaballs(
            cen_arr.as_array(),
            &radii,
            &intensities,
            &falloffs,
            background,
            &shape,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Create a 1D Gaussian IRF convolved monoexponential or multiexponential decay
/// curve.
///
/// Creates a 1D Gaussian instrument response function (IRF) convolved with an
/// ideal monoexponential or multiexponential decay curve defined as the sum of
/// one or more exponential components, each characterized by a lifetime (τ) and
/// fractional intensity:
///
/// ```text
/// I(t) = [Σᵢ αᵢ × exp(-t/τᵢ)] ⊗ IRF(t)
/// ```
///
/// Args:
///     samples: The number of discrete points that make up the decay curve.
///     period: The period (*i.e.* time interval).
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
///     total_counts: The total intensity count (*e.g.* photon count) of the
///         decay curve.
///     irf_center: The temporal position of the IRF peak within the time range.
///     irf_width: The full width at half maximum (FWHM) of the IRF.
///
/// Returns:
///     The 1D Gaussian IRF convolved monoexponential or multiexponential decay
///     curve.
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

/// Create a 3D Gaussian IRF convolved monoexponential or multiexponential decay
/// curve.
///
/// Creates a 3D Gaussian instrument response function (IRF) convolved with an
/// ideal monoexponential or multiexponential decay curve defined as the sum of
/// one or more exponential components, each characterized by a lifetime (τ) and
/// fractional intensity:
///
/// ```text
/// I(t) = [Σᵢ αᵢ × exp(-t/τᵢ)] ⊗ IRF(t)
/// ```
///
/// Args:
///     samples: The number of discrete points that make up the decay curve.
///     period: The period (*i.e.* time interval).
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
///     total_counts: The total intensity count (*e.g.* photon count) of the
///         decay curve.
///     irf_center: The temporal position of the IRF peak within the time range.
///     irf_width: The full width at half maximum (FWHM) of the IRF.
///     shape: The row and col shape to broadcast the decay curve into.
///
/// Returns:
///     The 3D Gaussian IRF convolved monoexponential or multiexponential decay
///     curve with dimension (row, col, t).
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

/// Create a 1D ideal monoexponential or multiexponential decay curve.
///
/// Creates a 1D ideal exponential decay curve by computing the sum of one or
/// more exponential components, each characterized by a lifetime (τ) and
/// fractional intensity as defined by:
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
///     period: The period (*i.e.* time interval).
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
///     total_counts: The total intensity count (*e.g.* photon count) of the
///         decay curve.
///
/// Returns:
///     The 1D monoexponential or multiexponential decay curve.
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

/// Create a 3D ideal monoexponential or multiexponential decay curve.
///
/// Creates a 3D ideal exponential decay curve by computing the sum of one or
/// more exponential components, each characterized by a lifetime (τ) and
/// fractional intensity as defined by:
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
///     period: The period (*i.e.* time interval).
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
///     total_counts: The total intensity count (*e.g.* photon count) of the
///         decay curve.
///     shape: The row and col shape to broadcast the decay curve into.
///
/// Returns:
///     The 3D monoexponential or multiexponential decay curve with dimensions
///     (row, col, t).
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

/// Create a 1D IRF convolved monoexponential or multiexponential decay curve.
///
/// Creates a 1D instrument response function (IRF) convolved with an ideal
/// monoexponential or multiexponential decay curve defined as the sum of one or
/// more exponential components, each characterized by a lifetime (τ) and
/// fractional intensity:
///
/// ```text
/// I(t) = [Σᵢ αᵢ × exp(-t/τᵢ)] ⊗ IRF(t)
/// ```
///
/// Args:
///     irf: The IRF as a 1D array.
///     samples: The number of discrete points that make up the decay curve.
///     period: The period (*i.e.* time interval).
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
///     total_counts: The total intensity count (*e.g.* photon count) of the
///         decay curve.
///
/// Returns:
///     The 1D IRF convolved monoexponential or multiexponential decay curve.
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

/// Create a 3D IRF convolved monoexponential or multiexponential decay curve.
///
/// Creates a 3D instrument response function (IRF) convolved with an ideal
/// monoexponential or multiexponential decay curve defined as the sum of one or
/// more exponential components, each characterized by a lifetime (τ) and
/// fractional intensity:
///
/// ```text
/// I(t) = [Σᵢ αᵢ × exp(-t/τᵢ)] ⊗ IRF(t)
/// ```
///
/// Args:
///     irf: The IRF as a 1D array.
///     samples: The number of discrete points that make up the decay curve.
///     period: The period (*i.e.* time interval).
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
///     total_counts: The total intensity count (*e.g.* photon count) of the
///         decay curve.
///     shape: The row and col shape to broadcast the decay curve into.
///
/// Returns:
///     The 3D IRF convolved monoexponential or multiexponential decay curve
///     with dimensions (row, col, t).
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

/// Create a 2D image with a linear gradient.
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
///     The 2D gradient image.
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

/// Create a 3D image with a linear gradient.
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
///     The 3D gradient image.
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

/// Create a 1D Gaussian instrument response function (IRF).
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
///     The simulated 1D IRF curve array.
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

/// Create a new n-dimensional image with Poisson noise.
///
/// Creates a new n-dimensional image of the input data with scaled Poisson
/// noise (*i.e.* shot noise) using Knuth's algorithm.
///
/// Args:
///     data: The input n-dimensonal image.
///     scale: The noise scale factor. Smaller values produce noiser output,
///         while larger values produce output closer to the original input.
///     seed: The seed value for the pseudo-random number generator.
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
///
/// Returns:
///     An image of the same dimensions as the input `data`, where each element
///     is a Poisson-distributed sample derived from the corresponding input
///     value.
#[pyfunction]
#[pyo3(name = "poisson_noise")]
#[pyo3(signature = (data, scale, seed=None, parallel=None))]
pub fn noise_poisson_noise<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    scale: f64,
    seed: Option<u64>,
    parallel: Option<bool>,
) -> PyResult<Bound<'py, PyAny>> {
    let parallel = parallel.unwrap_or(false);
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        Ok(
            simulation::noise::poisson_noise(arr.as_array(), scale, seed, parallel)
                .into_pyarray(py)
                .into_any(),
        )
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        Ok(
            simulation::noise::poisson_noise(arr.as_array(), scale, seed, parallel)
                .into_pyarray(py)
                .into_any(),
        )
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        Ok(
            simulation::noise::poisson_noise(arr.as_array(), scale, seed, parallel)
                .into_pyarray(py)
                .into_any(),
        )
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i64>>() {
        Ok(
            simulation::noise::poisson_noise(arr.as_array(), scale, seed, parallel)
                .into_pyarray(py)
                .into_any(),
        )
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        Ok(
            simulation::noise::poisson_noise(arr.as_array(), scale, seed, parallel)
                .into_pyarray(py)
                .into_any(),
        )
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        Ok(
            simulation::noise::poisson_noise(arr.as_array(), scale, seed, parallel)
                .into_pyarray(py)
                .into_any(),
        )
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Mutate an n-dimensional image with Poisson noise.
///
/// Mutates an n-dimensional image with scaled Poisson noise (*i.e.* shot noise)
/// using Knuth's algorithm.
///
/// Args:
///     data: The input n-dimensonal image to mutate.
///     scale: The noise scale factor. Smaller values produce noiser output,
///         while larger values produce output closer to the original input.
///     seed: The seed value for the pseudo-random number generator.
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
#[pyfunction]
#[pyo3(name = "poisson_noise_mut")]
#[pyo3(signature = (data, scale, seed=None, parallel=None))]
pub fn noise_poisson_noise_mut(
    data: Bound<PyAny>,
    scale: f64,
    seed: Option<u64>,
    parallel: Option<bool>,
) -> PyResult<()> {
    let parallel = parallel.unwrap_or(false);
    if let Ok(mut arr) = data.extract::<PyReadwriteArrayDyn<u8>>() {
        simulation::noise::poisson_noise_mut(arr.as_array_mut(), scale, seed, parallel);
        Ok(())
    } else if let Ok(mut arr) = data.extract::<PyReadwriteArrayDyn<u16>>() {
        simulation::noise::poisson_noise_mut(arr.as_array_mut(), scale, seed, parallel);
        Ok(())
    } else if let Ok(mut arr) = data.extract::<PyReadwriteArrayDyn<u64>>() {
        simulation::noise::poisson_noise_mut(arr.as_array_mut(), scale, seed, parallel);
        Ok(())
    } else if let Ok(mut arr) = data.extract::<PyReadwriteArrayDyn<i64>>() {
        simulation::noise::poisson_noise_mut(arr.as_array_mut(), scale, seed, parallel);
        Ok(())
    } else if let Ok(mut arr) = data.extract::<PyReadwriteArrayDyn<f32>>() {
        simulation::noise::poisson_noise_mut(arr.as_array_mut(), scale, seed, parallel);
        Ok(())
    } else if let Ok(mut arr) = data.extract::<PyReadwriteArrayDyn<f64>>() {
        simulation::noise::poisson_noise_mut(arr.as_array_mut(), scale, seed, parallel);
        Ok(())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}
