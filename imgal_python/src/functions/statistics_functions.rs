use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyReadwriteArray1};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::statistics;

/// Compute the effective sample size (ESS) of a weighted sample set.
///
/// Computes the effective sample size (ESS) of a weighted sample set. Only the
/// weights of the associated sample set are needed. The ESS is defined as:
///
/// ```text
/// ESS = (Σ wᵢ)² / Σ (wᵢ²)
/// ```
///
/// Args:
///     weights: A slice of non-negative weights where each element represents
///         the weight of an associated sample.
///
/// Returns:
///     The effective number of independent samples.
#[pyfunction]
#[pyo3(name = "effective_sample_size")]
pub fn statistics_effective_sample_size(weights: Vec<f64>) -> f64 {
    statistics::effective_sample_size(&weights)
}

/// Find the maximum value in an n-dimensional array.
///
/// Iterates through all elements of an n-dimensional array to determine the
/// maximum value.
///
/// Args:
///     data: The input n-dimensional array view.
///
/// Returns:
///     The maximum value in the input data array.
#[pyfunction]
#[pyo3(name = "max")]
pub fn statistics_max<'py>(data: Bound<'py, PyAny>) -> PyResult<f64> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        return Ok(statistics::max(arr.as_array()) as f64);
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        return Ok(statistics::max(arr.as_array()) as f64);
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        return Ok(statistics::max(arr.as_array()) as f64);
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        return Ok(statistics::max(arr.as_array()) as f64);
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(statistics::max(arr.as_array()));
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}

/// Find the minimum value in an n-dimensional array.
///
/// Iterates through all elements of an n-dimensional array to determine the
/// minimum value.
///
/// Args:
///     data: The input n-dimensional array view.
///
/// Returns:
///     The minimum value in the input data array.
#[pyfunction]
#[pyo3(name = "min")]
pub fn statistics_min<'py>(data: Bound<'py, PyAny>) -> PyResult<f64> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        return Ok(statistics::min(arr.as_array()) as f64);
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        return Ok(statistics::min(arr.as_array()) as f64);
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        return Ok(statistics::min(arr.as_array()) as f64);
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        return Ok(statistics::min(arr.as_array()) as f64);
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(statistics::min(arr.as_array()));
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}

/// Find the minimum and maximum values in an n-dimensional array.
///
/// Iterates through all elements of an n-dimensional array to determine the
/// minimum and maximum values.
///
/// Args:
///     data: The input n-dimensional array view.
///
/// Returns:
///     A tuple containing the minimum and maximum values (_i.e._ (min, max)) in
///     the given array.
#[pyfunction]
#[pyo3(name = "min_max")]
pub fn statistics_min_max<'py>(data: Bound<'py, PyAny>) -> PyResult<(f64, f64)> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        let mm = statistics::min_max(arr.as_array());
        return Ok((mm.0 as f64, mm.1 as f64));
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        let mm = statistics::min_max(arr.as_array());
        return Ok((mm.0 as f64, mm.1 as f64));
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        let mm = statistics::min_max(arr.as_array());
        return Ok((mm.0 as f64, mm.1 as f64));
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        let mm = statistics::min_max(arr.as_array());
        return Ok((mm.0 as f64, mm.1 as f64));
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        let mm = statistics::min_max(arr.as_array());
        return Ok((mm.0 as f64, mm.1 as f64));
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}

/// Compute the linear percentile over an n-dimensional array.
///
/// Calculates percentiles using linear interpolation between data points. The
/// computation can be performed either on the entire array (flattened) or along
/// a specified axis. The linear percentile is computed using:
///
/// ```text
/// h = (n - 1) × p
/// j = ⌊h⌋
/// γ = h - j
/// percentile = (1 - γ) × v[j] + γ × v[j+1]
/// ```
///
/// Where:
/// - `n` is the array length.
/// - `p` is the percentile in range `0` to `100`.
/// - `v[j]` is the value at index `j`.
/// - `⌊h⌋` is the floor function of `h`.
///
/// When `γ` is close to zero (within `epsilon`), the result is simply `v[j]`,
/// avoiding unnecessary interpolation.
///
/// Args:
///     data: An n-dimensional image or array.
///     percentile: The percentile value in thae range `0.0` to `100.0`. Values
///         out side this range will be clamped.
///     axis: The axis to compute percentiles along. If `None`, the input `data`
///         is flattened and a single percentile value is returned.
///     epsilon: The tolerance value used to decide the if the fractional index
///         is an integer. If `None`, then `epsilon = 1e-12`.
///
/// Returns:
///     The linear percentile of the input data. If `axis` is `None`, the result
///     shape is `(1,)` and contains a single percentile value of the flattened
///     input `data`. If `axis` is a valid axis value, the result has the same
///     shape as `data` with `axis` removed and contains the percentiles
///     calculated along `axis`.
#[pyfunction]
#[pyo3(name = "linear_percentile")]
#[pyo3(signature = (data, p, axis=None, epsilon=None))]
pub fn statistics_linear_percentile<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    p: f64,
    axis: Option<usize>,
    epsilon: Option<f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u8>>() {
        statistics::linear_percentile(arr.as_array(), p, axis, epsilon)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u16>>() {
        statistics::linear_percentile(arr.as_array(), p, axis, epsilon)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
        statistics::linear_percentile(arr.as_array(), p, axis, epsilon)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        statistics::linear_percentile(arr.as_array(), p, axis, epsilon)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        statistics::linear_percentile(arr.as_array(), p, axis, epsilon)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}

/// Compute the sum of the slice of numbers.
///
/// Computes the sum of numbers in the input slice.
///
/// Args:
///     data: A slice of numbers.
///
/// Returns:
///     The sum.
#[pyfunction]
#[pyo3(name = "sum")]
pub fn statistics_sum(data: Vec<f64>) -> f64 {
    statistics::sum(&data)
}

/// Compute the weighted Kendall's Tau-b rank correlation coefficient.
///
/// Calculates a weighted Kendall's Tau-b rank correlation coefficient between
/// two datasets. This implementation uses a weighted merge sort to count
/// discordant pairs (inversions), and applies tie corrections for both
/// variables to compute the final Tau-b coefficient. Here the weighted
/// observations contribute unequally to the final correlation coefficient.
///
/// The weighted Kendall's Tau-b is calculated using:
///
/// ```text
/// τ_b = (C - D) / √[(n₀ - n₁)(n₀ - n₂)]
/// ```
///
/// Where:
/// - `C` = number of weighted concordant pairs
/// - `D` = number of weighted discordant pairs
/// - `n₀` = total weighted pairs = `(Σwᵢ)² - Σwᵢ²`
/// - `n₁` = weighted tie correction for first variable
/// - `n₂` = weighted tie correction for second variable
///
/// Args:
///     data_a: The first dataset for correlation analysis. Must be the same
///         length as `data_b`.
///     data_b: The second dataset for correlation analysis. Must be the same
///         length as `data_a`.
///     weights: The associated weights for each observation pait. Must be the
///         same length as both input datasets.
///
/// Returns:
///     The weighted Kendall's Tau-b correlation coefficient, ranging between
///     `-1.0` (negative correlation), `0.0` (no correlation) and `1.0`
///     (positive correlation).
#[pyfunction]
#[pyo3(name = "weighted_kendall_tau_b")]
pub fn statistics_weighted_kendall_tau_b(
    data_a: Vec<f64>,
    data_b: Vec<f64>,
    weights: Vec<f64>,
) -> PyResult<f64> {
    statistics::weighted_kendall_tau_b(&data_a, &data_b, &weights)
        .map(|output| output)
        .map_err(map_imgal_error)
}

/// Sort 1-dimensional arrays of values and their associated weights.
///
/// Performs a bottom up merge sort on the input 1-dimensional data array along
/// with it's associated weights. Both the `data` and `weights` arrays are
/// _mutated_ during the sorting. The output of this function is a weighted
/// inversion count.
///
/// Args:
///     data: A 1-dimensional array/slice of numbers of the same length as
///         `weights`.
///     weights: A 1-dimensional array/slice of weights of the same length as
///         `data`.
///
/// Returns:
///     The number of swaps needed to sort the input array.
///
/// Reference:
///     <https://doi.org/10.1109/TIP.2019.2909194>
#[pyfunction]
#[pyo3(name = "weighted_merge_sort_mut")]
pub fn statistics_weighted_merge_sort_mut<'py>(
    data: Bound<'py, PyAny>,
    mut weights: PyReadwriteArray1<f64>,
) -> PyResult<f64> {
    // pattern match and extract the allowed array type
    if let Ok(mut d) = data.extract::<PyReadwriteArray1<u8>>() {
        return statistics::weighted_merge_sort_mut(
            d.as_slice_mut().unwrap(),
            weights.as_slice_mut().unwrap(),
        )
        .map(|output| output)
        .map_err(map_imgal_error);
    } else if let Ok(mut d) = data.extract::<PyReadwriteArray1<u16>>() {
        return statistics::weighted_merge_sort_mut(
            d.as_slice_mut().unwrap(),
            weights.as_slice_mut().unwrap(),
        )
        .map(|output| output)
        .map_err(map_imgal_error);
    } else if let Ok(mut d) = data.extract::<PyReadwriteArray1<u64>>() {
        return statistics::weighted_merge_sort_mut(
            d.as_slice_mut().unwrap(),
            weights.as_slice_mut().unwrap(),
        )
        .map(|output| output)
        .map_err(map_imgal_error);
    } else if let Ok(mut d) = data.extract::<PyReadwriteArray1<f32>>() {
        return statistics::weighted_merge_sort_mut(
            d.as_slice_mut().unwrap(),
            weights.as_slice_mut().unwrap(),
        )
        .map(|output| output)
        .map_err(map_imgal_error);
    } else if let Ok(mut d) = data.extract::<PyReadwriteArray1<f64>>() {
        return statistics::weighted_merge_sort_mut(
            d.as_slice_mut().unwrap(),
            weights.as_slice_mut().unwrap(),
        )
        .map(|output| output)
        .map_err(map_imgal_error);
    } else if let Ok(mut d) = data.extract::<PyReadwriteArray1<i32>>() {
        return statistics::weighted_merge_sort_mut(
            d.as_slice_mut().unwrap(),
            weights.as_slice_mut().unwrap(),
        )
        .map(|output| output)
        .map_err(map_imgal_error);
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}
