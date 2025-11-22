use numpy::{PyReadonlyArrayDyn, PyReadwriteArray1};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::statistics;

/// Compute the effective sample size (ESS) of a weighted sample set.
///
/// This function computes the effective sample size (ESS) of a weighted sample
/// set. Only the weights of the associated sample set are needed. The ESS is
/// defined as:
///
/// ESS = (Σ wᵢ)² / Σ (wᵢ²)
///
/// :param weights: A slice of non-negative weights where each element represents
///     the weight of an associated sample.
/// :return: The effective number of independent samples.
#[pyfunction]
#[pyo3(name = "effective_sample_size")]
pub fn statistics_effective_sample_size(weights: Vec<f64>) -> f64 {
    statistics::effective_sample_size(&weights)
}

/// Find the maximum value in an n-dimensional array.
///
/// This function iterates through all elements of an n-dimensional array to
/// determine the maximum value.
///
/// :param data: The input n-dimensional array view.
/// :return: The maximum value in the input data array.
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
/// This function iterates through all elements of an n-dimensional array to
/// determine the minimum value.
///
/// :param data: The input n-dimensional array view.
/// :return: The minimum value in the input data array.
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
/// This function iterates through all elements of an n-dimensional array to
/// determine the minimum and maximum values.
///
/// :param data: The input n-dimensional array view.
/// :return: A tuple containing the minimum and maximum values (_i.e._
///     (min, max)) in the given array. If the array is empty a minimum and
///     maximum value of 0 is returned in the tuple.
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

/// Compute the sum of a sequence of numbers.
///
/// :param data: The sequence of numbers.
/// :return: The sum.
#[pyfunction]
#[pyo3(name = "sum")]
pub fn statistics_sum(data: Vec<f64>) -> f64 {
    statistics::sum(&data)
}

/// Compute the weighted Kendall's Tau-b rank correlation coefficient.
///
/// This function calculates a weighted Kendall's Tau-b rank correlation
/// coefficient between two datasets. This implementation uses a weighted merge
/// sort to count discordant pairs (inversions), and applies tie corrections for
/// both variables to compute the final Tau-b coefficient. Here the weighted
/// observations contribute unequally to the final correlation coefficient.
///
/// The weighted Kendall's Tau-b is calculated using:
///
/// τ_b = (C - D) / √((n₀ - n₁)(n₀ - n₂))
///
/// Where:
/// - `C` = number of weighted concordant pairs
/// - `D` = number of weighted discordant pairs
/// - `n₀` = total weighted pairs = `(Σwᵢ)² - Σwᵢ²`
/// - `n₁` = weighted tie correction for first variable
/// - `n₂` = weighted tie correction for second variable
///
/// :param data_a: The first dataset for correlation analysis. Must be the same
///     length as `data_b`.
/// :param data_b: The second dataset for correlation analysis. Must be the same
///     length as `data_a`.
/// :param weights: The associated weights for each observation pait. Must be the
///     same length as both input datasets.
/// :return: The weighted Kendall's Tau-b correlation coefficient, ranging
///     between -1.0 (negative correlation), 0.0 (no correlation) and 1.0
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
/// This function performs a bottom up merge sort on the input 1-dimensional
/// data array along with it's associated weights. Both the "data" and "weights"
/// arrays are mutated during the sorting. The output of this function is a
/// weighted inversion count.
///
/// :param data: A 1-dimensional array/slice of numbers of the same length as
///    "weights".
/// :param weights: A 1-dimensional array/slice of weights of the same length as
///    "data".
/// :return: The number of swaps needed to sort the input array.
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
