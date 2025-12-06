use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyArrayDyn, PyReadonlyArray2, PyReadonlyArray3,
    PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::colocalization;

/// Compute 2-dimensional colocalization strength with Spatially Adaptive
/// Colocalization Analysis (SACA).
///
/// Computes a pixel-wise _z-score_ indicating colocalization and
/// anti-colocalization strength on 2-dimensional input images using the
/// Spatially Adaptive Colocalization Analysis (SACA) framework. Per pixel SACA
/// utilizes a propagation and separation strategy to adaptively expand a
/// weighted circular kernel that defines the pixel of consideration's
/// neighborhood. The pixels within the neighborhood are assigned weights based
/// on their distance from the center pixel (decreasing with distance), ranked
/// and their colocalization coefficient computed using Kendall's Tau-b rank
/// correlation.
///
/// Args:
///     data_a: A 2-dimensional input image to measure colocalization strength,
///         with the same shape as `data_b`.
///     data_b: A 2-dimensional input image to measure colocalization strength,
///         with the same shape as `data_a`.
///     threshold_a: Pixel intensity threshold value for `data_a`. Pixels below
///         this value are given a weight of `0.0` if the pixel is in the
///         circular neighborhood.
///     threshold_b: Pixel intensity threshold value for `data_b`. Pixels below
///         this value are given a weight of `0.0` if the pixel is in the
///         circular neighborhood.
///
/// Returns:
///     The pixel-wise _z-score_ indicating colocalization or
///     anti-colocalization by its sign and the degree or strength of the
///     relationship through its absolute values.
///
/// Reference:
///     <https://doi.org/10.1109/TIP.2019.2909194>
#[pyfunction]
#[pyo3(name = "saca_2d")]
pub fn colocalization_saca_2d<'py>(
    py: Python<'py>,
    data_a: Bound<'py, PyAny>,
    data_b: Bound<'py, PyAny>,
    threshold_a: f64,
    threshold_b: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if let Ok(arr_a) = data_a.extract::<PyReadonlyArray2<u8>>() {
        let arr_b = data_b.extract::<PyReadonlyArray2<u8>>()?;
        colocalization::saca_2d(
            arr_a.as_array(),
            arr_b.as_array(),
            threshold_a as u8,
            threshold_b as u8,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArray2<u16>>() {
        let arr_b = data_b.extract::<PyReadonlyArray2<u16>>()?;
        colocalization::saca_2d(
            arr_a.as_array(),
            arr_b.as_array(),
            threshold_a as u16,
            threshold_b as u16,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArray2<u64>>() {
        let arr_b = data_b.extract::<PyReadonlyArray2<u64>>()?;
        colocalization::saca_2d(
            arr_a.as_array(),
            arr_b.as_array(),
            threshold_a as u64,
            threshold_b as u64,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArray2<f32>>() {
        let arr_b = data_b.extract::<PyReadonlyArray2<f32>>()?;
        colocalization::saca_2d(
            arr_a.as_array(),
            arr_b.as_array(),
            threshold_a as f32,
            threshold_b as f32,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArray2<f64>>() {
        let arr_b = data_b.extract::<PyReadonlyArray2<f64>>()?;
        colocalization::saca_2d(arr_a.as_array(), arr_b.as_array(), threshold_a, threshold_b)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}

/// Compute 3-dimensional colocalization strength with Spatially Adaptive
/// Colocalization Analysis (SACA).
///
/// Computes a pixel-wise _z-score_ indicating colocalization and
/// anti-colocalization strength on 2-dimensional input images using the
/// Spatially Adaptive Colocalization Analysis (SACA) framework. Per pixel SACA
/// utilizes a propagation and separation strategy to adaptively expand a
/// weighted circular kernel that defines the pixel of consideration's
/// neighborhood. The pixels within the neighborhood are assigned weights based
/// on their distance from the center pixel (decreasing with distance), ranked
/// and their colocalization coefficient computed using Kendall's Tau-b rank
/// correlation.
///
/// Args:
///     data_a: A 3-dimensional input image to measure colocalization strength,
///         with the same shape as `data_b`.
///     data_b: A 3-dimensional input image to measure colocalization strength,
///         with the same shape as `data_a`.
///     threshold_a: Pixel intensity threshold value for `data_a`. Pixels below
///         this value are given a weight of `0.0` if the pixel is in the
///         circular neighborhood.
///     threshold_b: Pixel intensity threshold value for `data_b`. Pixels below
///         this value are given a weight of `0.0` if the pixel is in the
///         circular neighborhood.
///
/// Returns:
///     The pixel-wise _z-score_ indicating colocalization or
///     anti-colocalization by its sign and the degree or strength of the
///     relationship through its absolute values.
///
/// Reference:
///     <https://doi.org/10.1109/TIP.2019.2909194>
#[pyfunction]
#[pyo3(name = "saca_3d")]
pub fn colocalization_saca_3d<'py>(
    py: Python<'py>,
    data_a: Bound<'py, PyAny>,
    data_b: Bound<'py, PyAny>,
    threshold_a: f64,
    threshold_b: f64,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    if let Ok(arr_a) = data_a.extract::<PyReadonlyArray3<u8>>() {
        let arr_b = data_b.extract::<PyReadonlyArray3<u8>>()?;
        colocalization::saca_3d(
            arr_a.as_array(),
            arr_b.as_array(),
            threshold_a as u8,
            threshold_b as u8,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArray3<u16>>() {
        let arr_b = data_b.extract::<PyReadonlyArray3<u16>>()?;
        colocalization::saca_3d(
            arr_a.as_array(),
            arr_b.as_array(),
            threshold_a as u16,
            threshold_b as u16,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArray3<u64>>() {
        let arr_b = data_b.extract::<PyReadonlyArray3<u64>>()?;
        colocalization::saca_3d(
            arr_a.as_array(),
            arr_b.as_array(),
            threshold_a as u64,
            threshold_b as u64,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArray3<f32>>() {
        let arr_b = data_b.extract::<PyReadonlyArray3<f32>>()?;
        colocalization::saca_3d(
            arr_a.as_array(),
            arr_b.as_array(),
            threshold_a as f32,
            threshold_b as f32,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArray3<f64>>() {
        let arr_b = data_b.extract::<PyReadonlyArray3<f64>>()?;
        colocalization::saca_3d(arr_a.as_array(), arr_b.as_array(), threshold_a, threshold_b)
            .map(|output| output.into_pyarray(py))
            .map_err(map_imgal_error)
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}

/// Create a significant pixel mask from a pixel-wise _z-score_ array.
///
/// Creates a boolean array representing significant pixels (_i.e._ the mask) by
/// applying Bonferroni correction to adjust for multiple comparisons.
///
/// Args:
///     data: The pixel-wise _z-score_ indicating colocalization or
///         anti-colocalization strength.
///     alpha: The significance level representing the maximum type I error
///         (_i.e._ false positive error) allowed (default = 0.05).
///
/// Returns:
///     The significant pixel mask where `true` pixels represent significant
///     _z-score_ values.
///
/// Reference:
///     <https://doi.org/10.1109/TIP.2019.2909194>
#[pyfunction]
#[pyo3(name = "saca_significance_mask")]
#[pyo3(signature = (data, alpha=None))]
pub fn colocalization_saca_significance_mask<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    alpha: Option<f64>,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        let output = colocalization::saca_significance_mask(arr.as_array(), alpha);
        return Ok(output.into_pyarray(py));
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are f64.",
        ));
    }
}
