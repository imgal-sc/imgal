use std::collections::HashMap;

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::map_imgal_error;
use imgal::colocalization;

/// Compute the Pearson correlation coefficient between two n-dimensional arrays
/// and a ROI map.
///
/// Computes the Pearson correlation coefficient, a measure of linear
/// correlation between two sets of n-dimensional arrays and a ROI map. This
/// function iterates through each ROI in the map and computes the correlation
/// coefficient. Returning a `HashMap` of Pearson correlation coefficient values
/// and ROI labels.
///
/// Args:
///     data_a: The first n-dimensional array for Pearson colocalization
///         analysis.
///     data_b: the second n-dimensional array for Pearson colocalization
///         analysis.
///     rois: A HashMap of point clouds representing Regions of Interest (ROIs).
///         The individual ROIs must have the same dimensionality as the input
///         data.
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
///
/// Returns:
///     A HashMap where the keys are the ROI labels and values are the Pearson
///     correlation coefficients for each ROI respectively.
#[pyfunction]
#[pyo3(name = "pearson_roi_coloc")]
#[pyo3(signature = (data_a, data_b, rois, parallel=None))]
pub fn colocalization_pearson_roi_coloc<'py>(
    py: Python<'py>,
    data_a: Bound<'py, PyAny>,
    data_b: Bound<'py, PyAny>,
    rois: HashMap<u64, Py<PyArray2<usize>>>,
    parallel: Option<bool>,
) -> PyResult<HashMap<u64, f64>> {
    let parallel = parallel.unwrap_or(false);
    let rois = rois
        .into_iter()
        .map(|(k, v)| {
            let arr = v.bind(py).try_readonly()?;
            Ok((k, arr.as_array().to_owned()))
        })
        .collect::<PyResult<HashMap<u64, Array2<usize>>>>()?;
    if let Ok(arr_a) = data_a.extract::<PyReadonlyArrayDyn<u8>>() {
        let arr_b = data_b.extract::<PyReadonlyArrayDyn<u8>>()?;
        colocalization::pearson_roi_coloc(arr_a.as_array(), arr_b.as_array(), &rois, parallel)
            .map(|output| output)
            .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArrayDyn<u16>>() {
        let arr_b = data_b.extract::<PyReadonlyArrayDyn<u16>>()?;
        colocalization::pearson_roi_coloc(arr_a.as_array(), arr_b.as_array(), &rois, parallel)
            .map(|output| output)
            .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArrayDyn<u64>>() {
        let arr_b = data_b.extract::<PyReadonlyArrayDyn<u64>>()?;
        colocalization::pearson_roi_coloc(arr_a.as_array(), arr_b.as_array(), &rois, parallel)
            .map(|output| output)
            .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArrayDyn<i64>>() {
        let arr_b = data_b.extract::<PyReadonlyArrayDyn<i64>>()?;
        colocalization::pearson_roi_coloc(arr_a.as_array(), arr_b.as_array(), &rois, parallel)
            .map(|output| output)
            .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArrayDyn<f32>>() {
        let arr_b = data_b.extract::<PyReadonlyArrayDyn<f32>>()?;
        colocalization::pearson_roi_coloc(arr_a.as_array(), arr_b.as_array(), &rois, parallel)
            .map(|output| output)
            .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArrayDyn<f64>>() {
        let arr_b = data_b.extract::<PyReadonlyArrayDyn<f64>>()?;
        colocalization::pearson_roi_coloc(arr_a.as_array(), arr_b.as_array(), &rois, parallel)
            .map(|output| output)
            .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

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
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
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
#[pyo3(signature = (data_a, data_b, threshold_a, threshold_b, parallel=None))]
pub fn colocalization_saca_2d<'py>(
    py: Python<'py>,
    data_a: Bound<'py, PyAny>,
    data_b: Bound<'py, PyAny>,
    threshold_a: f64,
    threshold_b: f64,
    parallel: Option<bool>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let parallel = parallel.unwrap_or(false);
    if let Ok(arr_a) = data_a.extract::<PyReadonlyArray2<u8>>() {
        let arr_b = data_b.extract::<PyReadonlyArray2<u8>>()?;
        colocalization::saca_2d(
            arr_a.as_array(),
            arr_b.as_array(),
            threshold_a as u8,
            threshold_b as u8,
            parallel,
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
            parallel,
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
            parallel,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArray2<i64>>() {
        let arr_b = data_b.extract::<PyReadonlyArray2<i64>>()?;
        colocalization::saca_2d(
            arr_a.as_array(),
            arr_b.as_array(),
            threshold_a as i64,
            threshold_b as i64,
            parallel,
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
            parallel,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArray2<f64>>() {
        let arr_b = data_b.extract::<PyReadonlyArray2<f64>>()?;
        colocalization::saca_2d(
            arr_a.as_array(),
            arr_b.as_array(),
            threshold_a,
            threshold_b,
            parallel,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
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
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
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
#[pyo3(signature = (data_a, data_b, threshold_a, threshold_b, parallel=None))]
pub fn colocalization_saca_3d<'py>(
    py: Python<'py>,
    data_a: Bound<'py, PyAny>,
    data_b: Bound<'py, PyAny>,
    threshold_a: f64,
    threshold_b: f64,
    parallel: Option<bool>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let parallel = parallel.unwrap_or(false);
    if let Ok(arr_a) = data_a.extract::<PyReadonlyArray3<u8>>() {
        let arr_b = data_b.extract::<PyReadonlyArray3<u8>>()?;
        colocalization::saca_3d(
            arr_a.as_array(),
            arr_b.as_array(),
            threshold_a as u8,
            threshold_b as u8,
            parallel,
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
            parallel,
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
            parallel,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArray3<i64>>() {
        let arr_b = data_b.extract::<PyReadonlyArray3<i64>>()?;
        colocalization::saca_3d(
            arr_a.as_array(),
            arr_b.as_array(),
            threshold_a as i64,
            threshold_b as i64,
            parallel,
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
            parallel,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else if let Ok(arr_a) = data_a.extract::<PyReadonlyArray3<f64>>() {
        let arr_b = data_b.extract::<PyReadonlyArray3<f64>>()?;
        colocalization::saca_3d(
            arr_a.as_array(),
            arr_b.as_array(),
            threshold_a,
            threshold_b,
            parallel,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(map_imgal_error)
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
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
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
///
/// Returns:
///     The significant pixel mask where `true` pixels represent significant
///     _z-score_ values.
///
/// Reference:
///     <https://doi.org/10.1109/TIP.2019.2909194>
#[pyfunction]
#[pyo3(name = "saca_significance_mask")]
#[pyo3(signature = (data, alpha=None, parallel=None))]
pub fn colocalization_saca_significance_mask<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    alpha: Option<f64>,
    parallel: Option<bool>,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    let parallel = parallel.unwrap_or(false);
    if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
        let output = colocalization::saca_significance_mask(arr.as_array(), alpha, parallel);
        return Ok(output.into_pyarray(py));
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are f64.",
        ));
    }
}
