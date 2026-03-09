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

/// Compute the Pearson correlation coefficient between two n-dimensional images
/// and a ROI map.
///
/// Computes the Pearson correlation coefficient, a measure of linear
/// correlation between two sets of n-dimensional images and a ROI map. This
/// function iterates through each ROI in the map and computes the correlation
/// coefficient. Returning a `HashMap` of Pearson correlation coefficient values
/// and ROI label IDs.
///
/// Args:
///     data_a: The first n-dimensional image for Pearson colocalization
///         analysis.
///     data_b: the second n-dimensional image for Pearson colocalization
///         analysis.
///     rois: A map of point clouds representing Regions of Interest (ROIs).
///         The individual ROIs must have the same dimensionality as the input
///         data.
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
///
/// Returns:
///     A `HashMap` where the keys are the ROI label IDs and values are the
///     Pearson correlation coefficients for each ROI respectively.
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

/// Compute 2D colocalization strength with Spatially Adaptive Colocalization
/// Analysis (SACA).
///
/// Computes a pixel-wise *z-score* indicating colocalization and
/// anti-colocalization strength on 2D input images using the Spatially Adaptive
/// Colocalization Analysis (SACA) framework. Per pixel SACA utilizes a
/// propagation and separation strategy to adaptively expand a weighted
/// circular kernel that defines the pixel of consideration's neighborhood.
/// The pixels within the neighborhood are assigned weights based on their
/// distance from the center pixel (decreasing with distance), ranked and their
/// colocalization coefficient computed using Kendall's Tau-b rank correlation.
///
/// Args:
///     data_a: The 2D input image corresponding to the first channel.
///     data_b: The 2D input image corresponding to the second channel.
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
///     The pixel-wise *z-score* indicating colocalization or
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

/// Compute 3D colocalization strength with Spatially Adaptive Colocalization
/// Analysis (SACA).
///
/// Computes a pixel-wise *z-score* indicating colocalization and
/// anti-colocalization strength on 3D input images using the Spatially Adaptive
/// Colocalization Analysis (SACA) framework. Per pixel SACA utilizes a
/// propagation and separation strategy to adaptively expand a weighted
/// spherical kernel that defines the pixel of consideration's neighborhood.
/// The pixels within the neighborhood are assigned weights based on their
/// distance from the center pixel (decreasing with distance), ranked and
/// their colocalization coefficient computed using Kendall's Tau-b rank
/// correlation.
///
/// Args:
///     data_a: The 3D input image corresponding to the first channel.
///     data_b: The 3D input image corresponding to the second channel.
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
///     The pixel-wise *z-score* indicating colocalization or
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
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Create a significant pixel mask from a pixel-wise *z-score* array.
///
/// Creates a boolean image representing significant pixels (*i.e.* the mask) by
/// applying Bonferroni correction to adjust for multiple comparisons.
///
/// Args:
///     data: The pixel-wise *z-score* indicating colocalization or
///         anti-colocalization strength.
///     alpha: The significance level representing the maximum type I error
///         (*i.e.* false positive error) allowed. If `None` then
///         `alpha = 0.05`.
///     parallel: If `true`, parallel computation is used across multiple
///         threads. If `false`, sequential single-threaded computation is used.
///         If `None` then `parallel == false`.
///
/// Returns:
///     The significant pixel mask where `true` pixels represent significant
///     *z-score* values.
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
        Ok(output.into_pyarray(py))
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are f64.",
        ))
    }
}
