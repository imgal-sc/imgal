use std::collections::HashMap;

use ndarray::{Array2, ArrayBase, AsArray, Axis, Dimension, IxDyn, ViewRepr};
use rayon::prelude::*;

use crate::error::ImgalError;
use crate::statistics::pearson_correlation;
use crate::traits::numeric::AsNumeric;

/// Compute the Pearson correlation coefficient between two n-dimensional arrays
/// and a ROI map.
///
/// # Description
///
/// Computes the Pearson correlation coefficient, a measure of linear
/// correlation between two sets of n-dimensional arrays and a ROI map. This
/// function iterates through each ROI in the map and computes the correlation
/// coefficient. Returning a `HashMap` of Pearson correlation coefficient values
/// and ROI labels.
///
/// # Arguments
///
/// * `data_a`: The first n-dimensional array for Pearson colocalization
///   analysis.
/// * `data_b`: the second n-dimensional array for Pearson colocalization
///   analysis.
/// * `rois`: A HashMap of point clouds representing Regions of Interest (ROIs).
///   The individual ROIs must have the same dimensionality as the input data.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `Ok(HashMap<u64, f64>)`: A HashMap where the keys are the ROI labels and
///   values are the Pearson correlation coefficients for each ROI respectively.
/// * `Err(ImgalError)`: If `data_a.len() != data_b.len()`. If `data_a.len()` or
///   `data_b.len()` is <= 2.
pub fn pearson_roi_coloc<'a, T, A, D>(
    data_a: A,
    data_b: A,
    rois: &HashMap<u64, Array2<usize>>,
    parallel: bool,
) -> Result<HashMap<u64, f64>, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data_a: ArrayBase<ViewRepr<&'a T>, IxDyn> = data_a.into().into_dyn();
    let data_b: ArrayBase<ViewRepr<&'a T>, IxDyn> = data_b.into().into_dyn();
    let per_roi_pearson_corr = |k: u64, v: &Array2<usize>| -> Result<(u64, f64), ImgalError> {
        let n = v.dim().0;
        let mut buf_a: Vec<T> = Vec::with_capacity(n);
        let mut buf_b: Vec<T> = Vec::with_capacity(n);
        let roi_coords = v.lanes(Axis(1));
        roi_coords.into_iter().for_each(|p| {
            let pos_buf;
            let pos = if let Some(coord) = p.as_slice() {
                coord
            } else {
                pos_buf = p.to_vec();
                pos_buf.as_slice()
            };
            buf_a.push(data_a[IxDyn(pos)]);
            buf_b.push(data_b[IxDyn(pos)]);
        });
        let corr = pearson_correlation(&buf_a, &buf_b, false)?;
        Ok((k, corr))
    };
    if parallel {
        rois.into_par_iter()
            .map(|(&k, v)| per_roi_pearson_corr(k, v))
            .collect::<Result<HashMap<u64, f64>, ImgalError>>()
    } else {
        rois.into_iter()
            .map(|(&k, v)| per_roi_pearson_corr(k, v))
            .collect::<Result<HashMap<u64, f64>, ImgalError>>()
    }
}
