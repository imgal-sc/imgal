use ndarray::{Array, ArrayBase, AsArray, Dimension, ViewRepr, Zip};

use crate::prelude::*;

/// Create a boolean mask from a threshold value.
///
/// # Description
///
/// Creates a threshold mask (as a boolean array) from the input image at the
/// given threshold value.
///
/// # Arguments
///
/// * `data`: The input n-dimensional image.
/// * `threshold`: The image pixel threshold value.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
///
/// # Returns
///
/// * `Array<bool, D>`: A boolean image of the same shape as the input image
///   with pixels that are greater than the threshold value set as `true` and
///   pixels that are below the threshold value set as `false`.
#[inline]
pub fn manual_mask<'a, T, A, D>(data: A, threshold: T, threads: Option<usize>) -> Array<bool, D>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let mask_apply_seq = || {
        let mut mask = Array::from_elem(data.dim(), false);
        Zip::from(data.view()).and(&mut mask).for_each(|&ip, mp| {
            *mp = ip >= threshold;
        });
        mask
    };
    let mask_apply_par = || {
        let mut mask = Array::from_elem(data.dim(), false);
        Zip::from(data.view())
            .and(&mut mask)
            .par_for_each(|&ip, mp| {
                *mp = ip >= threshold;
            });
        mask
    };
    par!(threads, seq_exp: mask_apply_seq(), par_exp: mask_apply_par())
}
