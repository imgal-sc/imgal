use ndarray::{ArrayBase, ArrayD, AsArray, Dimension, ViewRepr, Zip};

use crate::traits::numeric::AsNumeric;

/// Create a boolean mask from a threshold value.
///
/// # Description
///
/// Creates a threshold mask (as a boolean array) from the input image at the
/// given threshold value.
///
/// # Arguments
///
/// * `data`: The input n-dimensional image or array.
/// * `threshold`: The image pixel threshold value.
///
/// # Returns
///
/// * `ArrayD<bool>`: A boolean array of the same shape as the input image with
///   pixels that are greater than the threshold value set as `true` and pixels
///   that are below the threshold value set as `false`.
pub fn manual_mask<'a, T, A, D>(data: A, threshold: T) -> ArrayD<bool>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let mut mask = ArrayD::<bool>::default(view.shape());
    Zip::from(view.into_dyn())
        .and(&mut mask)
        .par_for_each(|&ip, mp| {
            *mp = ip >= threshold;
        });

    mask
}
