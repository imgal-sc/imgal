use ndarray::{ArrayD, ArrayViewD, Zip};

use crate::traits::numeric::AsNumeric;

/// Create a boolean mask from a threshold value.
///
/// # Description
///
/// This function computes a threshold mask (as a boolean array) from the input
/// image at the given threshold value.
///
/// # Arguments
///
/// * `data`: An n-dimensional image or array.
/// * `threshold`: The image pixel threshold value.
///
/// # Returns
///
/// * `ArrayD<bool>`: A boolean array of the same shape as the input image
///    with pixels that are greater than the threshold value set as `true`
///    and pixels that are below the threshold value set as `false`.
pub fn manual_mask<T>(data: ArrayViewD<T>, threshold: T) -> ArrayD<bool>
where
    T: AsNumeric,
{
    // create output mask of same shape and apply threshold
    let mut mask = ArrayD::<bool>::default(data.dim());
    Zip::from(data).and(&mut mask).par_for_each(|&ip, mp| {
        *mp = ip > threshold;
    });

    mask
}
