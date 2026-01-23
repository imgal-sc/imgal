use ndarray::{Array, Dimension};

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

/// Create an n-dimensional blobs image.
///
/// # Description
///
/// todo
///
/// # Arguments
///
/// * `centers`:
/// * `radii`:
/// * `blob_values`:
/// * `background_value`:
/// * `shape`:
///
/// # Returns
///
/// * `Ok(Array<T, D>)`:
/// * `Err(ImgalError)`:
pub fn blobs<T, D>(
    centers: &[usize],
    radii: &[usize],
    blob_values: &[T],
    background_value: T,
    shape: &[usize],
) -> Result<Array<T, D>, ImgalError>
where
    D: Dimension,
    T: AsNumeric,
{
    if centers.len() != radii.len() {
        return Err(ImgalError::MismatchedArrayLengths {
            a_arr_name: "centers",
            a_arr_len: centers.len(),
            b_arr_name: "radii",
            b_arr_len: radii.len(),
        });
    };
    if centers.len() != blob_values.len() {
        return Err(ImgalError::MismatchedArrayLengths {
            a_arr_name: "centers",
            a_arr_len: centers.len(),
            b_arr_name: "radii",
            b_arr_len: radii.len(),
        });
    }

    // create output array
    let blob_arr = Array::from_elem(shape, T::default());
    // centers
    //     .iter()
    //     .zip(radii.iter())
    //     .zip(blob_values.iter())
    //     .for_each(|()| {});
    // iterator to draw circles with defines values
    // return array
    todo!();
}
