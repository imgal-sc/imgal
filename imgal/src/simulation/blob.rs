use ndarray::{Array, ArrayBase, AsArray, Dimension, Ix1, Ix2, ViewRepr};

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
pub fn blobs<'a, T, A, B, C, D>(
    centers: A,
    radii: B,
    intensities: C,
    background_value: T,
    shape: &[usize],
) -> Result<Array<T, D>, ImgalError>
where
    A: AsArray<'a, usize, Ix2>,
    B: AsArray<'a, usize, Ix1>,
    C: AsArray<'a, T, Ix1>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let cen_view: ArrayBase<ViewRepr<&'a usize>, Ix2> = centers.into();
    let rad_view: ArrayBase<ViewRepr<&'a usize>, Ix1> = radii.into();
    let int_view: ArrayBase<ViewRepr<&'a T>, Ix1> = intensities.into();
    if cen_view.len() != rad_view.len() {
        return Err(ImgalError::MismatchedArrayLengths {
            a_arr_name: "centers",
            a_arr_len: cen_view.len(),
            b_arr_name: "radii",
            b_arr_len: rad_view.len(),
        });
    };
    if cen_view.len() != int_view.len() {
        return Err(ImgalError::MismatchedArrayLengths {
            a_arr_name: "centers",
            a_arr_len: cen_view.len(),
            b_arr_name: "radii",
            b_arr_len: int_view.len(),
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
