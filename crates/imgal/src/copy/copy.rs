use ndarray::{Array, ArrayBase, ArrayViewMut, AsArray, Dimension, ViewRepr, Zip};

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

/// Duplicate an array.
///
/// # Description
///
/// Duplicates a given array by allocating a new array and copying elements into
/// it.
///
/// # Arguments
///
/// * `data`: The input n-dimensional array to duplicate.
/// * `parallel`: If `true`, parallel copying of the inpu tdata is used across
///   multiple threads. If `false`, sequential single-threaded copying is used.
///
/// # Returns
///
/// * `Array<T, D>`: A duplicate of the input array.
pub fn duplicate<'a, T, A, D>(data: A, parallel: bool) -> Array<T, D>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    if parallel {
        let mut dup: Array<T, D> = Array::from_elem(data.dim(), T::default());
        Zip::from(data).and(dup.view_mut()).par_for_each(|&v, d| {
            *d = v;
        });
        dup
    } else {
        data.to_owned()
    }
}

/// Duplicate an array's data into an exisiting container.
///
/// # Description
///
/// Duplicates a given array into an exisiting container (*i.e.* another array
/// with the same dtype and shape).
///
/// # Arguments
///
/// * `data_a`: The input n-dimensional array to copy data from.
/// * `data_b`: The input n-dimensional array to copy data to.
/// * `parallel`: If `true`, parallel copying of the inpu tdata is used across
///   multiple threads. If `false`, sequential single-threaded copying is used.
///
/// # Returns
///
/// * `Err(ImgalError)`: If `data_a.shape() != data_b.shape()`.
pub fn duplicate_into<'a, T, A, D>(
    data_a: A,
    mut data_b: ArrayViewMut<T, D>,
    parallel: bool,
) -> Result<(), ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data_a: ArrayBase<ViewRepr<&'a T>, D> = data_a.into();
    if data_a.shape() != data_b.shape() {
        return Err(ImgalError::MismatchedArrayShapes {
            a_arr_name: "data_a",
            a_shape: data_a.shape().to_vec(),
            b_arr_name: "data_b",
            b_shape: data_b.shape().to_vec(),
        });
    }
    if parallel {
        Zip::from(data_a).and(data_b).par_for_each(|&a, b| {
            *b = a;
        });
        Ok(())
    } else {
        data_b.assign(&data_a);
        Ok(())
    }
}
