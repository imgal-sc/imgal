use ndarray::{Array, ArrayBase, AsArray, Dimension, ViewRepr, Zip};

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
