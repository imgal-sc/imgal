use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr};
use rayon::prelude::*;

use crate::traits::numeric::AsNumeric;

/// Compute the sum of the slice of numbers.
///
/// # Description
///
/// Computes the sum of numbers in the input slice.
///
/// # Arguments
///
/// * `data`: A slice of numbers.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `T`: The sum.
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
///
/// use imgal::statistics::sum;
///
/// // create a 1-dimensional array
/// let arr = [1.82, 3.35, 7.13, 9.25];
///
/// // compute the sum of the array
/// let total = sum(&arr, false);
///
/// assert_eq!(total, 21.55);
/// ```
#[inline(always)]
pub fn sum<'a, T, A, D>(data: A, parallel: bool) -> T
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    if parallel {
        data.into_par_iter()
            .fold(|| T::default(), |acc, &v| acc + v)
            .reduce(|| T::default(), |acc, v| acc + v)
    } else {
        data.iter().fold(T::default(), |acc, &v| acc + v)
    }
}
