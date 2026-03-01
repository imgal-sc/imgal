use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr};
use rayon::prelude::*;

use crate::traits::numeric::AsNumeric;

/// Compute the sum of an n-dimensional array using Kahan compensated summation.
///
/// # Description
///
/// Computes the Kahan sum of an n-dimensional array. The Kahan compensated
/// summation algorithm corrects for floating-point rounding errors and
/// precision loss at each step of the summation. To compensate for
/// floating-point error an error residual is subtracted from each value per
/// iteration.
///
/// # Arguments
///
/// * `data`: An n-dimensonal array of numeric values.
///
/// # Returns
///
/// * `T`: The Kahan sum.
pub fn kahan_sum<'a, T, A, D>(data: A) -> T
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    data.iter()
        .fold((T::default(), T::default()), |acc, &v| {
            let adj = v - acc.1;
            let new_sum = acc.0 + adj;
            let comp = (new_sum - acc.0) - adj;
            (new_sum, comp)
        })
        .0
}

/// Compute the sum of an n-dimensional array.
///
/// # Description
///
/// Computes the sum of numerical values in the data array.
///
/// # Arguments
///
/// * `data`: An n-dimensonal array of numeric values.
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
