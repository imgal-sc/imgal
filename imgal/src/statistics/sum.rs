use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr};

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
/// let total = sum(&arr);
///
/// assert_eq!(total, 21.55);
/// ```
#[inline(always)]
pub fn sum<'a, T, A, D>(data: A) -> T
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();

    view.iter().fold(T::default(), |acc, &v| acc + v)
}
