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
pub fn sum<T>(data: &[T]) -> T
where
    T: AsNumeric,
{
    data.iter().fold(T::default(), |acc, &v| acc + v)
}
