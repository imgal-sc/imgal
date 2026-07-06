use ndarray::{ArrayBase, ArrayView, AsArray, Dimension, ViewRepr, Zip};
use rayon::prelude::*;

use crate::prelude::*;
use crate::simd_hint::unrolled_fold;

/// Compute the sum of an n-dimensional image using Kahan compensated summation.
///
/// # Description
///
/// Computes the Kahan sum of an n-dimensional image. The Kahan compensated
/// summation algorithm corrects for floating-point rounding errors and
/// precision loss at each step of the summation. To compensate for
/// floating-point error an error residual is subtracted from each value per
/// iteration.
///
/// # Arguments
///
/// * `data`: The input n-dimensional image.
///
/// # Returns
///
/// * `Ok(T)`: The Kahan sum.
/// * `Err(ImgalError)`: If `data.is_empty() == true`.
#[inline]
pub fn kahan_sum<'a, T, A, D>(data: A) -> Result<T, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    if data.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray { param_name: "data" });
    }
    Ok(data
        .iter()
        .fold((T::default(), T::default()), |acc, &v| {
            let adj = v - acc.1;
            let new_sum = acc.0 + adj;
            let comp = (new_sum - acc.0) - adj;
            (new_sum, comp)
        })
        .0)
}

/// Compute the sum of an n-dimensional image.
///
/// # Description
///
/// Computes the sum of numerical values in an n-dimensional image.
///
/// # Arguments
///
/// * `data`: The input n-dimensional image.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
///
/// # Returns
///
/// * `T`: The sum.
///
/// # Example
///
/// ```
/// use ndarray::Array1;
///
/// use imgal::statistics::sum;
///
/// let arr = [1.82, 3.35, 7.13, 9.25];
/// let total = sum(&arr, None);
/// assert_eq!(total, 21.55);
/// ```
#[inline]
pub fn sum<'a, T, A, D>(data: A, threads: Option<usize>) -> T
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    par!(threads,
        seq_exp: fast_sum(data),
        par_exp: Zip::from(data.rows())
            .into_par_iter()
            .fold(T::default, |acc, (r,)| acc + fast_sum(r))
            .reduce(T::default, T::add))
}

/// Compute the sum of an n-dimensional array using a manually unrolled fold
/// loop.
///
/// # Description
///
/// This function manually computes the sum of an n-dimensional array using a
/// special unrolled fold helper function that hints to the compiler to
/// autovectorize the sum. For `unrolled_fold` requires the data be in a 
/// contiguous memory order.
///
/// # Arguments
///
/// * `data`: The input n-dimensional array.
///
/// # Returns
///
/// * `T`: The sum of the n-dimensional array.
#[inline(always)]
fn fast_sum<T, D>(data: ArrayView<T, D>) -> T
where
    D: Dimension,
    T: AsNumeric,
{
    if let Some(s) = data.as_slice_memory_order() {
        unrolled_fold(s, T::default, T::add)
    } else {
        data.rows().into_iter().fold(T::default(), |acc, r| {
            if let Some(s) = r.as_slice_memory_order() {
                acc + unrolled_fold(s, T::default, T::add)
            } else {
                acc + r.iter().fold(T::default(), |acc, &v| acc + v)
            }
        })
    }
}
