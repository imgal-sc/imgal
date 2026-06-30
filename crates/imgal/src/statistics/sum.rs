use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr, Zip};

use crate::prelude::*;

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
        seq_exp: Zip::from(data).fold(T::default(), |acc, &v| acc + v),
        par_exp: Zip::from(data).par_fold(|| T::default(), |acc, &v| acc + v, |acc_a, acc_b| acc_a + acc_b))
}
