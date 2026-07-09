use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr, Zip};
use rayon::prelude::*;

use crate::prelude::*;
use crate::simd_hint::fast_fold;

/// Find the maximum value in an n-dimensional image.
///
/// # Description
///
/// Iterates through all elements of an n-dimensional image to determine the
/// maximum value.
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
/// * `Ok(T)`: The maximum value in the input n-dimensional image.
/// * `Err(ImgalError)`: If `data.is_empty() == true`.
#[inline]
pub fn max<'a, T, A, D>(data: A, threads: Option<usize>) -> Result<T, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let max_op = |acc, v| if v > acc { v } else { acc };
    let av = *data
        .first()
        .ok_or(ImgalError::InvalidParameterEmptyArray { param_name: "data" })?;
    Ok(par!(threads,
        seq_exp: fast_fold(data, || av, max_op),
        par_exp: Zip::from(data.rows()).into_par_iter()
            .fold(|| av, |acc, (r,)| {
                let m = fast_fold(r, || av, max_op);
                max_op(acc, m)
            }).reduce(|| av, max_op)))
}

/// Find the minimum value in an n-dimensional image.
///
/// # Description
///
/// Iterates through all elements of an n-dimensional image to determine the
/// minimum value.
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
/// * `Ok(T)`: The minimum value in the input n-dimensional image.
/// * `Err(ImgalError)`: If `data.is_empty() == true`.
#[inline]
pub fn min<'a, T, A, D>(data: A, threads: Option<usize>) -> Result<T, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let min_op = |acc, v| if v < acc { v } else { acc };
    let av = *data
        .first()
        .ok_or(ImgalError::InvalidParameterEmptyArray { param_name: "data" })?;
    Ok(par!(threads,
        seq_exp: fast_fold(data, || av, min_op),
        par_exp: Zip::from(data.rows()).into_par_iter()
            .fold(|| av, |acc, (r,)| {
                let m = fast_fold(r, || av, min_op);
                min_op(acc, m)
            }).reduce(|| av, min_op)))
}

/// Find the minimum and maximum values in an n-dimensional image.
///
/// # Description
///
/// Iterates through all elements of an n-dimensional image to determine the
/// minimum and maximum values.
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
/// * `Ok((T, T))`: A tuple containing the minimum and maximum values (*i.e.*
///   (min, max)) in the given n-dimensional image.
/// * `Err(ImgalError)`: If `data.is_empty() == true`.
#[inline]
pub fn min_max<'a, T, A, D>(data: A, threads: Option<usize>) -> Result<(T, T), ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let max_op = |acc, v| if v > acc { v } else { acc };
    let min_op = |acc, v| if v < acc { v } else { acc };
    let min_max_op = |acc: (T, T), v: (T, T)| {
        (
            if v.0 < acc.0 { v.0 } else { acc.0 },
            if v.1 > acc.1 { v.1 } else { acc.1 },
        )
    };
    let av = *data
        .first()
        .ok_or(ImgalError::InvalidParameterEmptyArray { param_name: "data" })?;
    Ok(par!(threads,
        seq_exp: Zip::from(data.rows()).fold((av, av), |acc, r| {
            let res = (fast_fold(r, || av, min_op), fast_fold(r, || av, max_op));
            min_max_op(acc, res)
            }),
        par_exp: Zip::from(data.rows()).into_par_iter()
            .fold(|| (av, av), |acc, (r,)| {
                let res = (fast_fold(r, || av, min_op), fast_fold(r, || av, max_op));
                min_max_op(acc, res)
            }).reduce(|| (av, av), min_max_op)))
}
