use ndarray::{ArrayBase, ArrayView, AsArray, Dimension, ViewRepr, Zip};
use rayon::prelude::*;

use crate::prelude::*;
use crate::simd_hint::unrolled_fold;

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
        seq_exp: fast_max(data, av),
        par_exp: Zip::from(data.rows()).into_par_iter()
            .fold(|| av, |acc, (r,)| {
                let m = fast_max(r, av);
                if m > acc { m } else { acc }
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
    T: 'a + PartialOrd + Clone + Sync,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let av = data
        .first()
        .ok_or(ImgalError::InvalidParameterEmptyArray { param_name: "data" })?;
    let min_cmp = |acc, v| if v < acc { v } else { acc };
    Ok(par!(threads,
        seq_exp: Zip::from(&data).fold(av, &min_cmp),
        par_exp: Zip::from(&data).par_fold(|| av, &min_cmp, &min_cmp))
    .clone())
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
    T: 'a + PartialOrd + Clone + Send + Sync,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let av = data
        .first()
        .ok_or(ImgalError::InvalidParameterEmptyArray { param_name: "data" })?;
    let mm_seq = || {
        let mm = Zip::from(&data).fold((av, av), |acc, v| {
            (
                if v < acc.0 { v } else { acc.0 },
                if v > acc.1 { v } else { acc.1 },
            )
        });
        (mm.0.clone(), mm.1.clone())
    };
    let mm_par = || {
        let mm = Zip::from(&data).par_fold(
            || (av, av),
            |acc, v| {
                (
                    if v < acc.0 { v } else { acc.0 },
                    if v > acc.1 { v } else { acc.1 },
                )
            },
            |acc, v| {
                (
                    if v.0 < acc.0 { v.0 } else { acc.0 },
                    if v.1 > acc.1 { v.1 } else { acc.1 },
                )
            },
        );
        (mm.0.clone(), mm.1.clone())
    };
    Ok(par!(threads, seq_exp: mm_seq(), par_exp: mm_par()))
}

/// TODO
#[inline(always)]
fn fast_max<T, D>(data: ArrayView<T, D>, init_value: T) -> T
where
    D: Dimension,
    T: AsNumeric,
{
    let max_op = |acc, v| if v > acc { v } else { acc };
    if let Some(s) = data.as_slice_memory_order() {
        unrolled_fold(s, || init_value, max_op)
    } else {
        data.rows().into_iter().fold(init_value, |acc, r| {
            if let Some(s) = r.as_slice_memory_order() {
                let m = unrolled_fold(s, || init_value, max_op);
                if m > acc { m } else { acc }
            } else {
                let m = r
                    .iter()
                    .fold(init_value, |acc, &v| if v > acc { v } else { acc });
                if m > acc { m } else { acc }
            }
        })
    }
}

/// TODO
#[inline(always)]
fn fast_min<T, D>(data: ArrayView<T, D>, init_value: T) -> T
where
    D: Dimension,
    T: AsNumeric,
{
    let min_op = |acc, v| if v < acc { v } else { acc };
    if let Some(s) = data.as_slice_memory_order() {
        unrolled_fold(s, || init_value, min_op)
    } else {
        data.rows().into_iter().fold(init_value, |acc, r| {
            if let Some(s) = r.as_slice_memory_order() {
                let m = unrolled_fold(s, || init_value, min_op);
                if m < acc { m } else { acc }
            } else {
                let m = r
                    .iter()
                    .fold(init_value, |acc, &v| if v < acc { v } else { acc });
                if m < acc { m } else { acc }
            }
        })
    }
}
