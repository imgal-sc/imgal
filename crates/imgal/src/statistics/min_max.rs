use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr, Zip};

use crate::prelude::*;

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
    T: 'a + PartialOrd + Clone + Sync + Send,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let av = data
        .first()
        .ok_or(ImgalError::InvalidParameterEmptyArray { param_name: "data" })?;
    let max_cmp = |acc, v| if v > acc { v } else { acc };
    Ok(par!(threads,
        seq_exp: Zip::from(&data).fold(av, &max_cmp),
        par_exp: Zip::from(&data).par_fold(|| av, &max_cmp, &max_cmp))
    .clone())
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
